# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
import copy
from typing import List, Tuple, Dict, Callable
from argparse import Namespace

import sentencepiece as spm
import fastBPE

import torch
from torch.distributions import Categorical
import numpy as np

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.data import Dictionary
from fairseq.sequence_generator import SequenceGenerator
from fairseq.criterions.helper_functions import collate_tokens
from fairseq.data import encoders


@dataclass
class ImitKDConfig(FairseqDataclass):
    expert: str = field(
        default="checkpoint_best.pt",
        metadata={"help": "NMT model to use as expert"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    beta: int = field(
        default=1,
        metadata={"help": "replacement prop"},
    )

    expert_vocab_tgt: str = field(
        default="wmt19.en-de.joined-dict.ensemble/dict.de.txt",
        metadata={"help": "vocab for nmt model output"},
    )
    expert_vocab_src: str = field(
        default="wmt19.en-de.joined-dict.ensemble/dict.en.txt",
        metadata={"help": "vocab for nmt model input"},
    )
    path: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/wmt19.en-de.joined-dict.ensemble/",
        metadata={"help": "directory with expert's dictionaries"},
    )
    bpe_codes: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/examples/speech_to_text/bpecodes",
        metadata={"help": "expert's bpe codes"},
    )
    eval_bleu: bool = field(
        default=False,
        metadata={"help": "hack - pass flag if best checkpoint metric is bleu to compute BLEU score"}
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
                    "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )



def imit_kd_loss(
        generated_dataset,
        model,
        expert,
        source_text,
        ignore_index,
        source_lengths
):
    """
    ImitKD-optimal: Calculates negative log-likelihood loss of  expert's predictions

    Args:
        generated_dataset: dataset batch with the reference translations
        model: the student model that is trained
        expert: the NMT expert
        source_text: the source text as subword units to give the NMT model as input
        ignore_index: index of pad sign in vocabulary
        source_lengths: lengths of the source texts
    Returns:
        negative log-likelihood loss of  expert's argmax predictions
    """
    sample_expert = {
        "id": generated_dataset["id"],
        "net_input": {
            "src_tokens": source_text.cuda(),
            "src_lengths": source_lengths,
            "prev_output_tokens": generated_dataset["net_input"]["prev_output_tokens"].cuda()
        },
        "target": generated_dataset["target"],
        "ntokens": generated_dataset["ntokens"],
        "nsentences": generated_dataset["nsentences"],
    }
    with torch.no_grad():
        expert_logits = expert.get_normalized_probs(expert(**sample_expert["net_input"]), log_probs=True).detach()
        preds = expert_logits.argmax(-1)
        preds = preds.to(torch.int64).cuda()

    generated_dataset["net_input"].pop("src_text", None)
    lprobs = model.get_normalized_probs(model(**generated_dataset["net_input"]), log_probs=True)
    if preds.dim() == lprobs.dim() - 1:
        preds = preds.unsqueeze(-1)
    # if preds.shape[1] > lprobs.shape[1]:
        # preds = preds[:, :lprobs.shape[1], :]
    imit_kd_loss_for_sample = -lprobs.gather(dim=-1, index=preds)
    if ignore_index is not None:
        pad_mask = sample_expert["net_input"]["prev_output_tokens"].unsqueeze(-1).eq(ignore_index)
        imit_kd_loss_for_sample.masked_fill_(pad_mask, 0.0)
    imit_kd_loss_for_sample = imit_kd_loss_for_sample.sum()
    return imit_kd_loss_for_sample

def valid_loss(lprobs, target, sample, model, tgt_dict, eval_bleu=False, ignore_index=None, reduce=True,
               tokenizer=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()

    # adapted from translation task
    logging_output = {}
    if eval_bleu:
        EVAL_BLEU_ORDER = 4
        logging_output = {}
        sequence_generator = SequenceGenerator([model], tgt_dict, beam_size=5)
        bleu = inference_with_bleu(sequence_generator, sample, model, tgt_dict, tokenizer)
        logging_output["_bleu_sys_len"] = bleu.sys_len
        logging_output["_bleu_ref_len"] = bleu.ref_len
        # we split counts into separate entries so that they can be
        # summed efficiently across workers using fast-stat-sync
        assert len(bleu.counts) == EVAL_BLEU_ORDER
        for i in range(EVAL_BLEU_ORDER):
            logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
            logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]

    return nll_loss, logging_output

# adapted from fairseq method _inference_with_bleu of translation task; 95% the same
def inference_with_bleu(generator, sample, model, tgt_dict, tokenizer=None):
    import sacrebleu

    def decode(toks, escape_unk=False):
        s = tgt_dict.string(
            toks.int().cpu(),
            # The default unknown string in fairseq is `<unk>`, but
            # this is tokenized by sacrebleu as `< unk >`, inflating
            # BLEU scores. Instead, we use a somewhat more verbose
            # alternative that is unlikely to appear in the real
            # reference, but doesn't get split into multiple tokens.
            unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            bpe_symbol="fastBPE"
        )
        if tokenizer:
            s = tokenizer.decode(s)
        return s

    gen_out = generator.generate([model], sample, prefix_tokens=None)
    hyps, refs = [], []
    for i in range(len(gen_out)):
        hyps.append(decode(gen_out[i][0]["tokens"]))
        refs.append(
            decode(
                utils.strip_pad(sample["target"][i], tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            )
        )
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")


@register_criterion(
    "imit_kd", dataclass=ImitKDConfig
)
class ImitKD(FairseqCriterion):
    def __init__(
            self,
            task,
            expert,
            expert_vocab_src,
            expert_vocab_tgt,
            path,
            beta,
            bpe_codes,
            eval_bleu,
            eval_bleu_detok,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task)
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.expert, _ = load_model_ensemble([expert], arg_overrides={"data": path})
        self.expert = self.expert[-1]
        self.expert = self.expert.eval()
        self.expert_vocab_src = Dictionary.load(expert_vocab_src)
        self.expert_vocab_tgt = Dictionary.load(expert_vocab_tgt)
        self.expert.requires_grad = False
        self.dict = task.tgt_dict
        self.eos = self.dict.eos()
        self.bpe = fastBPE.fastBPE(bpe_codes, expert_vocab_tgt)
        self.pad_idx = self.padding_idx
        self.sentence_avg = False
        self.beta = beta
        self.eval_bleu = eval_bleu
        if self.eval_bleu:
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=eval_bleu_detok)
            )
        else:
            self.tokenizer = None


    def forward(self, model, sample, reduce=True, valid=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_s = copy.deepcopy(sample)
        sample_s["net_input"].pop("src_text", None)
        net_output = model(**sample_s["net_input"])
        loss, logged_bleu = self.compute_loss(model, net_output, sample, reduce=reduce, valid=valid)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        if logged_bleu:
            logging_output = {
                "loss": loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
                "bleu": logged_bleu,
            }
        else:
            logging_output = {
                "loss": loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs, target

    def compute_loss(self, model, net_output, sample, reduce=True, valid=False):
        if valid:
            sample["net_input"].pop("src_text", None)
            lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
            loss, logged_bleu = valid_loss(lprobs, target, sample, model, self.dict, self.eval_bleu, self.padding_idx, reduce=reduce, tokenizer=self.tokenizer)
        else:
            source_text, source_lengths = self.transform_source_tokens_into_expert_voc(sample)
            sample_s = copy.deepcopy(sample)
            sample_s["net_input"].pop("src_text", None)
            generated_dataset = self.generate_imit_batch(model, sample_s)
            loss = imit_kd_loss(
                generated_dataset,
                model,
                self.expert,
                source_text,
                self.padding_idx,
                source_lengths
            )
            logged_bleu = None
        return loss, logged_bleu



    def generate_imit_batch(self, student: Callable, sample: Dict) -> Dict:
        """
        Uses student model to generate hypothesis according to beta

        Args:
            student: model to train
            sample: dataset batch
        Returns:
            dataset batch with prev_output_tokens == student hypothesis if beta_i = 1
        """
        with torch.no_grad():
            student = student.eval()
            targets = sample["net_input"]["prev_output_tokens"].data.tolist()
            max_length = max([len(i) for i in targets])  # let's avoid blowing up the GPU RAM, shall we?
            student_generator = SequenceGenerator([student], self.dict, beam_size=1, max_len=max_length)
            student_generator.cuda()
            hypos = student_generator._generate(sample)
            dist = Categorical(torch.tensor([self.beta, 1 - self.beta]))
            samp_mask = [dist.sample((sample["net_input"]["prev_output_tokens"].size(0),)) == 1][0]
            for i, h in enumerate(hypos):
                if samp_mask[i]:
                    if h[0]["tokens"][-1] != self.dict.eos():
                        targets[i] = torch.tensor([self.dict.eos()] + h[0]["tokens"].tolist())
                    else:
                        hypo = h[0]["tokens"].tolist()
                        targets[i] = torch.tensor([hypo[-1]] + hypo[0:-1])
                else:
                    targets[i] = torch.tensor(targets[i])
            sample["net_input"]["prev_output_tokens"] = collate_tokens(
                targets,
                self.dict.pad(),
                self.dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False
            ).cuda()
            student.train()
        return sample

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        target = target.view(-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
        # adapted from translation task
        if "bleu" in logging_outputs[0].keys():
            EVAL_BLEU_ORDER = 4

            def sum_logs(key):
                import torch
                result = sum(log["bleu"].get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))
            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def transform_source_tokens_into_expert_voc(
            self,
            sample: Dict,
            eos_at_beginning: bool = False
    ) -> Tuple[torch.IntTensor, List[int]]:
        """
        Turns the tokenized source text into bpe encodings.
        Args:
            sample: dataset batch
            eos_at_beginning: whether to put EOS token at the beginning of each sample (required for previous output tokens)
        Returns:
            Tuple of bpe encoded source text and list of integers determing the number of bp for each sample
        """
        if "src_text" not in sample["net_input"]:
            return sample["net_input"]["src_tokens"], [0]
        source_text = sample["net_input"]["src_text"]
        source_texts = []
        for line in source_text:
            if isinstance(line, list):
                for text in line:
                    source_texts.append(self.expert_vocab_src.encode_line(
                        self.bpe.apply([text])[0],
                        add_if_not_exist=False,
                        append_eos=True)
                    )
            else:
                source_texts.append(self.expert_vocab_src.encode_line(
                    self.bpe.apply([line])[0],
                    add_if_not_exist=False,
                    append_eos=True)
                )
        src_lengths = [len(text) for text in source_texts]
        source_text = collate_tokens(
            source_texts,
            self.expert_vocab_src.pad(),
            self.expert_vocab_src.eos(),
            left_pad=False,
            move_eos_to_beginning=eos_at_beginning
        )
        return source_text, src_lengths


