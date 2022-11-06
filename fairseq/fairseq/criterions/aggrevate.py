# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import copy
from dataclasses import dataclass, field
import json
from argparse import Namespace
from typing import List, Tuple, Dict
from torch.distributions import Categorical

import torch
import fastBPE
import numpy as np
import sacrebleu

from fairseq import metrics, utils
from fairseq.scoring import bleu
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.data import Dictionary
from fairseq.sequence_generator import SequenceGenerator
from fairseq.data import encoders
from fairseq.criterions.helper_functions import collate_tokens

@dataclass
class AggrevateConfig(FairseqDataclass):
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
    expert_action_chosen: bool = field(
        default=False,
        metadata={"help": "whether to take expert's action or  sample action uniformly"},
    )
    bpe_codes: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/examples/speech_to_text/bpecodes",
        metadata={"help": "expert's bpe codes"},
    )
    expert_generate: bool = field(
        default=False,
        metadata={"help": "whether to take expert's hypothesis or take dataset as expert demonstrations"},
    )
    no_data_mix: bool = field(
        default=False,
        metadata={"help": "whether to use only student hypotheses instead of relying on a expert/student data mixture"}
    )
    sample_action_prob: float = field(
        default=0.1,
        metadata={"help": "probability of either sampling the action uniformly (best-or-uniform) or taking the "
                          "expert's optimal action (best-or-expert)"}
    )
    eval_bleu: bool = field(
        default=False,
        metadata={"help": "hack - pass flag if best checkpoint metric is bleu to compute BLEU score"}
    )
    eval_bleu_args: str = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok_args: str = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )


def valid_loss(lprobs, target, sample, model, tgt_dict, eval_bleu=False, ignore_index=None, reduce=True, tokenizer=None):
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
    if eval_bleu:
        EVAL_BLEU_ORDER = 4
        logging_output = {}
        sequence_generator = SequenceGenerator([model], tgt_dict, beam_size=1)
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


# adapted from fairseq method _inference_with_bleu of translation task
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


def knn_forced_loss(bleu_diff, rs_student, indicator):
    #  reshaping unneeded, but makes debugging easier
    rs_student = rs_student.view(-1, 1)
    bleu_diff = bleu_diff.view(-1, 1)
    indicator = indicator.view(-1, 1)
    loss = (indicator * (rs_student - bleu_diff) ** 2).sum()
    #print(bleu_diff.flatten(), rs_student.flatten())
    # print(reward_expert.sum(), reward_student.sum())
    return loss, bleu_diff.sum(), rs_student.sum(), indicator.sum()


def get_action_probs_and_rewards(net_output, expert_reward_to_go, indices, current_reward, ats):
    rse = []
    for i, tensor in enumerate(net_output[0]):
        # why 8? -> https://math.stackexchange.com/questions/4242042/stretch-sigmoid-along-horizontal-x-interval
        max_value = torch.max(tensor[indices[i]])
        min_value = torch.min(tensor[indices[i]])
        scaled_tensor = 1 / (1 + torch.exp(-8 * (2 * tensor[indices[i], ats[i]] - abs(min_value + max_value)) / abs(min_value - max_value)))
        #scaled_tensor_method2 = torch.sigmoid(
         #   8*(tensor[indices[i], ats[i]] - (min_value + abs(min_value - max_value) / 2)) /
          #  abs(min_value - max_value)
        #)
        #print(scaled_tensor, scaled_tensor_method2)
        rse.append(scaled_tensor)
    rse = torch.stack(rse, dim=0)
    r_e = torch.tensor([bleu / 100 for bleu in expert_reward_to_go], device="cuda")
    r_s_b = torch.tensor([bleu / 100 for bleu in current_reward], device="cuda")
    indicator = [r_e > r_s_b][0]
    bleu_diff = r_e - r_s_b
    return rse, bleu_diff, indicator


@register_criterion(
    "aggrevate", dataclass=AggrevateConfig
)
class Aggrevate(FairseqCriterion):
    def __init__(
            self,
            task,
            expert,
            expert_vocab_src,
            expert_vocab_tgt,
            path,
            bpe_codes,
            beta,
            expert_action_chosen,
            expert_generate,
            no_data_mix,
            sample_action_prob,
            eval_bleu,
            eval_bleu_args,
            eval_bleu_detok_args,
            eval_bleu_detok,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task)
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.expert, _ = load_model_ensemble([expert], arg_overrides={"data": path})
        self.expert = self.expert[-1]
        self.expert_vocab_src = Dictionary.load(expert_vocab_src)
        self.expert_vocab_tgt = Dictionary.load(expert_vocab_tgt)
        self.expert.requires_grad = False
        self.dict = task.tgt_dict
        self.eos = self.dict.eos()
        self.bpe = fastBPE.fastBPE(bpe_codes, expert_vocab_tgt)
        self.sentence_avg = True
        self._scorer = BleuScorer(self.padding_idx, self.eos, self.dict.unk())
        self.beta = beta
        self.expert_action_chosen = expert_action_chosen
        self.eval_bleu = eval_bleu
        self.eval_bleu_args = eval_bleu_args
        self.eval_bleu_detok_args = eval_bleu_detok_args
        if  self.expert_action_chosen:
            self.uniform_sampling = False
        else:
            self.uniform_sampling = True
        if not self.expert_action_chosen:
            self.random_action_distribution = torch.distributions.Categorical(
                probs=torch.tensor([1.0 for a in self.dict.indices.values() if a != self.dict.pad() and a!= self.dict.eos() and a != self.dict.unk()])
            )
        self.gen = np.random.default_rng()
        self.expert_generate = expert_generate
        self.data_mix = no_data_mix
        self.sample_action_prob = sample_action_prob
        self.eval_bleu_detok = eval_bleu_detok
        if self.eval_bleu:
            detok_args = json.loads(self.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.eval_bleu_detok, **detok_args)
            )

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
        # doesn't occur in translation task: that has its own valid step - this hacks into fairseq valid step of speech2text
        # instead
        # otherwise valid includes running the entire loss step, including generating hypos, continuing hypos, etc...
        # for ImitKD we compare dev performance with NLL against target, not NLL against expert predicts
        # -> hack2 to avoid breaking hack1
        if valid:
            loss, logged_bleu = self.compute_loss(model, net_output, sample, reduce=reduce, valid=valid)
        else:
            loss, bleu_diff, r_student, indicator_sum = self.compute_loss(model, net_output, sample, reduce=reduce, valid=valid)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        if valid:
            logging_output = {
                "loss": loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample["ntokens"],
                "bleu": logged_bleu
            }
        else:
            logging_output = {
                "loss": loss.data,
                "bleu_diff": bleu_diff.data,
                "r_student": r_student.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
                "indicator_sum": indicator_sum
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
            sample["net_input"].pop("src_text", None)
            loss, logged_bleu = valid_loss(lprobs, target, sample, model, self.expert_vocab_tgt, self.eval_bleu,
                                           self.ignore_prefix_size, reduce=reduce, tokenizer=self.tokenizer)
            return loss, logged_bleu
        else:
            with torch.no_grad():
                source_text, _ = self.transform_source_tokens_into_expert_voc(sample)
                expert_input, indices, ats = self.get_student_predictions_and_pass_to_expert(model, sample, source_text)
                expert_output_samples = self.get_expert_output(sample, source_text, expert_input)
            model.train()  # call to SequenceGenerator sets model to eval mode, set model back to train
            bleu_diff, rs_student, indicator = self.calculate_reward_and_reward_to_go(sample, model,
                                                                                      expert_output_samples,
                                                                                      expert_input, indices, ats)
            loss, r_expert, r_student, indicator = knn_forced_loss(
                bleu_diff, rs_student, indicator
            )
        return loss, r_expert, r_student, indicator

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
        # ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        bleu_diff = sum(log.get("bleu_diff", 0) for log in logging_outputs)
        r_student = sum(log.get("r_student", 0) for log in logging_outputs)
        indicator_sum = sum(log.get("indicator_sum", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "bleu_diff", bleu_diff / sample_size, sample_size, round=3
        )
        if indicator_sum > 0:
            metrics.log_scalar(
                "bleu_diff_no_zeros", bleu_diff / indicator_sum, sample_size, round=3
            )
        metrics.log_scalar(
            "r_student", r_student / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "indicator_percentage", indicator_sum / sample_size, sample_size, round=3
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


    def calculate_bleu(self, target, hypos):
        bleu_scores = []
        for i, hypo in enumerate(hypos):
            ref = utils.strip_pad(target[i, :], self.padding_idx).cpu()
            #ref = self.tokenizer.decode(self.dict.string(utils.strip_pad(target[i, :], self.padding_idx).cpu(), unk_string="UNKNOWNTOKENINREF",
             #       bpe_symbol="fastBPE"))
            # only top scoring hypothesis is considered
           # hyp = self.tokenizer.decode(self.dict.string(hypo[0]["tokens"].int().cpu(), unk_string="UNKNOWNTOKENINHYP", bpe_symbol="fastBPE"))
            # not happy with this - corpus-level metric on sentence-level - try chRF?
            #bleu = sacrebleu.sentence_bleu(hyp, [ref], smooth_method="floor", smooth_value=0.1)     # not smoothing sets BLEU too low, smoothing exp. sets it far too high
            #self._scorer.score(utils.strip_pad(target[i, :], self.padding_idx).cpu()  , hypo[0]['tokens'].int().cpu(), order=self.bleu_ngramms)
            bleu_scores.append(self._scorer.score(ref, hypo[0]['tokens'].int().cpu())) # decreases training time by ca. 1/3, not as precise but works just as good -> useful despite drawbacks
            #bleu_scores.append(bleu.score)
        return bleu_scores

    def get_student_predictions_and_pass_to_expert(self, model, sample, source_text):
        hypo_including_t = []
        student_sample = copy.deepcopy(sample)
        student_sample["net_input"].pop("src_text", None)
        hypos_in = sample["target"].data.tolist()
        max_length = max([len(i) for i in hypos_in])  # avoid OOM
        self.beta = 0.005
        with torch.no_grad():
            if self.data_mix:
                expert_generator = SequenceGenerator([self.expert], self.dict, beam_size=1, max_len=max_length)
                expert_generator.cuda()
                dist = Categorical(torch.tensor([self.beta, 1 - self.beta]))
                data_replacement_samp_mask = [dist.sample((sample["net_input"]["prev_output_tokens"].size(0),)) == 1][0]
                if self.expert_generate:
                    expert_sample = copy.deepcopy(sample)
                    expert_sample["net_input"]["src_tokens"] = source_text.cuda()
                    expert_sample["net_input"].pop("src_text", None)
                    expert_hypos = expert_generator._generate(expert_sample)
                else:
                    expert_hypos = sample["target"]
            else:
                data_replacement_samp_mask = [True for _ in sample["net_input"]["prev_output_tokens"]]
            student_generator = SequenceGenerator([model], self.dict, beam_size=1, max_len=max_length)
            if self.uniform_sampling:
                dist = Categorical(torch.tensor([1 - self.sample_action_prob, self.sample_action_prob]))
                action_sampling_mask = [dist.sample((sample["net_input"]["prev_output_tokens"].size(0),)) == 1][0]
            student_generator.cuda()
            hypos = student_generator._generate(student_sample)
            indices = []
            eos_pad = torch.tensor([self.eos], device="cuda")
            for i, h in enumerate(hypos):
                if data_replacement_samp_mask[i]:
                    index = torch.distributions.Categorical(probs=torch.tensor([1.0 for _ in range(h[0]["tokens"].shape[0])])).sample().cuda()
                    indices.append(index)
                    hypos_in[i] = torch.cat((h[0]["tokens"][:index], eos_pad), dim=0)
                else:
                    if self.expert_generate:
                        index = torch.distributions.Categorical(
                            probs=torch.tensor([1.0 for _ in range(expert_hypos[i][0]["tokens"].shape[0])])).sample().cuda()
                        indices.append(index)
                        hypos_in[i] = torch.cat((expert_hypos[i][0]["tokens"][:index], eos_pad), dim=0)
                    else:
                        index = torch.distributions.Categorical(
                            probs=torch.tensor([1.0 for _ in range((hypos_in[i][0]["tokens"].shape[0]))])
                        ).sample().cuda()
                        indices.append(index)
                        hypos_in[i] = torch.cat((utils.strip_pad(torch.tensor(hypos_in[i], device="cuda"), self.padding_idx)[:index]), dim=0)
            hypos_up_to_t = collate_tokens(
                hypos_in,
                self.expert_vocab_src.pad(),
                self.expert_vocab_src.eos(),
                left_pad=False,
                move_eos_to_beginning=True
            )

            if self.expert_action_chosen:
                out = {
                    "id": sample["id"],
                    "net_input": {
                        "src_tokens": source_text.cuda(),
                        "src_lengths": [len(text) for text in source_text],
                        "prev_output_tokens": hypos_up_to_t.cuda(),
                    },
                    "target": sample["target"],
                    "target_lengths": sample["target_lengths"],
                    "ntokens": sample["ntokens"],
                    "nsentences": sample["nsentences"],
                }
                expert_output = self.expert.get_normalized_probs(self.expert(**out["net_input"]),
                                                                 log_probs=True).argmax(dim=-1)
            else:
                student_sample["net_input"]["prev_output_tokens"] = hypos_up_to_t.cuda()
                model_output = model.get_normalized_probs(model(**student_sample["net_input"]), log_probs=True).argmax(
                    dim=-1)
            ats = []
            for i, hypo in enumerate(hypos_in):
                if action_sampling_mask[i]:
                    a_t = self.random_action_distribution.sample().cuda()
                elif self.expert_action_chosen:
                    a_t = expert_output[i][indices[i]]
                else:
                    a_t = model_output[i][indices[i]]
                if not isinstance(hypo, torch.Tensor):
                    hypo = torch.tensor(hypo, device="cuda")
                if hypo.shape[0] > 1:
                    hypo_including_t.append(torch.cat((hypo[:-1], a_t.unsqueeze(dim=0)), dim=0))
                else:       # one element tensor
                    hypo_including_t.append(torch.cat((hypo, a_t.unsqueeze(dim=0)), dim=0))
                ats.append(a_t)
            hypos_to_t = collate_tokens(
                hypo_including_t,
                self.expert_vocab_src.pad(),
                self.expert_vocab_src.eos(),
                left_pad=False,
                move_eos_to_beginning=False
            )
        return hypos_to_t, indices, ats

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

    def get_expert_output(self, sample, source_texts, expert_input):
        if "target_lengths" in sample.keys():
            out = {
                "id": sample["id"],
                "net_input": {
                    "src_tokens": source_texts.cuda(),
                    "src_lengths": [len(text) for text in source_texts],
                    "prev_output_tokens": [],
                },
                "target": sample["target"],
                "target_lengths": sample["target_lengths"],
                "ntokens": sample["ntokens"],
                "nsentences": sample["nsentences"],
            }
        else:
            out = copy.deepcopy(sample)
        expert_generator = SequenceGenerator(
            [self.expert],
            self.expert_vocab_tgt,
            beam_size=1,
        )
        expert_output = expert_generator.generate(
            [self.expert],
            out,
            prefix_tokens=expert_input.cuda()
        )
        return expert_output

    def calculate_reward_and_reward_to_go(self, sample, model, expert_output_samples, expert_input, indices, ats):
        eos_pad = torch.tensor([self.dict.eos() for _ in range(expert_input.shape[0])], device="cuda").view(-1, 1)
        sample["net_input"]["prev_output_tokens"] = torch.cat((eos_pad, expert_input), dim=1).cuda()
        sample["net_input"].pop("src_text", None)
        targets = sample['target'].data.int()
        net_output = model(**sample["net_input"])
        # ugly hack to avoid rewriting calculate_bleu method TODO: rewrite method and delete ugly hack
        student_hypos = [[{"tokens": utils.strip_pad(expert_input_sample, self.dict.pad())}] for expert_input_sample in expert_input.cpu()]
        r_at_t = self.calculate_bleu(targets, student_hypos)
        rtg = self.calculate_bleu(targets, expert_output_samples)
        rs_student, bleu_diff, indicator = get_action_probs_and_rewards(net_output, rtg, indices,
                                                                        r_at_t, ats)
        return bleu_diff, rs_student, indicator


# adapted from https://github.com/jwieting/beyond-bleu/blob/master/fairseq/criterions/fairseq_sequence_criterion.py
class BleuScorer(object):

    def __init__(self, pad, eos, unk):
        from collections import namedtuple
        cfg_f = namedtuple('cfg_f', ['pad', 'eos', 'unk'])
        cfg = cfg_f(pad, eos, unk)
        self._scorer = bleu.Scorer(cfg)


    def score(self, ref, hypo):
        self._scorer.reset(one_init=True)
        self._scorer.add(ref, hypo)
        return self._scorer.score()
