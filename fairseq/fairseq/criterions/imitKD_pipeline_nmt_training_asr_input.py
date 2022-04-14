# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
import copy

from torch.distributions import Categorical
import torch
import fastBPE

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.data import Dictionary
from fairseq.sequence_generator import SequenceGenerator
from fairseq.scoring import bleu


@dataclass
class ImitKD_pipeline_training_asr_inputConfig(FairseqDataclass):
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
    asr_model: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/europarl_asr.pt",
        metadata={"help": "asr model"}
    )
    path_asr_model: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/v1.1",
        metadata={"help": "path to ASR model"}
    )
    bpe_codes: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/examples/speech_to_text/bpecodes",
        metadata={"help": "expert's bpe codes"},
    )


def valid_loss(lprobs, target, ignore_index=None, reduce=True):
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
    return nll_loss


def imit_kd_loss(
        generated_dataset,
        source_texts,
        model,
        expert,
        source_lengths,
        ignore_index,
):
    sample_expert = copy.deepcopy(generated_dataset)
    with torch.no_grad():
        expert_logits = expert.get_normalized_probs(expert(**sample_expert["net_input"]), log_probs=False)
        pad_mask = expert_logits.eq(ignore_index)
        expert_logits.masked_fill_(pad_mask, 0.0)
    lprobs = model.get_normalized_probs(model(**generated_dataset["net_input"]), log_probs=True)
    return -torch.sum(expert_logits * lprobs)


@register_criterion(
    "imit_pipeline_nmt_training_asr_input", dataclass=ImitKD_pipeline_training_asr_inputConfig
)
class ImitKD_pipeline_training_asr_input(FairseqCriterion):
    def __init__(
            self,
            task,
            expert,
            expert_vocab_src,
            expert_vocab_tgt,
            path,
            asr_model,
            path_asr_model,
            beta,
            bpe_codes,
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
        self.asr_model, _ = load_model_ensemble([asr_model], arg_overrides={"data": path_asr_model, "encoder_freezing_updates": 0})
        self.asr_model = self.asr_model[-1]
        self.asr_model = self.asr_model.eval()

    def forward(self, model, sample, reduce=True, valid=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if valid:
            sample["net_input"].pop("src_text", None)
            loss = self.compute_loss(model, sample, reduce=reduce, valid=valid)
        else:
            source_text, source_lengths = self.transform_source_tokens_into_expert_voc(sample)
            sample = self.generate_imit_batch(model, sample)
            sample_s = copy.deepcopy(sample)
            sample_s["net_input"].pop("src_text", None)
            net_output = model(**sample_s["net_input"])
            loss = self.compute_loss(model, net_output, sample, source_text, source_lengths, reduce=reduce, valid=valid)
        sample_size = (
           sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

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

    def generate_imit_batch(self, student, sample):
        with torch.no_grad():
            student = student.eval()
            prev_output_tokens = sample["net_input"]["prev_output_tokens"].data.tolist()
            max_length = max([len(i) for i in prev_output_tokens])  # let's avoid blowing up the GPU RAM, shall we?
            student_generator = SequenceGenerator([student], self.dict, beam_size=1, max_len=max_length)
            asr_generator = SequenceGenerator([self.asr_model], self.dict, beam_size=1, max_len=max_length)
            student_generator.cuda()
            sample["net_input"].pop("src_text", None)
            transcription_hypos = asr_generator._generate(sample)
            transcriptions = []
            lengths = []
            for i, h in enumerate(transcription_hypos):
                transcriptions.append(h[0]["tokens"])
                lengths.append(len(h[0]["tokens"]))
            transcriptions = collate_tokens(
                transcriptions,
                self.dict.pad(),
                self.dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False
            ).cuda()
            sample["net_input"]["src_tokens"] = transcriptions
            sample["net_input"]["src_lengths"] = torch.tensor(lengths, device="cuda")
            student_hypos = student_generator._generate(sample)
            dist = Categorical(torch.tensor([self.beta, 1 - self.beta]))
            samp_mask = [dist.sample((sample["net_input"]["prev_output_tokens"].size(0),)) == 1][0]
            for i, h in enumerate(student_hypos):
                if samp_mask[i]:
                    if h[0]["tokens"][-1] != self.dict.eos():
                        prev_output_tokens[i] = torch.tensor([self.dict.eos()] + h[0]["tokens"].tolist())
                    else:
                        hypo = h[0]["tokens"].tolist()
                        prev_output_tokens[i] = torch.tensor([hypo[-1]] + hypo[1:-1])
                else:
                    prev_output_tokens[i] = torch.tensor(prev_output_tokens[i])
            sample["net_input"]["prev_output_tokens"] = collate_tokens(
                prev_output_tokens,
                self.dict.pad(),
                self.dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False
            ).cuda()
        student.train()
        return sample

    def compute_loss(self, model, net_output, sample, source_text=None, source_lengths=None, reduce=True, valid=False):
        if valid:
            lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
            loss = valid_loss(lprobs, target, self.padding_idx, reduce=reduce)
        else:
            loss = imit_kd_loss(sample, source_text, model, self.expert, source_lengths, self.pad_idx)
        return loss

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
                "translation accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def add_bleu_to_hypotheses(self, sample, hypos):
        """Add BLEU scores to the set of hypotheses.

        This can be called from prepare_sample_and_hypotheses.
        """
        if 'includes_bleu' in sample:
            return hypos
        sample['includes_bleu'] = True

        if self._scorer is None:
            self.create_sequence_scorer()

        target = sample['target'].data.int()
        for i, hypos_i in enumerate(hypos):
            ref = utils.strip_pad(target[i, :], self.pad_idx).cpu()
            r = self.dict.string(ref, bpe_symbol='fastBPE', escape_unk=True)
            r = self.dict.encode_line(r, add_if_not_exist=False)
            for hypo in hypos_i:
                h = self.dict.string(hypo['tokens'].int().cpu(), bpe_symbol='fastBPE')
                h = self.dict.encode_line(h, add_if_not_exist=False)
                # use +1 smoothing for sentence BLEU
                hypo['bleu'] = self._scorer.score(r, h)/100
        return hypos

    def transform_source_tokens_into_expert_voc(self, sample, eos_at_beginning=False):
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


def collate_tokens(values, pad_idx, eos, left_pad, move_eos_to_beginning):
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos
            dst[0] = eos
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        if left_pad:
            copy_tensor(v, res[i][size - len(v):])
        else:
            copy_tensor(v, res[i][:len(v)])
    return res


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
    
