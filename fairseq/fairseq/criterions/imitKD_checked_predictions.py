# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
import copy
from typing import List, Tuple, Dict, Callable

import sentencepiece as spm
import fastBPE

import torch
from torch.nn.functional import kl_div
from torch.distributions import Categorical

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.data import Dictionary
from fairseq.sequence_generator import SequenceGenerator
from nltk.translate.bleu_score import sentence_bleu
from fairseq.criterions.helper_functions import valid_loss, collate_tokens


@dataclass
class ImitKDConfigCheckedPredictionsWithGoldReferences(FairseqDataclass):
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


def imit_kd_loss(
        generated_dataset,
        model,
        expert,
        source_text,
        model_dict,
        source_lengths
):
    sample_expert = {
        "id": generated_dataset["id"],
        "net_input": {
            "src_tokens": source_text.cuda(),
            "src_lengths": source_lengths,
            "prev_output_tokens": generated_dataset["net_input"]["prev_output_tokens"].cuda()
        },
        "target": generated_dataset["target"],
        "target_lengths": generated_dataset["target_lengths"],
        "ntokens": generated_dataset["ntokens"],
        "nsentences": generated_dataset["nsentences"],
    }
    with torch.no_grad():
        expert_out = expert.get_normalized_probs(expert(**sample_expert["net_input"]), log_probs=False)
        # expert_preds = expert_out.argmax(-1)
        pad_mask = expert_out.eq(model_dict.pad())
        expert_out.masked_fill_(pad_mask, 0.0)
    lprobs = model.get_normalized_probs(model(**generated_dataset["net_input"]), log_probs=True)
    # pad_mask = lprobs.eq(model_dict.pad())
    # lprobs.masked_fill_(pad_mask, 0.0)
    # kl_loss = kl_div(lprobs, expert_out, reduction="batchmean", log_target=True)
    # good old CE
    return -torch.sum(expert_out * lprobs)


@register_criterion(
    "imit_kd_checked_predictions", dataclass=ImitKDConfigCheckedPredictionsWithGoldReferences
)
class ImitKDCheckedPredictionsWithGoldReferences(FairseqCriterion):
    def __init__(
            self,
            task,
            expert,
            expert_vocab_src,
            expert_vocab_tgt,
            path,
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
        loss = self.compute_loss(model, net_output, sample, reduce=reduce, valid=valid)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy or valid:
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
            lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
            loss = valid_loss(lprobs, target, self.padding_idx, reduce=reduce)
        else:
            source_text, source_lengths = self.transform_source_tokens_into_expert_voc(sample)
            sample_s = copy.deepcopy(sample)
            sample_s["net_input"].pop("src_text", None)
            generated_dataset = self.generate_imit_batch(model, sample_s, source_text, source_lengths)
            loss = imit_kd_loss(
                generated_dataset,
                model,
                self.expert,
                source_text,
                self.dict,
                source_lengths
            )
        return loss

    def generate_imit_batch(
            self,
            student: Callable,
            sample: Dict,
            source_text: torch.IntTensor,
            source_lengths: List[int]
    ) -> Dict:
        """
        Use student model to generate hypothesis if the bleu score of the expert translation of the source text and the
        student hypothesis is above a chosen threshold (quality control).

        :param student: model to train
        :param sample: dataset batch
        :param source_text: bpe encoded source text in model vocabulary
        :param source_lengths: list of the lengths of the bpe encoded source texts
        :return: dataset batch with prev_output_tokens == student hypothesis if beta_i = 1
        """
        with torch.no_grad():
            student = student.eval()
            sample_expert = {
                "id": sample["id"],
                "net_input": {
                    "src_tokens": source_text.cuda(),
                    "src_lengths": source_lengths,
                    "prev_output_tokens": sample["net_input"]["prev_output_tokens"].cuda()
                },
                "target": sample["target"],
                "target_lengths": sample["target_lengths"],
                "ntokens": sample["ntokens"],
                "nsentences": sample["nsentences"],
            }
            targets = sample["net_input"]["prev_output_tokens"].data.tolist()
            max_length = max([len(i) for i in targets])  # let's avoid blowing up the GPU RAM, shall we?
            student_generator = SequenceGenerator([student], self.dict, beam_size=1, max_len=max_length)
            expert_generator = SequenceGenerator([self.expert], self.expert_vocab_tgt, beam_size=1, max_len=max_length)
            student_generator.cuda()
            hypos = student_generator._generate(sample)
            # using parameter , generated_length=max_length would achieve the same  as cutting of hypothesis at
            # [:max_length] after generation, but requires dealing with token matrix instead of the dict
            hypos_expert = expert_generator._generate(sample_expert)
            for i, hypo in enumerate(hypos):
                # prediction_list = hypo[0]["tokens"].tolist()
                prediction_list = self.dict.string(
                    utils.strip_pad(hypo[0]["tokens"], self.pad_idx), bpe_symbol='fastBPE', escape_unk=True, include_eos=False
                ).split()
                score = sentence_bleu(
                    [
                        self.dict.string(
                            utils.strip_pad(hypos_expert[i][0]["tokens"], self.pad_idx),
                            bpe_symbol='fastBPE',
                            escape_unk=True,
                            include_eos=False
                        ).split()],
                    prediction_list,
                    weights=(0.1, 0.4, 0.5)
                )
                if score >= 0.3:
                    # print(prediction_list, targets[i])
                    if hypo[0]["tokens"][-1] != self.dict.eos():
                        targets[i] = torch.tensor([self.dict.eos()] + hypo[0]["tokens"].tolist())
                    else:
                        hypo = hypo[0]["tokens"].tolist()
                        targets[i] = torch.tensor([hypo[-1]] + hypo[1:-1])
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
        Turn the tokenized source text into bpe encodings.
        :param sample: dataset batch
        :param eos_at_beginning: whether to put EOS token at the beginning of each sample (required for previous output tokens)
        :return: Tuple of bpe encoded source text and list of integers determing the number of bp for each sample
        """
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
