# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
import copy
from typing import List, Tuple, Dict

import torch
import fastBPE

import fairseq
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.data import Dictionary
from fairseq.sequence_generator import SequenceGenerator
from fairseq.criterions.helper_functions import valid_loss, collate_tokens

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


def valid_loss(
        lprobs: torch.FloatTensor,
        target: torch.IntTensor,
        ignore_index: int = None,
        reduce: bool = True
) -> torch.FloatTensor:
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


def kl_loss(
        generated_dataset: Dict,
        transcriptions: torch.IntTensor,
        model: fairseq.models.FairseqModel,
        expert: fairseq.models.FairseqModel,
        source_lengths: torch.FloatTensor,
        ignore_index: int,
) -> torch.FloatTensor:
    """
    Calculate Cross-Entropy loss between expert and student model.

    :rtype: torch.FloatTensor
    :param generated_dataset: dictionary containing all information of the batch of the dataset (net input, target values,...)
    :param transcriptions: Transcript of the input audio features to use as source text input for the expert
    :param model: model to be trained
    :param expert: expert for the knowledge distillation
    :param source_lengths: lengths of the transcripts
    :param ignore_index: index of pad values
    :return: Cross-Entropy loss between expert and model
    """
    sample_expert = copy.deepcopy(generated_dataset)
    sample_expert["net_input"]["src_tokens"] = transcriptions.cuda()
    sample_expert["net_input"]["src_lengths"] = source_lengths
    with torch.no_grad():
        expert_logits = expert.get_normalized_probs(expert(**sample_expert["net_input"]), log_probs=False)
        target = sample_expert["target"]
        target = target.unsqueeze(-1)
        pad_mask = target.eq(ignore_index)
        expert_logits.masked_fill_(pad_mask, 0.0)
    lprobs = model.get_normalized_probs(model(**generated_dataset["net_input"]), log_probs=True)
    return -torch.sum(expert_logits * lprobs)


@register_criterion(
    "kd_asr_input_nmt", dataclass=ImitKDConfig
)
class ImitKD(FairseqCriterion):
    def __init__(
            self,
            task,
            expert,
            expert_vocab_src,
            expert_vocab_tgt,
            path,
            asr_model,
            path_asr_model,
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
            net_output = model(**sample["net_input"])
            sample["net_input"].pop("src_text", None)
            loss = self.compute_loss(model, net_output, sample, reduce=reduce, valid=valid)
        else:
            sample, asr_transcriptions, source_lengths = self.generate_audio_transcripts(sample)
            sample_s = copy.deepcopy(sample)
            sample_s["net_input"].pop("src_text", None)
            net_output = model(**sample_s["net_input"])
            loss = self.compute_loss(model, net_output, sample, asr_transcriptions, source_lengths, reduce=reduce, valid=valid)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample,)
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

    def generate_audio_transcripts(self,
                                   sample: Dict
                                   ) -> Tuple[Dict, torch.IntTensor, List[int]]:
        """
        Use ASR model (self.asr_model) to transcribe the audio data in sample.
        :param sample: dataset batch to train on
        :return: dataset batch and corresponding asr model transcriptions of dataset batch's audio features
        """
        with torch.no_grad():
            prev_output_tokens = sample["net_input"]["prev_output_tokens"].data.tolist()
            sample["net_input"].pop("src_text")
            max_length = max([len(i) for i in prev_output_tokens])  # let's avoid blowing up the GPU RAM, shall we?
            asr_generator = SequenceGenerator([self.asr_model], self.dict, beam_size=1, max_len=max_length).cuda()
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
            sample["net_input"].pop("src_text", None)
        return sample, transcriptions, lengths

    def compute_loss(self, model, net_output, sample, transcriptions=None, source_lengths=None, reduce=True, valid=False):
        if valid:
            lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
            loss = valid_loss(lprobs, target, self.padding_idx, reduce=reduce)
        else:
            loss = kl_loss(sample, transcriptions, model, self.expert, source_lengths, self.pad_idx)
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



