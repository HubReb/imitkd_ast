# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from random import randint

from sacremoses import MosesDetokenizer

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.data import Dictionary
from fairseq.sequence_generator import SequenceGenerator


@dataclass
class OracleForcedDecodingConfig(FairseqDataclass):
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
    entire_sentence: bool = field(
        default=False,
        metadata={"help": "whether to complete entire sentence instead of sampling one word"},
    )



def knn_forced_loss(
        expert,
        student,
        lprobs,
        target,
        sample,
        model_vocab,
        expert_vocab_src,
        expert_vocab_tgt,
        entire_sentence,
        ignore_prefix_size,
        ignore_index=None,
        reduce=True,
        valid=False
        ):
    if valid:
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
        return nll_loss, nll_loss
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    batch_size = target.shape[0]
    student_output = lprobs.argmax(dim=-1)
    student_predictions = []
    text_predictions = []
    texts = []
    indices = []
    for prediction in student_output:
        prediction_string = model_vocab.string(
            utils.strip_pad(prediction, ignore_index),
            bpe_symbol="sentencepiece",
        )
        max_range = len(prediction_string.split())-1
        if max_range <= 1:
            index = 1
        else:
            index = randint(1, len(prediction_string.split())-1)
        indices.append(index)
        texts.append(prediction_string)
        prediction_string = " ".join(prediction_string.split()[:index])
        prediction_string_in_expert_vocab = expert_vocab_tgt.encode_line(
                prediction_string, add_if_not_exist=False, append_eos=False
                )
        if expert_vocab_tgt.unk() in prediction_string_in_expert_vocab:
            indices.pop()
            new_index = prediction_string_in_expert_vocab.tolist().index(expert_vocab_tgt.unk())
            prediction_string = expert_vocab_tgt.string(
                    utils.strip_pad(prediction_string_in_expert_vocab[:new_index], ignore_index),
                    bpe_symbol="fastBPE"
            )
            indices.append(len(prediction_string.split()))
            prediction_string_in_expert_vocab = prediction_string_in_expert_vocab[:new_index]
        # print("student: ", prediction_string)
        # print(prediction_string_in_expert_vocab)
        text_predictions.append(prediction_string)
        student_predictions.append(prediction_string_in_expert_vocab.to(torch.int64))
    expert_input = torch.nn.utils.rnn.pad_sequence(student_predictions, batch_first=True, padding_value=ignore_index)
    source_text = sample["net_input"]["src_text"]
    source_texts = []
    for line in source_text:
        if type(line) == list:
            for text in line:
                source_texts.append(expert_vocab_src.encode_line(text, add_if_not_exist=False, append_eos=True))
        else:
            source_texts.append(expert_vocab_src.encode_line(line, add_if_not_exist=False, append_eos=True))
    source_texts = torch.nn.utils.rnn.pad_sequence(source_texts, padding_value=ignore_index, batch_first=True)
    out = {
        "id": sample["id"],
        "net_input": {
            "src_tokens": source_texts.cuda(),
            "src_lengths": [len(text) for text in source_text],
            "prev_output_tokens": [],
        },
        "target": sample["target"],
        "target_lengths": sample["target_lengths"],
        "ntokens": sample["ntokens"],
        "nsentences": sample["nsentences"],
        }
 
    expert_output = expert.generate(
        [expert],
        out,
        prefix_tokens=expert_input.cuda()
    )
    # get best
    if type(expert_output[0]) != dict:
        expert_output_samples = []
        for expert_sample in expert_output:
            expert_output_samples.append(expert_sample[0]["tokens"])
    else:
        expert_output_samples = []
        for expert_sample in expert_output:
            expert_output_samples.append(expert_sample["tokens"])
    # print(expert_output_samples, batch_size)
    expert_output = torch.nn.utils.rnn.pad_sequence(expert_output_samples, padding_value=ignore_index, batch_first=True)
    # print(sample["net_input"]["prev_output_tokens"])
    # print(expert_output.shape)
    for index, output in enumerate(expert_output):
        # print(index)
        expert_output_string = expert_vocab_tgt.string(
            utils.strip_pad(output, ignore_index),
            bpe_symbol="fastBPE",
        )
        print(texts[index])
        print(expert_output_string)
        if entire_sentence:
            line = expert_output_string
        else:
            if not len(text_predictions[index].split()) == len(expert_output_string.split()):
                line = text_predictions[index] + " " + expert_output_string.split()[indices[index]]
            else:
                # expert failed to complete student's input - nothing to take from this
                continue
        # print(line)
        new_student_input = model_vocab.encode_line(
            line,
            add_if_not_exist=False,
            append_eos=False
        )
        if entire_sentence:
            new_student_input = new_student_input.tolist()
            while len(new_student_input) < len(sample["net_input"]["prev_output_tokens"][index]):
                new_student_input.append(ignore_index)
            new_student_input = torch.LongTensor(new_student_input)
            sample["net_input"]["prev_output_tokens"][index, 1:] = new_student_input[:len(sample["net_input"]["prev_output_tokens"][index])-1].cuda()
        else:
            sample["net_input"]["prev_output_tokens"][index, 1:indices[index]+2] = new_student_input.cuda()
    net_output = student(
        **sample["net_input"]
    )
    lprobs_new = student.get_normalized_probs(net_output, log_probs=True)
    if ignore_prefix_size > 0:
        if getattr(lprobs, "batch_first", False):
            lprobs_new = lprobs_new[:, ignore_prefix_size:, :].contiguous()
        else:
            lprobs_new = lprobs_new[ignore_prefix_size:, :, :].contiguous()
    lprobs_new = lprobs_new.view(-1, lprobs.size(-1))
    target = target.view(-1)
    if target.dim() == lprobs_new.dim() - 1:
        target = target.unsqueeze(-1)
    loss = -lprobs_new.gather(dim=-1, index=target)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    nll_loss = -lprobs.gather(dim=-1, index=target)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        loss.masked_fill_(pad_mask, 0.0)
        nll_loss.masked_fill_(pad_mask, 0.0)
    else:
        loss = loss.squeeze(-1)
        nll_loss = nll_loss.squeeze(-1)
    if reduce:
        loss = loss.sum()
        nll_loss = nll_loss.sum()

    return loss, nll_loss


@register_criterion(
    "oracle_nmt", dataclass=OracleForcedDecodingConfig
)
class OracleForcedDecoding(FairseqCriterion):
    def __init__(
        self,
        task,
        expert,
        expert_vocab_src,
        expert_vocab_tgt,
        path,
        entire_sentence,
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
        self.expert = SequenceGenerator(
                [self.expert],
                self.expert_vocab_tgt
        )
        self.expert.requires_grad = False
        self.dict = task.tgt_dict
        self.sentence_avg = False
        self.entire_sentence = entire_sentence


    def forward(self, model, sample, reduce=True, valid=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce, valid=valid)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss" : loss.data,
            "nll_loss": nll_loss.data,
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
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs, target

    def compute_loss(self, model, net_output, sample, reduce=True, valid=False):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss = knn_forced_loss(
            self.expert,
            model,
            lprobs,
            target,
            sample,
            self.dict,
            self.expert_vocab_src,
            self.expert_vocab_tgt,
            self.entire_sentence,
            self.ignore_prefix_size,
            ignore_index=self.padding_idx,
            reduce=reduce,
            valid=valid
        )
        return loss

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
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
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


