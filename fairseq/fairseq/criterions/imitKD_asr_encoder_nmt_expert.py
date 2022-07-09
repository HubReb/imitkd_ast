# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
import copy
from typing import List, Tuple, Dict, Callable

from torch.distributions import Categorical
import torch
import fastBPE

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.data import Dictionary
from fairseq.sequence_generator import SequenceGenerator
from fairseq.criterions.helper_functions import valid_loss, collate_tokens
import numpy as np


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


def imit_kd_loss(
        generated_dataset,
        transcriptions,
        model,
        expert,
        source_lengths,
        ignore_index,
        booleans,
        model_dict,
        counter,
        source_text,
        source_len
):
    sample_expert = copy.deepcopy(generated_dataset)
    sample_expert["net_input"]["src_tokens"] = transcriptions.cuda()
    sample_expert["net_input"]["src_lengths"] = source_lengths
    with torch.no_grad():
        expert_logits = expert.get_normalized_probs(expert(**sample_expert["net_input"]), log_probs=False)
        """
        preds = expert_logits.argmax(-1)

        for i in preds:
            expert_output = model_dict.string(
                utils.strip_pad(i, ignore_index),
                bpe_symbol='fastBPE',
                escape_unk=True,
                include_eos=True
            )
            print(expert_output)
        sample_expert["net_input"]["src_tokens"] = source_text.cuda()
        sample_expert["net_input"]["src_lengths"] = source_len
        expert_logits_original = expert.get_normalized_probs(expert(**sample_expert["net_input"]), log_probs=False)
        preds = expert_logits.argmax(-1)
        for i in preds:
            expert_output = model_dict.string(
                utils.strip_pad(i, ignore_index),
                bpe_symbol='fastBPE',
                escape_unk=True,
                include_eos=True
            )
            print(expert_output)
        """
        target = sample_expert["net_input"]["prev_output_tokens"]
        if target.dim() == expert_logits.dim() - 1:
            target = target.unsqueeze(-1)
        pad_mask = target.eq(ignore_index)
        expert_logits.masked_fill_(pad_mask, 0.0)

        sample_expert["net_input"]["src_tokens"] = source_text.cuda()
        sample_expert["net_input"]["src_lengths"] = source_len
        expert_logits_original = expert.get_normalized_probs(expert(**sample_expert["net_input"]), log_probs=False)
        expert_logits_original.masked_fill_(pad_mask, 0.0)

    lprobs = model.get_normalized_probs(model(**generated_dataset["net_input"]), log_probs=True)
    wers = 'WER\n'

    t_s = "transcript\toriginal source text\ttarget\thypo\tWER\tmax prob list\tmax prob list original data\tcounter\tin data counter\n"

    # t_s = ""
    numpy_array = []
    numpy_array_orig = []
    in_counter = 0
    for i, boolean in enumerate(booleans):
        if not boolean:
            transcript = model_dict.string(
                utils.strip_pad(transcriptions[i], ignore_index),
                bpe_symbol='fastBPE',
                escape_unk=True,
                include_eos=False
            )
            original_source = model_dict.string(
                utils.strip_pad(source_text[i], ignore_index),
                bpe_symbol='fastBPE',
                escape_unk=True,
                include_eos=False
            )
            #if transcript == original_source:
             #   continue
            wer_rate = wer(transcript, original_source)
            #if wer_rate < 10:
             #   continue
            # print(wer_rate, transcript, original_source)

            target = model_dict.string(
                utils.strip_pad(sample_expert["target"][i], ignore_index),
                bpe_symbol=None,
                escape_unk=True,
                include_eos=True
            )
            hypo = model_dict.string(
                utils.strip_pad(sample_expert["net_input"]["prev_output_tokens"][i], ignore_index),
                bpe_symbol=None,
                escape_unk=True,
                include_eos=True
            )
            while len(target) < expert_logits[i].shape[0]:
                target += " pad"

            wers += f"{wer_rate}\n"

            probs_expert = expert_logits[i]
            probs_expert = probs_expert.cpu().detach().numpy()
            max_prob_list = [str(np.max(l)) for l in probs_expert]
            numpy_array.append(probs_expert)
            probs_expert_orig = expert_logits_original[i]
            probs_expert_orig = probs_expert_orig.cpu().detach().numpy()
            max_prob_list_orig = [str(np.max(l)) for l in probs_expert_orig]
            numpy_array_orig.append(probs_expert_orig)
            t_s += f"{transcript}\t{original_source}\t{target}\t{hypo}\t{wer_rate}\t{', '.join(max_prob_list)}\t{', '.join(max_prob_list_orig)}\t{counter}\t{in_counter}\n"

            in_counter += 1

    if numpy_array_orig:
        numpy_array = np.stack(numpy_array)
        numpy_array_orig = np.stack(numpy_array_orig)
        np.save(f"expert_probs_kd_on_translations_few/{counter}_covost_from_transcripts.npy", numpy_array)
        np.save(f"expert_probs_kd_on_translations_few/{counter}_covost_original.npy", numpy_array_orig)
        with open(f"expert_probs_kd_on_translations_few/transcripts_to_translation.txt", "a") as f:
            f.write(t_s)

    with open(f"covost_wers_small.csv", "a") as f:
        f.write(wers)
    return -torch.sum(expert_logits * lprobs)


@register_criterion(
    "imit_asr_input_nmt", dataclass=ImitKDConfig
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
            beta,
            bpe_codes,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task)
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.counter = 0
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
        self.asr_model, _ = load_model_ensemble(
            [asr_model],
            arg_overrides={"data": path_asr_model, "encoder_freezing_updates": 0}
        )
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
            original_sample = sample
            net_output = model(**sample["net_input"])
            loss = self.compute_loss(model, net_output, sample, [], [], reduce=reduce, valid=valid)
            original_net_output = net_output
        else:
            original_sample = copy.deepcopy(sample)
            original_sample["net_input"].pop("src_text", None)
            original_net_output = model(**original_sample["net_input"])
            source_text, source_lengths = self.transform_source_tokens_into_expert_voc(sample)
            sample, asr_transcriptions, source_lengths, replaced_booleans = self.generate_imit_batch(model, sample)
            sample_s = copy.deepcopy(sample)
            sample_s["net_input"].pop("src_text", None)
            net_output = model(**sample_s["net_input"])
            loss = self.compute_loss(
                model,
                net_output,
                sample,
                asr_transcriptions,
                source_lengths,
                replaced_booleans,
                source_text,
                source_lengths,
                reduce=reduce,
                valid=valid
            )
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
            n_correct, total = self.compute_accuracy(model, original_net_output, original_sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        self.counter += 1
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

    def generate_imit_batch(self, student: Callable, sample: Dict) -> Tuple[Dict, torch.IntTensor, List[int], List[bool]]:
        """
        Use student model to generate hypothesis if probability function beta yields 1.

        :param student: model to train
        :param sample: dataset batch
        :return: dataset batch with prev_output_tokens == student hypothesis if beta_i = 1
        """
        with torch.no_grad():
            student = student.eval()
            prev_output_tokens = sample["net_input"]["prev_output_tokens"].data.tolist()
            sample["net_input"].pop("src_text")
            max_length = max([len(i) for i in prev_output_tokens])  # let's avoid blowing up the GPU RAM, shall we?
            student_generator = SequenceGenerator([student], self.dict, beam_size=1, max_len=max_length)
            asr_generator = SequenceGenerator([self.asr_model], self.dict, beam_size=1, max_len=max_length)
            student_generator.cuda()
            student_hypos = student_generator._generate(sample)
            transcription_hypos = asr_generator._generate(sample)
            transcriptions = []
            lengths = []
            replaced = []
            self.beta = 1
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


            dist = Categorical(torch.tensor([self.beta, 1 - self.beta]))
            samp_mask = [dist.sample((sample["net_input"]["prev_output_tokens"].size(0),)) == 1][0]
            for i, h in enumerate(student_hypos):
                if samp_mask[i]:
                    if h[0]["tokens"][-1] != self.dict.eos():
                        prev_output_tokens[i] = torch.tensor([self.dict.eos()] + h[0]["tokens"].tolist())
                    else:
                        hypo = h[0]["tokens"].tolist()
                        prev_output_tokens[i] = torch.tensor([hypo[-1]] + hypo[0:-1])
                    replaced.append(True)
                else:
                    prev_output_tokens[i] = torch.tensor(prev_output_tokens[i])
                    replaced.append(False)
            sample["net_input"]["prev_output_tokens"] = collate_tokens(
                prev_output_tokens,
                self.dict.pad(),
                self.dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False
            ).cuda()

        student.train()
        return sample, transcriptions, lengths, replaced

    def compute_loss(self, model, net_output, sample, transcriptions=None, source_lengths=None, booleans=None, source_text=None, source_len=None, reduce=True, valid=False):
        if valid:
            lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
            loss = valid_loss(lprobs, target, self.padding_idx, reduce=reduce)
        else:
            loss = imit_kd_loss(sample, transcriptions, model, self.expert, source_lengths, self.pad_idx, booleans, self.dict, self.counter, source_text, source_len)
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


# everything below this point is taken from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
def editDistance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def getStepList(r, h, d):
    """
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    """
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0:
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y - 1] + 1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + 1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]


def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    """
    # build the matrix
    d = editDistance(r, h)

    # find out the manipulation steps
    list = getStepList(r, h, d)

    # print the result in aligned way
    return float(d[len(r)][len(h)]) / len(r) * 100
