#!/usr/bin/env python3

import math
from dataclasses import dataclass, field
from random import choice

from sacremoses import MosesDetokenizer
import sentencepiece as spm
import fastBPE

import torch
from torch.distributions.categorical import Categorical
import numpy as np
from torch.autograd import Variable
from numpy.random import uniform

from fairseq import metrics, utils
from fairseq.scoring import bleu
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.data import Dictionary
from fairseq.sequence_generator import SequenceGenerator


@dataclass
class DifferenceConfig(FairseqDataclass):
    expert: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/wmt19.en-de.joined-dict.ensemble/model1.pt",
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
    path: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/wmt19.en-de.joined-dict.ensemble/",
        metadata={"help": "directory with expert's dictionaries"},
    )
    frozen_student_encoder_path: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/examples/speech_to_text/asr_model/checkpoint_best.pt",
        metadata={"help": "path to encoder"},
    )
    expert_vocab_tgt: str = field(
        default="wmt19.en-de.joined-dict.ensemble/dict.de.txt",
        metadata={"help": "vocab for nmt model output"},
    )
    expert_vocab_src: str = field(
        default="wmt19.en-de.joined-dict.ensemble/dict.en.txt",
        metadata={"help": "vocab for nmt model input"},
    )
    bpe_codes: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/examples/speech_to_text/bpecodes",
        metadata={"help": "expert's bpe codes"},
    )
    frozen_student: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/examples/speech_to_text/checkpoints/checkpoint_last.pt",
        metadata={"help": "last updated student model"},
    )
    frozen_student_path: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/examples/speech_to_text/covost/en",
        metadata={"help": "last updated student model"},
    )



def valid_loss(lprobs, target, ignore_prefix_size, ignore_index=None, reduce=True):
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


def adbleu(sample, lprobs, ignore_index=None):
    partial_hypos = sample["partial_hypos"]
    h_shape = partial_hypos.shape[-1]
    n_shape = lprobs.shape[1]
    if h_shape > n_shape:
        partial_hypos = partial_hypos[:, :n_shape]
    # we sum over the scores in sequence_forward, so we can cut off the 0 probs here as prob would be 0
    elif h_shape < n_shape:
        lprobs = lprobs[:, :h_shape, :]
    if partial_hypos.dim() == lprobs.dim() - 1:
        partial_hypos = partial_hypos.unsqueeze(-1)
    loss = (-lprobs.gather(dim=-1, index=partial_hypos) * sample["reward_difference"])
    if ignore_index is not None:
        pad_mask = partial_hypos.eq(ignore_index)
        loss.masked_fill_(pad_mask, 0.0)
    loss = loss.sum()
    return loss

@register_criterion(
    "reward_difference", dataclass=DifferenceConfig
)
class Difference(FairseqCriterion):
    def __init__(
        self,
        task,
        expert,
        frozen_student,
        expert_vocab_src,
        expert_vocab_tgt,
        path,
        frozen_student_path,
        frozen_student_encoder_path,
        beta,
        bpe_codes,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.expert, _ = load_model_ensemble([expert], arg_overrides={"data": path})
        self.frozen_student_filename = frozen_student
        self.frozen_student = frozen_student
        self.frozen_student_encoder_path = frozen_student_encoder_path
        self.frozen_student_path = frozen_student_path
        self.expert = self.expert[-1]
        self.expert_vocab_src = Dictionary.load(expert_vocab_src)
        self.expert_vocab_tgt = Dictionary.load(expert_vocab_tgt)
        self.expert.requires_grad = False
        self.dict = task.tgt_dict
        self.eos = self.dict.eos()
        self.bpe = fastBPE.fastBPE(bpe_codes, expert_vocab_tgt)
        self.pad_idx = self.padding_idx
        self.sentence_avg = False
        self.beta = beta
        self.frozen_student, _ = load_model_ensemble([self.frozen_student], arg_overrides={"data": self.frozen_student_path, "load_pretrained_encoder_from": self.frozen_student_encoder_path})
        self.frozen_student = self.frozen_student[-1]
        self.frozen_student.requires_grad = False
        




    def forward(self, model, sample, reduce=True, valid=False):
        if valid:
            sample_s= {
                "id": sample["id"],
                "net_input" : { 
                "src_tokens": sample["net_input"]["src_tokens"],
                "src_lengths":sample["net_input"]["src_lengths"],
                "prev_output_tokens": sample["net_input"]["prev_output_tokens"]
                },
                "target": sample["target"],
                "target_lengths": sample["target_lengths"],
                "ntokens": sample["ntokens"],
                "nsentences": sample["nsentences"],
            }
            net_output = model(**sample["net_input"])
            loss = self.compute_loss(model, net_output, sample, reduce=reduce, valid=valid)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "loss" : loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            if self.report_accuracy:
                n_correct, total = self.compute_accuracy(model, net_output, sample)
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)
            return loss, sample_size, logging_output
        source_text = self.transform_source_tokens_into_expert_voc(sample)
        sample, sum_expert_reward, sum_student_reward, number_of_non_zero_rewards, std_expert_reward, std_student_reward = self.generate_dataset(sample, source_text)
        sample_s= {
            "id": sample["id"],
            "net_input" : { 
            "src_tokens": sample["net_input"]["src_tokens"],
            "src_lengths":sample["net_input"]["src_lengths"],
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"]
            },
            "target": sample["target"],
            "target_lengths": sample["target_lengths"],
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            }
        net_output = model(**sample_s["net_input"])
        loss = self.compute_loss(model, net_output, sample, reduce=reduce, valid=valid)
        number_sentences = (sample["target"].size(0))
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss" : loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "sum_expert_reward": sum_expert_reward.copy(),
            "sum_student_reward": sum_student_reward.copy(),
            "std_expert_reward": std_expert_reward.copy(),
            "std_student_reward": std_student_reward.copy(),
            "non_zero_rewards": number_of_non_zero_rewards,
            "number_of_sentences": number_sentences
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
        if valid:
            lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
            loss = valid_loss(lprobs, target, self.ignore_prefix_size, self.padding_idx, reduce=reduce)
            # we need to reload this after every epoch - thankfully checkpoint_last.pt is updated after a epoch and we validate only at the end of the epoch so this hack works
            new_frozen_student, _ = load_model_ensemble([self.frozen_student_filename], arg_overrides={"data": self.frozen_student_path, "load_pretrained_encoder_from": self.frozen_student_encoder_path})
            self.frozen_student = new_frozen_student[-1]
            self.frozen_student.requires_grad = False
            print("Updated frozen student model to new checkpoint!")
        else:
            lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
            loss = adbleu(
                sample,
                lprobs,
                self.padding_idx
            )
        return loss


    def transform_source_tokens_into_expert_voc(self, sample):
        source_text = sample["net_input"]["src_text"]
        source_texts = []
        for line in source_text:
            if type(line) == list:
                for text in line:
                    source_texts.append(self.expert_vocab_src.encode_line(text, add_if_not_exist=False, append_eos=True))
            else:
                source_texts.append(self.expert_vocab_src.encode_line(line, add_if_not_exist=False, append_eos=True))
        source_text = collate_tokens(
            source_texts,
            self.expert_vocab_src.pad(),
            self.expert_vocab_src.eos(),
            left_pad=False,
            move_eos_to_beginning=False
        )
        return source_text


    def generate_dataset(self, sample, source_texts):
        with torch.no_grad():
            student_generator = SequenceGenerator([self.frozen_student], self.dict, beam_size=5)
            expert_generator = SequenceGenerator([self.expert], self.expert_vocab_tgt, beam_size=5)
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
 
            sample_s= {
                "id": sample["id"],
                "net_input" : { 
                "src_tokens": sample["net_input"]["src_tokens"],
                "src_lengths":sample["net_input"]["src_lengths"],
                "prev_output_tokens": sample["net_input"]["prev_output_tokens"]
                },
                "target": sample["target"],
                "target_lengths": sample["target_lengths"],
                "ntokens": sample["ntokens"],
                "nsentences": sample["nsentences"],
            }
            student_generator.cuda()
            hypos = student_generator._generate(sample_s)
            expert_input = []
            partial_hypos = []
            for i in range(len(hypos)):
                t = int(uniform(low=1, high=len(hypos[i][0]["tokens"]), size=None))
                u = uniform(low=0.0, high=1.0, size=None)
                worked, total = 0, 0
                if u > 0.1:
                    sample_frozen = sample.copy()
                    sample_frozen = {
                        "net_input": {
                            "src_tokens": sample["net_input"]["src_tokens"][i].unsqueeze(0).cuda(),
                            "src_lengths": sample["net_input"]["src_lengths"][i].unsqueeze(0),
                            "prev_output_tokens": hypos[i][0]["tokens"][:t].unsqueeze(0),
                        },
                    }
                    try:
                        c = torch.argmax(self.frozen_student.get_normalized_probs(self.frozen_student(**sample_frozen["net_input"]), log_probs=True)[:, t-1, :])
                        worked += 1
                    except RuntimeError:
                        c = hypos[i][0]["tokens"][t]
                    total += 1
                    hypo = torch.cat((hypos[i][0]["tokens"][:t], torch.LongTensor([c]).cuda()))
                else:
                    c = choice(list(self.dict.indices.values()))
                    hypo = torch.cat((hypos[i][0]["tokens"][:t], torch.LongTensor([c]).cuda()))
                partial_hypos.append(hypo)
                expert_input.append(
                    self.expert_vocab_tgt.encode_line(
                        self.bpe.apply([
                            self.dict.string(
                                utils.strip_pad(hypo.clone().detach(), self.pad_idx),
                                bpe_symbol="sentencepiece"
                            )
                        ])[0], add_if_not_exist=False, append_eos=False
                    )
                )
            # if total > 0:
                # print(f"{worked/total} of getting normalized probs worked out")
            partial_hypos = collate_tokens(
                partial_hypos,
                self.dict.pad(),
                self.dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False
            )
            sample_frozen = {
                "net_input": {
                    "src_tokens": sample["net_input"]["src_tokens"],
                    "src_lengths": sample["net_input"]["src_lengths"],
                    "prev_output_tokens": partial_hypos.cuda() 
                },
            }
            hypos = student_generator._generate(sample_frozen)
            expert_input = collate_tokens(
                expert_input,
                self.expert_vocab_tgt.pad(),
                self.expert_vocab_tgt.eos(),
                left_pad=False,
                move_eos_to_beginning=False
            ).to(torch.int64).detach().cuda()
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
            expert_output = expert_generator.generate(
                [self.expert],
                out,
                prefix_tokens=expert_input.cuda(),
            )
            expert_output_samples = []
            for expert_sample in expert_output:
                expert_output_samples.append(expert_sample[0]["tokens"])
            expert_output = collate_tokens(
                expert_output_samples,
                self.expert_vocab_tgt.pad(),
                self.expert_vocab_tgt.eos(),
                left_pad=False,
                move_eos_to_beginning=False
            )
            return self.add_bleu_score_to_sample(sample, hypos, partial_hypos, expert_output)



    def add_bleu_score_to_sample(self, sample, hypos, partial_hypos, expert_hypos):
        """Add BLEU scores to the set of hypotheses.

        """
        target = sample['target'].data.int()
        reward_difference = []
        # student_scorer = BleuScorer(self.dict.pad(), self.dict.eos(), self.dict.unk())
        # expert_scorer = BleuScorer(self.expert_vocab_tgt.pad(), self.expert_vocab_tgt.eos(), self.expert_vocab_tgt.unk())
        reward_expert = []
        reward_student = []
        non_zero_rewards = 0
        for i, hypos_i in enumerate(hypos):     # iterate over dataset
            ref = utils.strip_pad(target[i, :], self.dict.pad()).cpu()
            ref_string = self.dict.string(ref, bpe_symbol='sentencepiece', escape_unk=True)
            # r = self.dict.encode_line(ref_string, add_if_not_exist=False)
            h = self.dict.string(utils.strip_pad(hypos_i[0]['tokens'].int().cpu(), self.dict.pad()), bpe_symbol='sentencepiece')
            jaccard_sim_student = jaccard_sim(ref_string, h)
            h = self.dict.encode_line(h, add_if_not_exist=False)
            # use +1 smoothing for sentence BLEU
            # bleu_student_hypo = student_scorer.score(r, h)
            h_partial = self.dict.string(utils.strip_pad(partial_hypos[i].int().cpu(), self.dict.pad()), bpe_symbol='sentencepiece')
            jaccard_sim_student_partial = jaccard_sim(ref_string, h_partial)
            # h_partial = self.dict.encode_line(h_partial, add_if_not_exist=False)
            # use +1 smoothing for sentence BLEU
            # bleu_student_partial_hypo = student_scorer.score(r, h_partial)
            ref = self.expert_vocab_tgt.encode_line(
                    self.bpe.apply([ref_string])[0], add_if_not_exist=False, append_eos=False
            )
            h = self.expert_vocab_tgt.string(utils.strip_pad(expert_hypos[i].int().cpu(), self.expert_vocab_tgt.pad()), bpe_symbol='fastBPE')
            jaccard_sim_expert = jaccard_sim(ref_string, h)
            # h = self.expert_vocab_tgt.encode_line(h, add_if_not_exist=False)
            # bleu_expert_hypo = expert_scorer.score(ref, h)
            reward_expert.append(jaccard_sim_expert)
            reward_student.append(jaccard_sim_student)
            if jaccard_sim_student > jaccard_sim_expert:
                reward_difference.append(0)
            else:
                reward_difference.append(abs(jaccard_sim_student_partial - jaccard_sim_expert))
                non_zero_rewards += 1
        sample["reward_difference"] = torch.FloatTensor(reward_difference).cuda()
        sample["partial_hypos"] = partial_hypos.clone().detach()
        return sample, np.sum(reward_expert), np.sum(reward_student), non_zero_rewards, reward_expert, reward_student


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
        expert_reward_sum = sum(log.get("sum_expert_reward", 0) for log in logging_outputs)
        student_reward_sum = sum(log.get("sum_student_reward", 0) for log in logging_outputs)
        expert_reward_std = np.std([log.get("std_expert_reward", 0) for log in logging_outputs])
        student_reward_std = np.std([log.get("std_student_reward", 0) for log in logging_outputs])
        number_of_non_zero_rewards = sum(log.get("non_zero_rewards", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        n_sentences =  sum(log.get("number_of_sentences", 0) for log in logging_outputs)

        if number_of_non_zero_rewards > 0:
            expert_mean = expert_reward_sum / number_of_non_zero_rewards
            student_mean = student_reward_sum / number_of_non_zero_rewards
            metrics.log_scalar(
                "reward_expert_over_all_samples", expert_reward_sum / n_sentences, sample_size, round=3
            )
            metrics.log_scalar(
                "reward_student_over_all_samples", student_reward_sum / n_sentences, sample_size, round=3
            )
        else:
            expert_mean, student_mean = 0, 0
        metrics.log_scalar(
            "reward_expert_over_kept_samples", expert_mean, number_of_non_zero_rewards, round=3
        )
        metrics.log_scalar(
            "reward_student_over_kept_samples", student_mean, number_of_non_zero_rewards, round=3
        )
        metrics.log_scalar(
            "reward_expert_over_kept_samples_std", utils.item(expert_reward_std)
        )
        metrics.log_scalar(
            "reward_student_over_kept_samples_std", utils.item(student_reward_std)
        )
 
        metrics.log_scalar(
            "number_of_kept_samples", utils.item(number_of_non_zero_rewards)
        )
        metrics.log_scalar(
            "total_number_of_samples", utils.item(n_sentences)
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
            copy_tensor(v, res[i][size-len(v):])
        else:
            copy_tensor(v, res[i][:len(v)])
    return res


def jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
