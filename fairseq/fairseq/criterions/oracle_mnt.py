# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from random import randint

from sacremoses import MosesDetokenizer

import torch
from torch.autograd import Variable

from fairseq import metrics, utils
from fairseq.scoring import bleu
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.data import Dictionary
from fairseq.sequence_generator import SequenceGenerator


@dataclass
class OracleForcedDecodingMNTConfig(FairseqDataclass):
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
    guided: int = field(
        default=1,
        metadata={"help": "How many bpe to take from expert beam"},
    )
    avoid_unk: bool = field(
        default=False,
        metadata={"help": "whether to split the model prediction before the first unknown"},
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


def knn_forced_loss(
        scores,
        sample,
        lengths
        ):
    avg_scores = scores.sum(2)/lengths
    probs = torch.nn.functional.log_softmax(avg_scores.exp_())
    loss = -(probs * sample['reward'].type_as(probs)).sum()
    return loss


@register_criterion(
    "oracle_nmt_mnt", dataclass=OracleForcedDecodingMNTConfig
)
class OracleForcedDecodingMNT(FairseqCriterion):
    def __init__(
        self,
        task,
        expert,
        expert_vocab_src,
        expert_vocab_tgt,
        path,
        guided,
        avoid_unk,
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
        self.pad_idx = self.padding_idx
        self.sentence_avg = False
        self.guidance = guided
        self._scorer = BleuScorer(self.pad_idx, self.eos, self.dict.unk())
        self.avoid_unk = avoid_unk


    def forward(self, model, sample, reduce=True, valid=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
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
        if valid:
            loss = valid_loss(lprobs, target, self.ignore_prefix_size, self.ignore_prefix_size, reduce=reduce)
        else:
            expert_input, texts, cut_texts, indices = self.get_student_predictions_and_pass_to_expert(model, target, sample)
            source_text = self.transform_source_tokens_into_expert_voc(sample)
            expert_output_samples = self.get_expert_output(sample, source_text, expert_input)
            prefix_tokens = self.transform_expert_output_into_student_voc(
                sample,
                expert_output_samples,
                texts,
                cut_texts,
                indices
            )
            scores, sample, lengths = self.get_hypos_and_scores(sample, model, prefix_tokens, lprobs)
            loss = knn_forced_loss(
                scores,
                sample,
                lengths
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
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
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


    def collate_tokens(self, values, pad_idx, eos, left_pad, move_eos_to_beginning):
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

    def update_sample_with_hypos(self, sample, hypos):
        num_hypos_per_batch = len(hypos[0])
        assert all(len(h) == num_hypos_per_batch for h in hypos)

        def repeat_num_hypos_times(t, dim2=False):
            ta = t.repeat(1, num_hypos_per_batch, 1, 1)
            return ta.view(num_hypos_per_batch*t.size(0), t.size(1), t.size(2))

        input = sample['net_input']
        bsz = input['src_tokens'].size(0)
        #input['src_tokens'].data = repeat_num_hypos_times(input['src_tokens'].data)
        #input['prev_output_tokens'].data = repeat_num_hypos_times(input['prev_output_tokens'].data, dim2=True)

        input_hypos = [h['tokens'] for hypos_i in hypos for h in hypos_i]
        sample['hypotheses'] = self.collate_tokens(
            input_hypos, self.pad_idx, self.eos, left_pad=False, move_eos_to_beginning=False)
        # only needed for authors' model
        # input['input_tokens'] = self.collate_tokens(
            # input_hypos, self.pad_idx, self.eos, left_pad=True, move_eos_to_beginning=True)
        # input['input_positions'] = self.collate_positions(
            # input_hypos, self.pad_idx, left_pad=True)

        # sample['target'].data = repeat_num_hypos_times(sample['target'].data, dim2=True)
        sample['ntokens'] = sample['target'].data.ne(self.pad_idx).sum()
        sample['bsz'] = bsz
        sample['num_hypos_per_batch'] = num_hypos_per_batch
        return sample

    def add_reference_to_hypotheses(self, sample, hypos):
        """Add the reference translation to the set of hypotheses.

        This can be called from prepare_sample_and_hypotheses.
        """
        if 'includes_reference' in sample:
            return hypos
        sample['includes_reference'] = True

        target = sample['target'].data
        for i, hypos_i in enumerate(hypos):
            # insert reference as first hypothesis
            ref = utils.strip_pad(target[i, :], self.pad_idx)
            hypos_i.insert(0, {
                'tokens': ref,
                'score': None,
            })
        return hypos

    def add_bleu_to_hypotheses(self, sample, hypos):
        """Add BLEU scores to the set of hypotheses.

        This can be called from prepare_sample_and_hypotheses.
        """
        if 'includes_bleu' in sample:
            return hypos
        sample['includes_bleu'] = True

        target = sample['target'].data.int()
        for i, hypos_i in enumerate(hypos):
            ref = utils.strip_pad(target[i, :], self.pad_idx).cpu()
            r = self.dict.string(ref, bpe_symbol='sentencepiece', escape_unk=True)
            r = self.dict.encode_line(r, add_if_not_exist=False)
            for hypo in hypos_i:
                h = self.dict.string(hypo['tokens'].int().cpu(), bpe_symbol='sentencepiece')
                h = self.dict.encode_line(h, add_if_not_exist=False)
                # use +1 smoothing for sentence BLEU
                hypo['bleu'] = self._scorer.score(r, h)
        return hypos


    def prepare_sample_and_hypotheses(self, model, sample, hypos):
        """Apply criterion-specific modifications to the given sample/hypotheses."""
        scale_scores = lambda x : x

        # compute BLEU reward for each hypothesis
        hypos = self.add_bleu_to_hypotheses(sample, hypos)
        reward = torch.FloatTensor([
            scale_scores([h['bleu'] for h in hypos_i])
            for hypos_i in hypos
        ])
        sample['reward'] = Variable(reward, requires_grad=False)
        return sample, hypos

    def get_student_predictions_and_pass_to_expert(self, model, target, sample):
        student_predictions = []
        student_generator = SequenceGenerator([model], self.dict, beam_size=5)
        student_generator.cuda()
        hypos = student_generator._generate(sample)
        texts = []
        indices = []
        cut_texts = []
        for prediction in hypos:
            prediction_string = self.dict.string(
                utils.strip_pad(prediction[0]["tokens"], self.padding_idx),
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
            cut_texts.append(prediction_string)
            prediction_string_in_expert_vocab = self.expert_vocab_tgt.encode_line(
                    prediction_string, add_if_not_exist=False, append_eos=False
                    )
            if self.expert_vocab_tgt.unk() in prediction_string_in_expert_vocab and self.avoid_unk:
                indices.pop()
                new_index = prediction_string_in_expert_vocab.tolist().index(self.expert_vocab_tgt.unk())
                prediction_string = self.expert_vocab_tgt.string(
                        utils.strip_pad(prediction_string_in_expert_vocab[:new_index], self.expert_vocab_tgt.pad()),
                        bpe_symbol="fastBPE"
                )
                indices.append(len(prediction_string.split()))
                prediction_string_in_expert_vocab = prediction_string_in_expert_vocab[:new_index]
            student_predictions.append(prediction_string_in_expert_vocab.to(torch.int64))
        expert_input = self.collate_tokens(
                student_predictions,
                self.expert_vocab_src.pad(),
                self.expert_vocab_src.eos(),
                left_pad=False,
                move_eos_to_beginning=False
        )
        return expert_input, texts, cut_texts, indices

    def transform_source_tokens_into_expert_voc(self, sample):
        source_text = sample["net_input"]["src_text"]
        source_texts = []
        for line in source_text:
            if type(line) == list:
                for text in line:
                    source_texts.append(self.expert_vocab_src.encode_line(text, add_if_not_exist=False, append_eos=True))
            else:
                source_texts.append(self.expert_vocab_src.encode_line(line, add_if_not_exist=False, append_eos=True))
        source_text = self.collate_tokens(
            source_texts,
            self.expert_vocab_src.pad(),
            self.expert_vocab_src.eos(),
            left_pad=False,
            move_eos_to_beginning=False
        )
        return source_text

    def get_expert_output(self, sample, source_texts, expert_input):
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
        expert_generator = SequenceGenerator(
                [self.expert],
                self.expert_vocab_tgt
        )
        expert_output = expert_generator.generate(
            [self.expert],
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
        return expert_output_samples

    def transform_expert_output_into_student_voc(self, sample, expert_output_samples, text_predictions, cut_predictions, indices):
        prefix_tokens = []
        for index, output in enumerate(expert_output_samples):
            # print(index)
            expert_output_string = self.expert_vocab_tgt.string(
                utils.strip_pad(output, self.expert_vocab_tgt.pad()),
                bpe_symbol="fastBPE",
            )
            # print("student prediction: ", text_predictions[index])
            # print("expert prediction: ", expert_output_string)
            # print("index: ", indices[index])
            if not len(cut_predictions[index].split()) == len(expert_output_string.split()):
                line = cut_predictions[index] + " " + " ".join(expert_output_string.split()[indices[index]:indices[index]+self.guidance+1])
            else:
                # expert failed to complete student's input - nothing to take from this
                line = cut_predictions[index]
            new_student_input = self.dict.encode_line(
                line,
                add_if_not_exist=False,
                append_eos=False
            )
            prefix_tokens.append(new_student_input.to(torch.int64))
        # initialize student generator
        prefix_tokens = self.collate_tokens(prefix_tokens, self.pad_idx, self.eos, left_pad=False, move_eos_to_beginning=False)
        return prefix_tokens
    

    def get_hypos_and_scores(self, sample, model, prefix_tokens, lprobs):
        student_generator = SequenceGenerator([model], self.dict, beam_size=5)
        student_generator.cuda()
        hypos = student_generator._generate(sample, prefix_tokens=prefix_tokens.cuda())
        sample, hypos = self.prepare_sample_and_hypotheses(model, sample, hypos)
        sample = self.update_sample_with_hypos(sample, hypos)
        bzw, target_len, vocab_len = lprobs.size()
        lengths = Variable(sample['hypotheses'].view(bzw, 5, -1).ne(self.dict.pad()).sum(2).float(), requires_grad=False)
        scores = self.get_hypothesis_scores(lprobs, sample)
        return scores, sample, lengths
 

    def get_hypothesis_scores(self, lprobs, sample):
        hypotheses = Variable(sample['hypotheses'], requires_grad=False)
        bzw, target_len, vocab_len = lprobs.size()
        hypotheses = hypotheses.view(bzw, 5, -1, 1)
        net_output = lprobs.repeat(1, 5, 1, 1).view(bzw, 5, -1, vocab_len)
        h_shape = hypotheses.shape[2]
        n_shape = net_output.shape[2]
        if h_shape > n_shape:
            hypotheses = hypotheses[:, :, :n_shape, :]
        # we sum over the scores in sequence_forward, so we can cut off the 0 probs here as prob would be 0
        elif h_shape < n_shape:
            net_output = net_output[:, :, :h_shape, :]
        scores = net_output.gather(3, hypotheses)
        scores = scores * hypotheses.ne(self.dict.pad()).float()
        scores = scores.squeeze(3)
        return scores


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

