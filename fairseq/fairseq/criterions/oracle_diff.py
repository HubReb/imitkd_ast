# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import math
from dataclasses import dataclass, field
from random import choice, randint

import torch
from torch.autograd import Variable
import fastBPE
from torch.distributions import Categorical

from fairseq import metrics, utils
from fairseq.checkpoint_utils import load_model_ensemble
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import Dictionary
from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import bleu
from fairseq.sequence_generator import SequenceGenerator
from nltk.translate.bleu_score import sentence_bleu


@dataclass
class OracleDiffConfig(FairseqDataclass):
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

    bpe_codes: str = field(
        default="/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/examples/speech_to_text/bpecodes",
        metadata={"help": "expert's bpe codes"},
    )
    adbleu_loss: bool = field(
        default=False,
        metadata={"help":"whether to use Luca Hormann's ADBLEU loss instead of AGGREVATEs BSE"}
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


def knn_forced_loss(
        scores,
        sample_student,
        sample_expert,
        indices,
        adbleu_loss,
        complete_student_hypos
):
    # avg_scores = scores.sum(2)/lengths
    # probs = torch.nn.functional.softmax(avg_scores.exp_())
    probs = scores
    indices = torch.tensor(indices, device="cuda").view(-1, 1).unsqueeze(-1)
    # print(indices.shape, probs.shape)
    probs = probs.gather(dim=-1, index=indices)
    reward_student = sample_student["reward"]
    reward_expert = sample_expert["reward"][:, 0].view(-1, 1).cuda()  # we only care about best hypo's reward
    if adbleu_loss:
        complete_student_reward = torch.FloatTensor([
            [h['bleu'] for h in hypos_i]
            for hypos_i in complete_student_hypos
        ]).cuda()
        indicator = []
        for i, reward_row in enumerate(reward_student):
            indicator_row = []
            for j, r in enumerate(reward_row):
                if r > sample_expert["reward"][i][j]:
                    indicator_row.append(0)
                else:
                    indicator_row.append(1)
            indicator.append(indicator_row)
        indicator = torch.LongTensor(indicator).cuda()
    if adbleu_loss:
        loss = -(probs * indicator * (torch.tanh(reward_student).cuda() - (reward_expert - complete_student_reward))**2).sum()
    else:
        loss = -(probs * ((reward_student.cuda() - reward_expert) ** 2).type_as(probs)).sum()
    return loss, reward_expert.sum(), reward_student.sum()


@register_criterion(
    "oracle_diff", dataclass=OracleDiffConfig
)
class OracleDiff(FairseqCriterion):
    def __init__(
            self,
            task,
            expert,
            expert_vocab_src,
            expert_vocab_tgt,
            path,
            bpe_codes,
            adbleu_loss,
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
        self.expert = self.expert.eval()
        self.expert_vocab_src = Dictionary.load(expert_vocab_src)
        self.expert_vocab_tgt = Dictionary.load(expert_vocab_tgt)
        self.expert.requires_grad = False
        self.dict = task.tgt_dict
        self.eos = self.dict.eos()
        self.bpe = fastBPE.fastBPE(bpe_codes, expert_vocab_tgt)
        self.pad_idx = self.padding_idx
        self.sentence_avg = False
        self.adbleu_loss = adbleu_loss
        self._scorer = BleuScorer(self.pad_idx, self.eos, self.dict.unk())

    def forward(self, model, sample, reduce=True, valid=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_f = copy.deepcopy(sample)
        sample_f["net_input"].pop("src_text", None)
        net_output = model(**sample_f["net_input"])
        loss, r_expert, r_student = self.compute_loss(model, net_output, sample, reduce=reduce, valid=valid)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "reward_expert": r_expert,
            "reward_student": r_student,
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
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        if valid:
            loss = valid_loss(lprobs, target, self.ignore_prefix_size, reduce=reduce)
            r_expert, r_student = 0, 0
        else:
            source_text, source_lens = self.transform_source_tokens_into_expert_voc(sample)
            expert_input, texts, cut_texts, indices, student_hypos, original_student_hypos = self.get_student_predictions_and_pass_to_expert(
                model, sample)
            expert_output_samples = self.get_expert_output(sample, source_text, source_lens, expert_input)
            scores, sample_student, sample_expert, lengths, original_student_hypos = self.get_hypos_and_scores(sample, lprobs,
                                                                                       expert_output_samples,
                                                                                       student_hypos,
                                                                                       original_student_hypos
                                                                                       )
            loss, r_expert, r_student = knn_forced_loss(
                scores,
                sample_student,
                sample_expert,
                indices,
                self.adbleu_loss,
                original_student_hypos
            )
        return loss, r_expert, r_student

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
        sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        reward_expert_sum = sum(log.get("reward_expert", 0) for log in logging_outputs)
        reward_student_sum = sum(log.get("reward_student", 0) for log in logging_outputs)
        number_of_sentences = sum(log.get("nsentences", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "mean_reward_expert", reward_expert_sum / number_of_sentences, sample_size, round=3
        )
        metrics.log_scalar(
            "mean_reward_student", reward_student_sum / number_of_sentences, sample_size, round=3
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

    @staticmethod
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

    def update_sample_with_hypos(self, sample, hypos):
        num_hypos_per_batch = len(hypos[0])
        assert all(len(h) == num_hypos_per_batch for h in hypos)

        input_for_sample = sample['net_input']
        bsz = input_for_sample['src_tokens'].size(0)
        # input['src_tokens'].data = repeat_num_hypos_times(input['src_tokens'].data)
        # input['prev_output_tokens'].data = repeat_num_hypos_times(input['prev_output_tokens'].data, dim2=True)

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

    def add_bleu_to_hypotheses(self, sample, hypos, student=False):
        """Add BLEU scores to the set of hypotheses.

        This can be called from prepare_sample_and_hypotheses.
        """
        sample['includes_bleu'] = True

        target = sample['target'].data.int()
        for i, hypos_i in enumerate(hypos):
            ref = utils.strip_pad(target[i, :], self.pad_idx).cpu()
            r = self.dict.string(ref, bpe_symbol='fastBPE', escape_unk=True)#.split()
            r = self.expert_vocab_tgt.encode_line(r, add_if_not_exist=False)
            for hypo in hypos_i:
                if student:
                    h = self.expert_vocab_tgt.string(utils.strip_pad(hypo['tokens'][:-1], self.pad_idx).int().cpu(),
                                                     bpe_symbol='fastBPE')#.split()
                else:
                    h = self.expert_vocab_tgt.string(
                        utils.strip_pad(hypo['tokens'], self.pad_idx).int().cpu(),
                        bpe_symbol='fastBPE'
                    )#.split()
                h = self.expert_vocab_tgt.encode_line(h, add_if_not_exist=False)

                """
                score = sentence_bleu(
                    [r],
                    h
                )
                hypo['bleu'] = score
                """
                # use +1 smoothing for sentence BLEU
                hypo['bleu'] = self._scorer.score(r, h)
        return hypos

    def prepare_sample_and_hypotheses(self, sample, hypos, student=False):
        """Apply criterion-specific modifications to the given sample/hypotheses."""
        scale_scores = lambda x: x

        # compute BLEU reward for each hypothesis
        if student:
            hypos = self.add_bleu_to_hypotheses(sample, hypos, student)
        else:
            hypos = self.add_bleu_to_hypotheses(sample, hypos)
        reward = torch.FloatTensor([
            scale_scores([h['bleu'] for h in hypos_i])
            for hypos_i in hypos
        ])
        sample['reward'] = Variable(reward, requires_grad=False)
        return sample, hypos

    def get_student_predictions_and_pass_to_expert(self, model, sample):
        with torch.no_grad():
            student_predictions = []
            model = model.eval()
            targets = sample["net_input"]["prev_output_tokens"].data.tolist()
            sample["net_input"].pop("src_text", None)
            net_output = model.get_normalized_probs(model(**sample["net_input"]), log_probs=True)
            top_k_predictions = torch.topk(net_output, k=5, dim=-1).indices
            max_length = max([len(i) for i in targets])  # let's avoid blowing up the GPU RAM, shall we?
            student_generator = SequenceGenerator([model], self.dict, beam_size=1, max_len=max_length)
            student_generator.cuda()
            self.beta = 0.1
            dist = Categorical(torch.tensor([self.beta, 1 - self.beta]))
            samp_mask = [dist.sample((sample["net_input"]["prev_output_tokens"].size(0),)) == 1][0]
            hypos = student_generator._generate(sample)
            original_hypos = copy.deepcopy(hypos)
            texts = []
            indices = []
            cut_texts = []
            for i, prediction in enumerate(hypos):
                max_range = len(prediction[0]["tokens"]) - 1
                if max_range <= 1:
                    index = 1
                else:
                    index = randint(1, max_range)
                indices.append(index)
                if samp_mask[i]:
                    student_predictions.append(prediction[0]["tokens"][:index+1])
                    hypos[i][0]["tokens"] = prediction[0]["tokens"][:index+1].cuda()
                else:
                    new_hypo = torch.tensor(
                        prediction[0]["tokens"][:index].data.tolist() +
                        [choice(top_k_predictions[i][index].tolist())]
                    )
                    student_predictions.append(new_hypo)
                    hypos[i][0]["tokens"] = new_hypo.cuda()

            expert_input = self.collate_tokens(
                student_predictions,
                self.expert_vocab_src.pad(),
                self.expert_vocab_src.eos(),
                left_pad=False,
                move_eos_to_beginning=False
            )
        model = model.train()
        return expert_input, texts, cut_texts, indices, hypos, original_hypos

    def transform_source_tokens_into_expert_voc(self, sample):
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
            move_eos_to_beginning=False
        )
        return source_text, src_lengths

    def get_expert_output(self, sample, source_texts, source_lens, expert_input):
        targets = sample["net_input"]["prev_output_tokens"].data.tolist()
        max_length = max([len(i) for i in targets])  # let's avoid blowing up the GPU RAM, shall we?
        out = {
            "id": sample["id"],
            "net_input": {
                "src_tokens": source_texts.cuda(),
                "src_lengths": source_lens,
                "prev_output_tokens": [],
            },
            "target": sample["target"],
            "target_lengths": sample["target_lengths"],
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
        }
        expert_generator = SequenceGenerator(
            [self.expert],
            self.expert_vocab_tgt,
            beam_size=5,
            max_len=max_length
        )
        with torch.no_grad():
            expert_output = expert_generator.generate(
                [self.expert],
                out,
                prefix_tokens=expert_input.cuda()
            )
            for i, j in enumerate(expert_output):
                expert_output[i] = [j[0]]
        return expert_output

    def get_hypos_and_scores(self, sample, lprobs, expert_output_samples, student_hypos, complete_student_hypos):
        sample_expert = sample.copy()
        complete_hypothesis_student = self.add_bleu_to_hypotheses(sample, complete_student_hypos)
        sample_student, hypos_student = self.prepare_sample_and_hypotheses(sample, student_hypos, student=True)
        sample_expert, hypos_expert = self.prepare_sample_and_hypotheses(sample_expert, expert_output_samples)
        sample_student = self.update_sample_with_hypos(sample_student, hypos_student)
        sample_expert = self.update_sample_with_hypos(sample_expert, hypos_expert)
        bzw, _, vocab_len = lprobs.size()
        lengths = Variable(sample_student['hypotheses'].view(bzw, 1, -1).ne(self.dict.pad()).sum(2).float(),
                           requires_grad=False)
        scores = self.get_hypothesis_scores(lprobs, sample)
        return scores, sample_student, sample_expert, lengths, complete_hypothesis_student

    def get_hypothesis_scores(self, lprobs, sample):
        hypotheses = Variable(sample['hypotheses'], requires_grad=False)
        bzw, target_len, vocab_len = lprobs.size()
        hypotheses = hypotheses.view(bzw, 1, -1, 1)
        net_output = lprobs.repeat(1, 1, 1, 1).view(bzw, 1, -1, vocab_len)
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
