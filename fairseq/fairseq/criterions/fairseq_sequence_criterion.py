# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

from torch.autograd import Variable
import torch.nn.functional as F

from fairseq.criterions import LegacyFairseqCriterion
import torch
import numpy as np
import sentencepiece as spm
import torch._utils
from sacremoses import MosesDetokenizer
from nltk.tokenize import TreebankWordTokenizer

from fairseq import tokenizer, utils
from fairseq import sequence_generator
from fairseq.scoring import bleu
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor

    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


class FairseqSequenceCriterion(LegacyFairseqCriterion):
    """Base class for sequence-level criterions."""

    def __init__(self, args, dst_dict):
        super().__init__(args, dst_dict)
        self.args = args
        self.dst_dict = dst_dict
        self.pad_idx = dst_dict.pad()
        self.eos_idx = dst_dict.eos()
        self.unk_idx = dst_dict.unk()
        self._generator = None
        self._scorer = None

#
    # Methods to be defined in sequence-level criterions
    #

    def prepare_sample_and_hypotheses(self, model, sample, hypos):
        """Apply criterion-specific modifications to the given sample/hypotheses."""
        return sample, hypos

    def sequence_forward(self, net_output, model, sample):
        """Compute the sequence-level loss for the given hypotheses.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError

    @staticmethod
    def grad_denom(sample_sizes):
        """Compute the gradient denominator for a set of sample sizes."""
        return sum(sample_sizes)

    #
    # Helper methods
    #

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

        if self._scorer is None:
            self.create_sequence_scorer()

        target = sample['target'].data.int()
        for i, hypos_i in enumerate(hypos):
            ref = utils.strip_pad(target[i, :], self.pad_idx).cpu()
            r = self.dst_dict.string(ref, bpe_symbol='sentencepiece', escape_unk=True)
            r = tokenizer.Tokenizer.tokenize(r, self.dst_dict, add_if_not_exist=True)
            for hypo in hypos_i:
                h = self.dst_dict.string(hypo['tokens'].int().cpu(), bpe_symbol='sentencepiece')
                h = tokenizer.Tokenizer.tokenize(h, self.dst_dict, add_if_not_exist=True)
                # use +1 smoothing for sentence BLEU
                hypo['bleu'] = self._scorer.score(r, h)
        return hypos

    def create_sequence_scorer(self):
        if self.args.seq_scorer == "bleu":
            self._scorer = BleuScorer(self.pad_idx, self.eos_idx, self.unk_idx)
        else:
            raise Exception("Unknown sequence scorer {}".format(self.args.seq_scorer))

    def get_hypothesis_scores(self, net_output, sample, score_pad=False):
        """Return a tensor of model scores for each hypothesis.

        The returned tensor has dimensions [bsz, nhypos, hypolen]. This can be
        called from sequence_forward.
        """
        bsz, target_len, vocab_len = net_output.size()
        # hard code 5 hypothesis
        hypotheses = Variable(sample['hypotheses'], requires_grad=False).view(bsz, 8, -1, 1)
        net_output = net_output.repeat(1, 8, 1, 1).view(bsz, 8, -1, vocab_len)
        h_shape = hypotheses.shape[2]
        n_shape = net_output.shape[2]
        if h_shape > n_shape:
            hypotheses = hypotheses[:, :, :n_shape, :]
        # we sum over the scores in sequence_forward, so we can cut off the 0 probs here as prob would be 0
        elif h_shape < n_shape:
            net_output = net_output[:, :, :h_shape, :]
        scores = net_output.gather(3, hypotheses)
        if not score_pad:
            scores = scores * hypotheses.ne(self.pad_idx).float()
        return scores.squeeze(3)

    def get_hypothesis_lengths(self, net_output, sample):
        """Return a tensor of hypothesis lengths.

        The returned tensor has dimensions [bsz, nhypos]. This can be called
        from sequence_forward.
        """
        bsz, target_len, vocab_len = net_output.size()
        # hard code values here - so we don't spend even more time
        lengths = sample['hypotheses'].view(bsz, 8, -1).ne(self.pad_idx).sum(2).float()
        return Variable(lengths, requires_grad=False)

    #
    # Methods required by FairseqCriterion
    #

    def forward(self, model, sample):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model_state = model.training
        if model_state:
            model.train(self.args.seq_hypos_dropout)

        # generate hypotheses
        hypos = self._generate_hypotheses(model, sample)

        model.train(model_state)

        # apply any criterion-specific modifications to the sample/hypotheses
        sample, hypos = self.prepare_sample_and_hypotheses(model, sample, hypos)

        # create a new sample out of the hypotheses
        sample = self._update_sample_with_hypos(sample, hypos)


        # run forward and get sequence-level loss
        net_output = self.get_net_output(model, sample)
        loss, sample_size, logging_output = self.sequence_forward(net_output, model, sample)

        return loss, sample_size, logging_output

    def get_net_output(self, model, sample):
        """Return model outputs as log probabilities."""
        net_output = model(**sample['net_input'])
        # S2T transformer returns tuple for fairseq comp., so change this
        # let's stick to the simple form for now - might have best chances of succeeding
        lprobs =  utils.log_softmax(net_output[0], dim=-1)
        return lprobs


    def _generate_hypotheses(self, model, sample):
        args = self.args

        # initialize generator
        if self._generator is None:
            self._generator = sequence_generator.SequenceGenerator(
                [model], self.dst_dict, unk_penalty=args.seq_unkpen, beam_size=8, max_len_a=args.seq_max_len_a,
                max_len_b=args.seq_max_len_b)
            self._generator.cuda()

        # generate hypotheses
        # input = sample['net_input']
        hypos = self._generator._generate(
            sample,
            )

        # add reference to the set of hypotheses
        if self.args.seq_keep_reference:
            hypos = self.add_reference_to_hypotheses(sample, hypos)

        return hypos

    def _update_sample_with_hypos(self, sample, hypos):
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
            input_hypos, self.pad_idx, self.eos_idx, left_pad=False, move_eos_to_beginning=False)
        # only needed for authors' model
        # input['input_tokens'] = self.collate_tokens(
            # input_hypos, self.pad_idx, self.eos_idx, left_pad=True, move_eos_to_beginning=True)
        # input['input_positions'] = self.collate_positions(
            # input_hypos, self.pad_idx, left_pad=True)

        # sample['target'].data = repeat_num_hypos_times(sample['target'].data, dim2=True)
        sample['ntokens'] = sample['target'].data.ne(self.pad_idx).sum()
        sample['bsz'] = bsz
        sample['num_hypos_per_batch'] = num_hypos_per_batch
        return sample

    def collate_tokens(self, values, pad_idx, eos_idx, left_pad, move_eos_to_beginning):
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            if left_pad:
                copy_tensor(v, res[i][size-len(v):])
            else:
                copy_tensor(v, res[i][:len(v)])
        return res

    def collate_positions(self, values, pad_idx, left_pad):
        start = pad_idx + 1
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)
        for i, v in enumerate(values):
            if left_pad:
                torch.arange(start, start + len(v), out=res[i][size-len(v):])
            else:
                torch.arange(start, start + len(v), out=res[i][:len(v)])
        return res


    @staticmethod
    def add_args(parser):
        parser.add_argument('--seq-keep-reference', action='store_true',
                           help='keep the reference in the set of hypotheses')
        parser.add_argument('--seq-max-len-a', default=0, type=float, metavar='N',
                           help=('generate sequences of maximum length ax + b, '
                                 'where x is the source length'))
        parser.add_argument('--seq-max-len-b', default=200, type=int, metavar='N',
                           help=('generate sequences of maximum length ax + b, '
                                 'where x is the source length'))
        parser.add_argument('--seq-combined-loss-alpha', metavar='D', default=0, type=float,
                           help='combined loss = \\alpha*token_loss + seq_loss')
        parser.add_argument('--seq-scorer', metavar='SCORER',
                           help='Optimization metric for sequence level training', default='bleu')
        parser.add_argument('--seq-risk-normbleu', action='store_true',
                           help='Normalize bleu')
        parser.add_argument('--seq-beam', default=8, type=int, metavar='N',
                       help='beam size for sequence training')

        parser.add_argument('--seq-unkpen', default=0, type=float,
                           help='unknown word penalty to be used in seq generation')

        parser.add_argument('--seq-hypos-dropout', action='store_true',
                           help="Use dropout to generate hypos")

        parser.add_argument('--seq-margin-cost-scale-factor', type=float, default=1, metavar='D',
                           help='Scale optimized metric with respect to token loss, '
                                'only relevant for margin losses')




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

