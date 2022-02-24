#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import pickle
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
from numpy.random import uniform
import faiss
import fastBPE
import sentencepiece as spm

import torch
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig


def main(cfg: DictConfig):
    # get args to see if we are dealing with knmmmt
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)


    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)
    # we need model variable for knnmt, so hacking it in here (RH)
    model = models[-1]
    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs, task_extra_args=cfg.task
    )

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)
    from fairseq.checkpoint_utils import load_model_ensemble
    expert, _ = load_model_ensemble(["/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/wmt19.en-de.joined-dict.ensemble/model1.pt"], arg_overrides={"data": "/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/wmt19.en-de.joined-dict.ensemble/", "knn_keytype": "last_ffn_input"})
    expert = expert[-1].cuda()
    from fairseq.sequence_generator import SequenceGenerator
    from fairseq.data import Dictionary
    bpe_expert_src = fastBPE.fastBPE(
        "/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/examples/speech_to_text/bpecodes",
        "wmt19.en-de.joined-dict.ensemble/dict.en.txt"
    )
    expert_vocab_tgt = Dictionary.load("wmt19.en-de.joined-dict.ensemble/dict.de.txt")
    expert_vocab_src = Dictionary.load("wmt19.en-de.joined-dict.ensemble/dict.en.txt")
    knn_args = Namespace(
        knnmt=True,
        knn_keytype="last_ffn_input",
        probe=32,
        dstore_size=100000,
        k=64,
        knn_temp=100,
        indexfile="/home/rebekka/dstores/covost/index_only.80000.index",
        lmbda=0.55,
        knn_sim_func=None,
        faiss_metric_type="do_not_recomp_l2",
        no_load_keys=True,
        move_dstore_to_mem=False,
        use_faiss_only=True,
        dstore_mmap=None,
        knn_embed_dim=1024,
        knn_start=-1,
        knn_proc=-1,
        dstore_fp16=False,
        save_knns=False,
        save_knns_filename=None,
        save_knn_subset=False,
        save_knn_subset_num=1000000,
        knn_add_to_idx=False,
        knn_trim_data=False,
        knn_add_num_to_idx=10000000,
        knn_add_idx_global_id=None,
        knn_add_idx_pos_in_dataset=None,
        knn_q2gpu=False,
        drop_lang_tok=False,
        knn_backoff=False,
        trained_index=False,
    )
    expert_generator = SequenceGenerator([expert], expert_vocab_tgt, beam_size=1)#, args=knn_args)
    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x
    ## knn saving code
    if cfg.task.save_knn_dstore:
        print('keytype being saved:', cfg.task.knn_keytype)
        if cfg.task.knn_start > -1:
            chunk_size = 100000
            if cfg.task.dstore_fp16:
                print('Saving fp16')
                dstore_keys = np.zeros([chunk_size, model.decoder.embed_dim], dtype=np.float16)
                dstore_vals = np.zeros([chunk_size, 1], dtype=np.int16)
            else:
                print('Saving fp32')
                dstore_keys = np.zeros([chunk_size, model.decoder.embed_dim], dtype=np.float32)
                dstore_vals = np.zeros([chunk_size, 1], dtype=np.int)

        else:
            assert not (cfg.task.save_knn_subset and cfg.task.knn_add_to_idx)
            dstore_size = cfg.task.dstore_size
            if cfg.task.save_knn_subset:
                dstore_size = cfg.task.save_knn_subset_num
            if cfg.task.dstore_fp16:
                print('Saving fp16')
                if cfg.task.knn_add_to_idx:
                    faiss_indices = []
                    for tindex in [cfg.task.trained_index]:
                        print("Reading trained index from %s" % tindex)
                        faiss_indices.append(faiss.read_index(tindex))
                        if cfg.task.knn_q2gpu:
                            assert len([cfg.task.trained_index]) == 1
                            print("Moving quantizer to GPU")
                            index_ivf = faiss.extract_index_ivf(faiss_indices[0])
                            quantizer = index_ivf.quantizer
                            quantizer_gpu = faiss.index_cpu_to_all_gpus(quantizer, ngpu=1)
                            index_ivf.quantizer = quantizer_gpu
                else:
                    dstore_keys = np.memmap(cfg.task.dstore_mmap+'_keys.npy', dtype=np.float16, mode='w+', shape=(dstore_size, model.decoder.embed_dim))
                    dstore_vals = np.memmap(cfg.task.dstore_mmap+'_vals.npy', dtype=np.int16, mode='w+', shape=(dstore_size, 1))
            else:
                print('Saving fp32')
                if cfg.task.knn_add_to_idx:
                    faiss_indices = []
                    for tindex in [cfg.task.trained_index]:
                        print("Reading trained index from %s" % tindex)
                        faiss_indices.append(faiss.read_index(tindex))
                        if cfg.task.knn_q2gpu:
                            assert len([cfg.task.trained_index]) == 1
                            print("Moving quantizer to GPU")
                            index_ivf = faiss.extract_index_ivf(faiss_indices[0])
                            quantizer = index_ivf.quantizer
                            quantizer_gpu = faiss.index_cpu_to_all_gpus(quantizer, ngpu=1)
                            index_ivf.quantizer = quantizer_gpu
                else:
                    print(dstore_size, model.decoder.embed_dim)
                    dstore_keys = np.memmap(cfg.task.dstore_mmap+'_keys.npy', dtype=np.float32, mode='w+', shape=(dstore_size, model.decoder.embed_dim))
                    dstore_vals = np.memmap(cfg.task.dstore_mmap+'_vals.npy', dtype=np.int, mode='w+', shape=(dstore_size, 1))

        dstore_idx = 0
        total_saved = 0
        knn_num_samples_proc = 0
        to_skip = -1
        if cfg.task.knn_start > -1:
            to_skip = cfg.task.knn_start # examples
            start_pos = 0
        if cfg.task.knn_add_to_idx:
            adding_to_faiss = 0
        # save the sample ids and the lengths for backtracking the neighbors
        sample_order_lens = [[],[]]
    if cfg.task.knnmt and cfg.task.save_knns:
        to_save_objects = []
    ## knn saving code


    scorer = scoring.build_scorer(cfg.scoring, tgt_dict)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()

    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue
        ## For processing in parallel
        if cfg.task.save_knn_dstore and to_skip > 0:
            num_samples = sample['target'].shape[0]
            if to_skip - num_samples > 0:
                to_skip -= num_samples
                target_tokens = utils.strip_pad(sample['target'], tgt_dict.pad()).int().cpu()
                start_pos += len(target_tokens)
                continue

            for i, sample_id in enumerate(sample['id'].tolist()):
                if to_skip > 0:
                    to_skip -= 1
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()
                    start_pos += len(target_tokens)
                else:
                    tgt_tokens = utils.strip_pad(sample['target'][i:], tgt_dict.pad()).int().cpu()
                    new_sample = {
                            'id': sample['id'][i:],
                            'nsentences': len(sample['id'][i:]),
                            'ntokens': len(tgt_tokens),
                            'net_input': {
                                'src_tokens': sample['net_input']['src_tokens'][i:],
                                'src_lengths': sample['net_input']['src_lengths'][i:],
                                'prev_output_tokens': sample['net_input']['prev_output_tokens'][i:],
                            },
                            'target': sample['target'][i:]

                    }
                    sample = new_sample
                    break

            print('Starting the saving at location %d in the mmap' % start_pos)
        ## For processing in parallel

 
        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()

        if sample["net_input"].get("src_text"):
            sample['net_input'].pop("src_text")
        hypos = task.inference_step(
            generator,
            models,
            # sample_s,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)
        if cfg.task.knn_add_to_idx:
            saving = sample['ntokens']
            if cfg.task.drop_lang_tok:
                saving = sample['ntokens'] - sample['target'].shape[0]
            keys = np.zeros([saving, model.decoder.embed_dim], dtype=np.float32)
            addids = np.zeros([saving], dtype=np.int)
            save_idx = 0
        source_text = sample["net_input"]["src_text"]
        source_texts = []
        for i, line in enumerate(source_text):
            if type(line) == list:
                for text in line:
                    source_texts.append(expert_vocab_src.encode_line(bpe_expert_src([text])[0], add_if_not_exist=False, append_eos=True))
            else:
                if line is None:
                    print(task.tgt_dict.string(sample['target'][i], bpe_symbol="fastBPE"))
                source_texts.append(expert_vocab_src.encode_line(bpe_expert_src.apply([line])[0], add_if_not_exist=False, append_eos=True))
        source_texts = collate_tokens(
            source_texts,
            expert_vocab_src.pad(),
            expert_vocab_src.eos(),
            left_pad=False,
            move_eos_to_beginning=False
        )
        prev_output_tokens = []
        for b, hyp in enumerate(hypos):
            # consider only the top scoring hypothesis
            t = int(uniform(low=1, high=len(hyp[0]["tokens"]), size=None))
            #t = 2
            prev_output_tokens.append(
                hyp[0]["tokens"][:t].clone().detach()
            )
        expert_input = collate_tokens(
            prev_output_tokens,
            expert_vocab_tgt.pad(),
            expert_vocab_tgt.eos(),
            left_pad=False,
            move_eos_to_beginning=False
        )

        out = {
            "id": sample["id"],
            "net_input": {
                "src_tokens": source_texts.cuda(),
                "src_lengths": [len(text) for text in source_texts],
                "prev_output_tokens": [expert_input],
            },
            "target": sample["target"],
            "target_lengths": sample["target_lengths"],
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            }
        hypos = expert_generator.generate(
            [expert],
            out,
            prefix_tokens=expert_input.cuda(),
        )
        for b, hypo in enumerate(hypos):
            for k in range(cfg.generation.nbest):
                hypos[b][k]["tokens"] = hypo[k]["tokens"]
        for i, sample_id in enumerate(sample["id"].tolist()):
            has_target = sample["target"] is not None
            # Remove padding
            if "src_tokens" in sample["net_input"]:
                src_tokens = utils.strip_pad(
                    sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                )
            else:
                src_tokens = None

            target_tokens = None
            if has_target:
                target_tokens = (
                    utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
                )
            ## knn saving code
            if cfg.task.save_knn_dstore:
                hypo = hypos[i][0]
                num_items = len(hypo['tokens'])
                #print(num_items, hypo['dstore_keys_mt'].shape)
                #print(hypo['tokens'])
                #print(hypo['dstore_keys_mt'])
                #exit(0)
                #sample_order_lens[0].append(sample_id)
                #sample_order_lens[1].append(num_items)
                #if dstore_idx + shape[0] > cfg.task.dstore_size:
                #    shape = [cfg.task.dstore_size - dstore_idx]
                #    hypo['dstore_keys_mt'] = hypo['dstore_keys_mt'][:shape[0]]
                if cfg.task.knn_start > -1:
                    if dstore_idx + num_items > dstore_keys.shape[0]:
                        if cfg.task.dstore_fp16:
                            dstore_keys = np.concatenate([dstore_keys, np.zeros([chunk_size, model.decoder.embed_dim], dtype=np.float16)], axis=0)
                            dstore_vals = np.concatenate([dstore_vals, np.zeros([chunk_size, 1], dtype=np.int16)], axis=0)
                        else:
                            dstore_keys = np.concatenate([dstore_keys, np.zeros([chunk_size, model.decoder.embed_dim], dtype=np.float32)], axis=0)
                            dstore_vals = np.concatenate([dstore_vals, np.zeros([chunk_size, 1], dtype=np.int)], axis=0)

                skip = 0
                if cfg.task.drop_lang_tok:
                    skip += 1

                if cfg.task.save_knn_subset:
                    if total_saved + num_items - skip > cfg.task.save_knn_subset_num:
                        num_items = cfg.task.save_knn_subset_num - total_saved + skip

                if cfg.task.knn_add_to_idx:
                    keys[save_idx:save_idx+num_items-skip] = hypo['dstore_keys_mt'][skip:num_items].view(
                            -1, model.decoder.embed_dim).cpu().numpy().astype(np.float32)
                    addids[save_idx:save_idx+num_items-skip] = hypo['tokens'][skip:num_items].view(
                            -1).cpu().numpy().astype(np.int)
                    save_idx += num_items - skip

                if not cfg.task.knn_add_to_idx:
                    if cfg.task.dstore_fp16:
                        dstore_keys[dstore_idx:num_items-skip+dstore_idx] = hypo['dstore_keys_mt'][skip:num_items].view(
                                -1, model.decoder.embed_dim).cpu().numpy().astype(np.float16)
                        dstore_vals[dstore_idx:num_items-skip+dstore_idx] = hypo['tokens'][skip:num_items].view(
                                -1, 1).cpu().numpy().astype(np.int16)
                    else:
                        dstore_keys[dstore_idx:num_items-skip+dstore_idx] = hypo['dstore_keys_mt'][skip:num_items].view(
                                -1, model.decoder.embed_dim).cpu().numpy().astype(np.float32)
                        dstore_vals[dstore_idx:num_items-skip+dstore_idx] = hypo['tokens'][skip:num_items].view(
                                -1, 1).cpu().numpy().astype(np.int)

                dstore_idx += num_items - skip
                total_saved += num_items - skip
                knn_num_samples_proc += 1
            ## knn saving code
            if cfg.generation.score_reference:
                continue

            ## error analysis knnmt: save knns, vals and probs
            if cfg.task.knnmt and cfg.task.save_knns:
                to_save_objects.append(
                        {
                            "id": sample_id,
                            "src": src_tokens,
                            "tgt": target_tokens,
                            "hypo": hypos[i],
                        }
                    )
            ## error analysis knnmt: save knns, vals and probs


            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(cfg.dataset.gen_subset).src.get_original_text(
                    sample_id
                )
                target_str = task.dataset(cfg.dataset.gen_subset).tgt.get_original_text(
                    sample_id
                )
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                else:
                    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens,
                        cfg.common_eval.post_process,
                        escape_unk=True,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                            generator
                        ),
                    )

            src_str = decode_fn(src_str)
            if has_target:
                target_str = decode_fn(target_str)
            if not cfg.common_eval.quiet:
                if src_dict is not None:
                    print("S-{}\t{}".format(sample_id, src_str), file=output_file)
                if has_target:
                    print("T-{}\t{}".format(sample_id, target_str), file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][: cfg.generation.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                if not cfg.common_eval.quiet:
                    score = hypo["score"] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print(
                        "H-{}\t{}\t{}".format(sample_id, score, hypo_str),
                        file=output_file,
                    )
                    # detokenized hypothesis
                    print(
                        "D-{}\t{}\t{}".format(sample_id, score, detok_hypo_str),
                        file=output_file,
                    )
                    print(
                        "P-{}\t{}".format(
                            sample_id,
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    # convert from base e to base 2
                                    hypo["positional_scores"]
                                    .div_(math.log(2))
                                    .tolist(),
                                )
                            ),
                        ),
                        file=output_file,
                    )

                    if cfg.generation.print_alignment == "hard":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in alignment
                                    ]
                                ),
                            ),
                            file=output_file,
                        )
                    if cfg.generation.print_alignment == "soft":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [
                                        ",".join(src_probs)
                                        for src_probs in alignment
                                    ]
                                ),
                            ),
                            file=output_file,
                        )

                    if cfg.generation.print_step:
                        print(
                            "I-{}\t{}".format(sample_id, hypo["steps"]),
                            file=output_file,
                        )

                    if cfg.generation.retain_iter_history:
                        for step, h in enumerate(hypo["history"]):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h["tokens"].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print(
                                "E-{}_{}\t{}".format(sample_id, step, h_str),
                                file=output_file,
                            )

                # Score only the top hypothesis
                if has_target and j == 0:
                    if align_dict is not None or cfg.common_eval.post_process is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(
                            target_str, add_if_not_exist=True
                        )
                        hypo_tokens = tgt_dict.encode_line(
                            detok_hypo_str, add_if_not_exist=True
                        )
                    if hasattr(scorer, "add_string"):
                        scorer.add_string(target_str, detok_hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)

            if cfg.task.knn_start > -1 and knn_num_samples_proc == cfg.knn_proc:
                break
            if cfg.task.save_knn_subset and total_saved >= cfg.task.save_knn_subset_num:
                break
            #if i > 10:
            #    break
        if cfg.task.knn_start > -1 and knn_num_samples_proc == cfg.task.knn_proc:
            break
        if cfg.task.save_knn_subset and total_saved >= cfg.task.save_knn_subset_num:
            break
        if cfg.task.knn_add_to_idx:
            adding_to_faiss += keys.shape[0]
            for fidx in range(len([cfg.task.trained_index])):
                faiss_indices[fidx].add_with_ids(keys, addids)
            #print(f"loop time {time.time()-knn_start_loop}s")

        #print(idx)
        #if idx == 0:
        #    break




        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )
    if cfg.task.knn_q2gpu:
        index_ivf.quantizer = quantizer
        del quantizer_gpu

    if cfg.task.save_knn_dstore:
        if cfg.task.knn_start > -1:
            dstore_keys = dstore_keys[:total_saved]
            dstore_vals = dstore_vals[:total_saved]
            np.savez(cfg.task.dstore_mmap+".keys_vals.%d.%d" % (cfg.task.knn_start, cfg.task.knn_start + knn_num_samples_proc - 1), keys=dstore_keys, vals=dstore_vals)
            print("Final dstore position = %d" % (start_pos + total_saved - 1))
            print("Number of examples processed = %d" % knn_num_samples_proc)
            knn_samples_savefile = cfg.task.dstore_mmap+".samples.%d.%d" % (cfg.task.knn_start, cfg.task.knn_start + knn_num_samples_proc - 1)
        #else:
        #    knn_samples_savefile = cfg.task.dstore_mmap+".samples"
        #np.save(knn_samples_savefile, np.array(sample_order_lens, dtype=np.int))
        print("dstore_idx", dstore_idx, "final number of items added", num_items - skip, "total saved", total_saved)
        if not cfg.task.knn_add_to_idx:
            print("Keys", dstore_keys.shape, dstore_keys.dtype)
            print("Vals", dstore_vals.shape, dstore_vals.dtype)
        else:
            for widx, write_index in enumerate([cfg.task.write_index]):
                faiss.write_index(faiss_indices[widx], write_index)
                print("Added to faiss", adding_to_faiss)
                #print("Final global position %d" % global_end)

    if cfg.task.knnmt and cfg.task.save_knns:
        pickle.dump(to_save_objects, open(cfg.task.save_knns_filename, "wb"))

 
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info(
        "Translated {:,} sentences ({:,} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    if has_target and not cfg.generation.score_reference:
        if cfg.bpe and not cfg.generation.sacrebleu:
            if cfg.common_eval.post_process:
                logger.warning(
                    "BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization"
                )
            else:
                logger.warning(
                    "If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization"
                )
        # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
        print(
            "Generate {} with beam={}: {}".format(
                cfg.dataset.gen_subset, cfg.generation.beam, scorer.result_string()
            ),
            file=output_file,
        )

    return scorer


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


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


if __name__ == "__main__":
    cli_main()
