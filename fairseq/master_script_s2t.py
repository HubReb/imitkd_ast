# adapted from https://github.com/urvashik/knnmt/blob/master/master_script.py
import argparse
import itertools
import os
import math
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--log-folder', type=str, help='Folder to log stdout and stderr')
# Model params
parser.add_argument('--bytes-per-token', type=int, default=2, help='check the binary file size to be calculate (size of file / number of tokens), 2 when using the wmt19 vocab')
parser.add_argument('--model', type=str, help='pytorch checkpoint to use')
parser.add_argument('--config', type=str)
# dstore saving params
parser.add_argument('--dstore-size', type=int, action='append', default=None, help='list of sizes of dstores in number of tokens')
parser.add_argument('--save-data', type=str, action='append', help='list of datasets to save in dstore')
parser.add_argument('--num-shards', type=int, action='append', help='number of shards for each file')
parser.add_argument('--dstore-mmap', type=str, help='dstore location for datasets')
parser.add_argument('--num-for-training', type=int, help='number of items to save per dataset for training the index')
parser.add_argument('--code-size', type=int, default=64, help='size that vectors are quantized to')
parser.add_argument('--decoder-dimension', type=int, default=256, help='size of the final decoder embedding layer')
parser.add_argument('--ncentroids', type=int, action='append', help='number of faiss clusters')
parser.add_argument('--train-index', type=str, action='append',  help='list of files to use for the trained faiss index, must be the same length as ncentroids')
parser.add_argument('--faiss-index', type=str, action='append', help='list of files to use for the faiss index')
parser.add_argument('--write-merged-index', type=str, action='append', help='list of files to use for the faiss index')
parser.add_argument('--corpus-identifiers', type=str, action='append', help='list of ids to use for the distributed faiss indices')
# run job params
parser.add_argument('--save-job', action='store_true', default=False)
parser.add_argument('--merge-dstore-job', action='store_true', default=False)
parser.add_argument('--train-index-job', action='store_true', default=False)
parser.add_argument('--add-keys-job', action='store_true', default=False)
parser.add_argument('--merge-index-job', action='store_true', default=False)
parser.add_argument('--max-tokens', type=int, default=50000)
args = parser.parse_args()


dstore_size = args.dstore_size
if not dstore_size:
    dstore_size = []
    for dataset in args.save_data:
        filestats = os.stat(dataset)
        size = filestats.st_size / args.bytes_per_token # 2 bytes per token for most
        dstore_size.append(int(size))

        print("%s with num tokens %d" % (dataset, size))

# SAVE KEYS/VALUES SUBSET FOR TRAINING
if args.save_job:
    save_jobs = []
    for dataset_idx, curr_save_data in enumerate(args.save_data):
        curr_save_subset_mmap = args.dstore_mmap + ".subset." + str(dataset_idx)
        save_cmd = f"fairseq-generate {curr_save_data} --config-yaml {args.config} --gen-subset train_st --task speech_to_text  --path {args.model} --max-tokens {args.max_tokens} --beam 5 --scoring sacrebleu --score-reference --dstore-mmap {curr_save_subset_mmap} --knn-keytype last_ffn_input  --model-overrides " + "\"{\'knn_keytype\':\'last_ffn_input\'}\"" + f" --save-knn-dstore --save-knn-subset --save-knn-subset-num {args.num_for_training} --quiet"
        print(save_cmd)
# SAVE KEYS/VALUES SUBSET FOR TRAINING

# MERGE SUBSET KEYS/VALUES
if args.merge_dstore_job:
    print("Merging subsets saved for training")
    num_datasets = len(args.save_data)
    merge_subset_cmd = f"python merge_subset_dstores.py --dstore_mmap {args.dstore_mmap} --num_datasets {num_datasets} --size {args.num_for_training} --dimension {args.decoder_dimension}"
    print(merge_subset_cmd)
# MERGE SUBSET KEYS/VALUES

# TRAIN INDEX
if args.train_index_job:
    train_jobs = []
    assert len(args.ncentroids) == len(args.train_index)
    dstore_mmap = args.dstore_mmap + ".subset"
    size = len(args.save_data) * args.num_for_training
    for ncentroid, train_index in zip(args.ncentroids, args.train_index):
        print("echo \"Training index with %d centroids\"" % (ncentroid))
        train_cmd = f"python train_index.py --dstore_mmap {dstore_mmap} --dstore_size {size} --dimension {args.decoder_dimension} --code_size {args.code_size} --ncentroids {ncentroid} --train_index {train_index} --from_subset --gpu"
        print(train_cmd)
# TRAIN INDEX

# Add keys to an already trained index.
# This is done from multiple files, passed through the command line using append.
if args.add_keys_job:
    print("Adding keys to the faiss index")
    add_jobs = []
    assert len(args.train_index) == len(args.faiss_index)
    assert len(args.save_data) == len(args.corpus_identifiers)
    assert len(args.save_data) == len(args.num_shards)
    total_added = 0
    for dataset_idx, (curr_save_data, curr_dstore_size, num_shards) in enumerate(zip(args.save_data, dstore_size, args.num_shards)):
        print("Saving %s, of size %d" % (curr_save_data, curr_dstore_size))
        # iterations it will take to add all keys to index
        index_id = 0 # which index is being written
        train_index = " ".join([f"--trained-index {tindex}" for tindex in args.train_index])
        for shard_idx in range(num_shards):
            write_index = " ".join([f"--write-index {faiss_index}.{args.corpus_identifiers[dataset_idx]}.{index_id}" for faiss_index in args.faiss_index])
            add_cmd = f"fairseq-generate {curr_save_data} --config-yaml {args.config} --gen-subset train_st --task speech_to_text  --path {args.model} --max-tokens {args.max_tokens} --beam 5 --scoring sacrebleu --score-reference --knn-keytype last_ffn_input  --model-overrides " + "\"{\'knn_keytype\':\'last_ffn_input\'}\"" + f" --save-knn-dstore  --knn-add-to-idx --num-shards {num_shards} --shard-id {shard_idx} {train_index} {write_index} --quiet --knn-q2gpu" 
            print(add_cmd)
            index_id += 1 # remember this is 1 greater than the actual ids for indices, i.e. there are index_id number of indices but the last one is index_id - 1.

        print("echo \"Number of indices for this dataset %d\"" % (index_id))
    print("echo \"Total keys meant to be added = %d\"" % (sum(dstore_size)))


    for job in add_jobs:
        job.result()
# Add keys to an already trained index.

# MERGE FAISS INDICES
if args.merge_index_job:
    merge_index_jobs = []
    print('echo "Merging indices"')
    corpus_identifiers = " ".join([f"--corpus_identifiers {cid}" for cid in args.corpus_identifiers])
    num_shards = " ".join([f"--num_shards_per_file {ns}" for ns in args.num_shards])
    for tindex, findex, wmindex in zip(args.train_index, args.faiss_index, args.write_merged_index):
        merge_idx_cmd = f"python merge_index.py --faiss_index {findex} --train_index {tindex} {corpus_identifiers} --write_index {wmindex} {num_shards}"
        print(merge_idx_cmd)

    for job in merge_index_jobs:
        job.result()
# MERGE FAISS INDICES
