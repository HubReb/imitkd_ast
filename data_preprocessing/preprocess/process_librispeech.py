#!/usr/bin/env python3


import pandas as pd
from process_tedlium import (
    get_pseudo_labels_from_fairseq_output,
)


def get_librispeech_source(dataset, output_folder):
    train_filenames = [
        "train-clean-100.tsv",
        "train-clean-360.tsv",
        "train-other-500.tsv",
    ]
    dev_filenames = ["dev-clean.tsv", "dev-other.tsv"]
    test_filenames = ["test-clean.tsv", "test-other.tsv"]
    type2filename = {
        "train": train_filenames,
        "dev": dev_filenames,
        "test": test_filenames,
    }
    for filetype, filenames in type2filename.items():
        entire_set = []
        for filename in filenames:
            dataframe = pd.read_csv(f"{dataset}/{filename}", sep="\t")
            source_text = dataframe.tgt_text.values.tolist()
            entire_set.append("\n".join(source_text))
        with open(f"{output_folder}/{filetype}.en", "w") as f:
            f.write("\n".join(entire_set))


def create_librispeech_with_nmt_generated_targets(dataset):
    train_filenames = [
        "train-clean-100.tsv",
        "train-clean-360.tsv",
        "train-other-500.tsv",
    ]
    dev_filenames = ["dev-clean.tsv", "dev-other.tsv"]
    test_filenames = ["test-clean.tsv", "test-other.tsv"]
    type2filename = {
        "train": train_filenames,
        "dev": dev_filenames,
        "test": test_filenames,
    }
    type2translationfile = {
        "train": "libri_train.txt",
        "dev": "libri_dev.txt",
        "test": "libri_test.txt",
    }
    for filetype, filenames in type2filename.items():
        entire_dataframe = []
        for filename in filenames:
            dataframe = pd.read_csv(f"{dataset}/{filename}", sep="\t")
            entire_dataframe.append(dataframe)
        complete_dataframe = pd.concat(entire_dataframe)
        pseudo_labels_file = type2translationfile[filetype]
        with open(f"{dataset}/{pseudo_labels_file}") as f:
            pseudo_labels_file = f.read().split("\n")
        pseudo_labels = get_pseudo_labels_from_fairseq_output(pseudo_labels_file)
        complete_dataframe = complete_dataframe.rename(columns={"tgt_text": "src_text"})
        complete_dataframe = complete_dataframe.assign(
            tgt_text=pd.Series(pseudo_labels)
        )
        new_filename = filename.split('-', maxsplit=1)[0] + "_pseudo_labeled.tsv"
        complete_dataframe.to_csv(f"{dataset}/{new_filename}", sep="\t")


if __name__ == "__main__":
    get_librispeech_source(
        "/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/knnmt/data/Librispeech",
        "/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/knnmt/data/Librispeech",
    )
    create_librispeech_with_nmt_generated_targets(
        "/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/knnmt/data/Librispeech",
    )
