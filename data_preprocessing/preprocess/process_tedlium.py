#!/usr/bin/env python3


import pandas as pd


def get_tedlium_source(dataset, output_folder):
    """ Extract target text from tsv dataset files and write it into a new txt file"""
    filenames = ["train.tsv", "dev.tsv", "test.tsv"]
    for filename in filenames:
        dataframe = pd.read_csv(f"{dataset}/{filename}", sep="\t")
        source_text = dataframe.tgt_text.values.tolist()
        with open(f"{output_folder}/{filename.split('.tsv')[0]}.en", "w") as f:
            f.write("\n".join(source_text))


def get_pseudo_labels_from_fairseq_output(output):
    """ Extract auto translation from a fairseq-generate output file and sort them"""
    translation_all_information = [line for line in output if line.startswith("D-")]
    translations = [line.split("\t") for line in translation_all_information]
    translations_in_order = ["" for _ in translations]
    for sentence_tuple in translations:
        index = int(sentence_tuple[0].split("-")[1])
        translations_in_order[index] = sentence_tuple[2]
    return translations_in_order


def create_tedlium_with_nmt_generated_targets(dataset):
    """ Add NMT translations as targets to ASR dataset """
    type2translationfile = {
        "train": "tedlium_train.txt",
        "dev": "tedlium_dev.txt",
        "test": "tedlium_test.txt",
    }
    type2filename = {
        "train": "train.tsv",
        "dev": "dev.tsv",
        "test": "test.tsv",
    }
    for filetype, filename in type2filename.items():
        dataframe = pd.read_csv(f"{dataset}/{filename}", sep="\t")
        pseudo_labels_file = type2translationfile[filetype]
        with open(f"{dataset}/{pseudo_labels_file}") as f:
            pseudo_labels_file = f.read().split("\n")
        pseudo_labels = get_pseudo_labels_from_fairseq_output(pseudo_labels_file)
        dataframe = dataframe.rename(columns={"tgt_text": "src_text"})
        dataframe = dataframe.assign(tgt_text=pd.Series(pseudo_labels))
        new_filename = filename.split('.tsv', maxsplit=1)[0] + "_pseudo_labeled.tsv"
        dataframe.to_csv(f"{dataset}/{new_filename}", sep="\t")


if __name__ == "__main__":
    get_tedlium_source(
        "/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/examples/speech_to_text/tedlium",
        "/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/examples/speech_to_text/tedlium",
    )
    create_tedlium_with_nmt_generated_targets(
        "/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/examples/speech_to_text/tedlium",
    )
