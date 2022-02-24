#!/usr/bin/env python3


import pandas as pd


def get_covost_source(dataset, output_folder):
    filename_tuples= [
        ("train_st_en_de.tsv", "train_asr_en.tsv"),
        ("dev_st_en_de.tsv", "dev_asr_en.tsv"),
        ("test_st_en_de.tsv", "test_asr_en.tsv")
    ]
    for filename_tuple in filename_tuples:
        source = pd.read_csv(f"{dataset}/{filename_tuple[1]}", sep="\t")
        target = pd.read_csv(f"{dataset}/{filename_tuple[0]}", sep="\t")
        translation = [t.split("\n")[0] for t in target.tgt_text.values.tolist()]
        source_text = []
        # ensure same order and remove english audio not in en->de set
        source_dict = {}
        source_info = source.values.tolist()
        for tuple_info in source_info:
            source_dict[tuple_info[0]] = tuple_info[3]
        for id_covost in target.id.tolist():
            source_text.append(source_dict[id_covost])
        print(len(source_text), len(translation))
        print(len("\n".join(translation).split("\n")))
        assert len(source_text) == len(translation)
        with open(f"{output_folder}/{filename_tuple[0].split('_st_')[0]}.en", "w") as f:
            f.write("\n".join(source_text))
        with open(f"{output_folder}/{filename_tuple[0].split('_st_')[0]}.de", "w") as f:
            f.write("\n".join(translation))

def get_covost_source_alone(dataset, output_folder):
    filenames= [
        "train_asr_en.tsv", "dev_asr_en.tsv",  "test_asr_en.tsv"
    ]
    for filename in filenames:
        source = pd.read_csv(f"{dataset}/{filename}", sep="\t")
        source_text = []
        source_dict = {}
        source_info = source.values.tolist()
        for tuple_info in source_info:
            source_dict[tuple_info[0]] = tuple_info[3]
        for id_covost in source.id.tolist():
            source_text.append(str(source_dict[id_covost]))
        with open(f"{output_folder}/{filename.split('.tsv')[0]}.txt", "w") as f:
            f.write("\n".join(source_text))

if __name__ == "__main__":
    get_covost_source_alone(
        "/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/knnmt/data/covost/en",
        "/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/knnmt/data/covost/en",
    )
