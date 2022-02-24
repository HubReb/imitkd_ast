#!/usr/bin/env python3


import pandas as pd


def get_librivox_source(dataset, output_folder):
    dataset = pd.read_csv(f"{dataset}", sep="\t")
    books = list(set(dataset["book"].values.tolist()))
    train_set = books[:int(len(books) * 0.75)]
    valid_set = books[int(len(books) * 0.75):int(len(books) * 0.85)]
    test_set = books[int(len(books) * 0.9):]
    datasets = {"train": train_set, "dev": valid_set, "test": test_set}
    for i, d_set in datasets.items():
        with open(f"{output_folder}/{i}_booklist.txt", "w") as f:
            f.write("\n".join(d_set))
        dataset_split = dataset[dataset["book"].isin(d_set)]
        dataset_split = dataset_split.dropna()
        source = [t.split("\n")[0] for t in dataset_split.en_sentence.values.tolist()]
        translation = [t.split("\n")[0] for t in dataset_split.de_sentence.values.tolist()]
        print(i, len(source))
        assert len(source) == len(translation)
        with open(f"{output_folder}/{i}_librivox.en", "w") as f:
            f.write("\n".join(source))
        with open(f"{output_folder}/{i}_librivox.de", "w") as f:
            f.write("\n".join(translation))


if __name__ == "__main__":
    get_librivox_source(
        "/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/data/librivox/librivoxdeen/tables/text2text.tsv",
        "/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/fairseq/data/librivox",
    )
