#!/usr/bin/env python3
# author: R. Hubert
# email: hubert@cl.uni-heidelberg.de


def change_format(data):
    source, translation = [], []
    if not isinstance(data, list):
        raise TypeError
    if [] == data:
        raise IndexError("empty list given")
    for line in data:
        try:
            source_sentence, translated_sentence = line.split("\t")
        # check if we have multiple tabs
        except ValueError:
            source_sentence, translated_sentence = line.split("\t")[0], line.split("\t")[-1]
            assert translated_sentence != []
            assert source_sentence != []
            assert source_sentence != translated_sentence
            if line.split("\t")[1] != []:
                raise ValueError("Format is broken!")
        source_sentence = source_sentence.strip()
        translated_sentence = translated_sentence.strip()
        if source_sentence == [] and translated_sentence == []:
            print("warning: empty line encountered")
            continue
        if source_sentence == [] or translated_sentence == []:
            raise ValueError("could not sucessfully split line - abort")
        if source_sentence not in source:
            source.append(source_sentence)
        if translated_sentence not in translation:
            translation.append(translated_sentence)
    return source, translation


def split_europarl_into_seperate_file_per_language(folder, filename):
    with open(f"{folder}/{filename}") as f:
        file_content = f.read().split("\n")
    source_language, target_language = change_format(file_content)
    language_one, language_two = filename.split(".")[-2].split("-")
    for language in (language_one, language_two):
        with open(f"{folder}/europarl-v9.{language}.tsv", "w") as f:
            f.write("\n".join(source_language))


if __name__ == "__main__":
    DATADIR = "/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/knnmt/data/"
    filename = "europarl-v9.de-en.tsv"
    split_europarl_into_seperate_file_per_language(DATADIR, filename)
