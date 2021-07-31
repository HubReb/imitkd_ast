#!/usr/bin/env python3
# author: R. Hubert
# email: hubert@cl.uni-heidelberg.de


import regex as re


def change_format_news(data):
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
            source_sentence, translated_sentence = (
                line.split("\t")[0],
                line.split("\t")[-1],
            )
            assert translated_sentence != []
            assert source_sentence != []
            assert source_sentence != translated_sentence
            if len(line.split("\t")) > 2:
                pattern = r"^1\d."
                if (
                    "" not in line.split("\t")
                    and len(line.split("\t")) == 3
                    and re.match(pattern, line.split("\t")[0])
                ):
                    source_sentence = source_sentence.strip() + " " + line.split("\t")[1].strip()
                else:
                    translated_sentence = " ".join(line.split("\t")[1:])
                    translated_sentence = re.sub(r"\s\s+", " ", translated_sentence)
        source_sentence = source_sentence.strip()
        translated_sentence = translated_sentence.strip()
        source.append(source_sentence)
        translation.append(translated_sentence)
    non_doubled_source_sentences, non_doubled_translation_sentences = [], []
    for source_sentence, translated_sentence in zip(source, translation):
        if source_sentence not in non_doubled_source_sentences:
            non_doubled_source_sentences.append(source_sentence)
            non_doubled_translation_sentences.append(translated_sentence)
    return non_doubled_source_sentences, non_doubled_translation_sentences


def split_news_into_seperate_file_per_language(folder, filename):
    with open(f"{folder}/{filename}") as f:
        file_content = f.read().split("\n")[:-1]
    source_language, target_language = change_format_news(file_content)
    language_one, language_two = filename.split(".")[-2].split("-")
    texts = (source_language, target_language)
    for i, language in enumerate((language_one, language_two)):
        with open(f"{folder}/news-commentary-v14.{language}.tsv", "w") as f:
            f.write("\n".join(texts[i]))


if __name__ == "__main__":
    DATADIR = "/home/rebekka/t2b/Projekte/ma/knn_ast_kd_nmt/knnmt/data/"
    filename = "news-commentary-v14.de-en.tsv"
    split_news_into_seperate_file_per_language(
        DATADIR, filename
    )
