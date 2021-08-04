#!/usr/bin/env python3
# author: R. Hubert
# email: hubert@cl.uni-heidelberg.de


from translate.storage.tmx import tmxfile
import xml.etree.cElementTree as ET

def get_corpora_from_tmx(filename):
    source, target = [], []
    iter_variable = 0
    for event, elem in ET.iterparse(filename, events=("start", "end")):
        if event == "start" and elem.tag == "seg":
            if iter_variable % 2 == 0:
                if elem.text is not None:
                    source.append(elem.text.strip())
                    iter_variable += 1
            else:
                if elem.text is not None:
                    target.append(elem.text.strip())
                    assert len(target) == len(source)
                    iter_variable += 1
        elem.clear()
    return "\n".join(source), "\n".join(target)

def write_corpus(corpus, output_directory, language):
    with open(f"{output_directory}/{language}.corpus.txt", "w") as f:
        f.write(corpus)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data', metavar='D', type=str, help='dataset file')
    parser.add_argument('--output', metavar='O', type=str, help='directory to store corpora in')
    args = parser.parse_args()
    source_corpus, target_corpus = get_corpora_from_tmx(args.data)
    write_corpus(source_corpus, args.output, "en")
    write_corpus(target_corpus, args.output, "de")
