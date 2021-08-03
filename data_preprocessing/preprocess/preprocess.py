#!/usr/bin/env python3
# author: R. Hubert
# email: hubert@cl.uni-heidelberg.de


from translate.storage.tmx import tmxfile
import xml.etree.cElementTree as ET

def get_corpora_from_tmx(filename, output_directory, languages):
    source, target = [], []
    iter_variable = 0
    for event, elem in ET.iterparse(filename, events=("start", "end")):
        if event == "start" and elem.tag == "seg":
            if iter_variable % 2 == 0:
                if not elem.text == None:
                    source.append(elem.text.strip())
            else:
                if not elem.text == None:
                    target.append(elem.text.strip())
            iter_variable += 1
            if iter_variable % 1000000 == 0:
                write_corpus("\n".join(source), output_directory, languages[0])
                source = []
                write_corpus("\n".join(target), output_directory, languages[1])
                target = []

def write_corpus(corpus, output_directory, language):
    with open(f"{output_directory}/{language}.corpus.txt", "a") as f:
        f.write(corpus)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data', metavar='D', type=str, help='dataset file')
    parser.add_argument('--output', metavar='O', type=str, help='directory to store corpora in')
    args = parser.parse_args()
    get_corpora_from_tmx(args.data, args.output, ("en", "de"))

