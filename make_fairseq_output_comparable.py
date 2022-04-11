#! /usr/bin/env python3

def read_file(filename):
    with open(filename) as f:
        data = f.read().split("\n")[:-1]
    return data


def combine_source_and_hypothesis_in_dictionary(source_data, hypothesis_data):
    source_to_hypo_dict = {}
    for line_source, line_hypo in zip(source_data, hypothesis_data):
        source_to_hypo_dict[line_source] = line_hypo
    return source_to_hypo_dict


def sort_data_streams(reference_list, args):
    if len(args) < 2:
        raise TypeError("At least two source/hypo dictionaries are required!")
    new_hypo_files = []
    for dictionary in args:
        new_hypo_order = []
        for key in reference_list:
            new_hypo_order.append(dictionary[key])
        new_hypo_files.append(new_hypo_order)
    return new_hypo_files


def write_data_to_file(filename, data):
    with open(f"/home/rebekka/reordered_hypos/{filename.split('/')[-1].split('.txt')[0] + '_ordered.txt'}", "w") as f:
        assert isinstance(data, list)
        f.write("\n".join(data))


def check_if_folder_exits_or_create(foldername):
    from os.path import isdir

    if isdir(foldername):
        return
    import os
    os.makedirs(foldername)


def run(source_filename, translation_filenames):
    assert "source" in source_filename
    foldername = "/home/rebekka/reordered_hypos"
    print(foldername)
    check_if_folder_exits_or_create(foldername)
    original_data = []
    for filename in translation_filenames:
        original_data.append(read_file(filename))
    source_data = read_file(source_filename)
    dictionary_list = []
    for index, data_stream in enumerate(original_data):
        assert "hypo" in translation_filenames[index]
        dictionary_list.append(combine_source_and_hypothesis_in_dictionary(source_data, data_stream))
    new_hypo_data = sort_data_streams(source_data, dictionary_list)
    for index, data in enumerate(new_hypo_data):
        write_data_to_file(translation_filenames[index], data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="handle different orders of fairseq-generate outputs")
    parser.add_argument('-s', '--source_references', help="file containing the reference sentences line by line", required=True)
    parser.add_argument('-t', '--translations', nargs='+', required=True)
    args = parser.parse_args()
    run(args.source_references, args.translations)
