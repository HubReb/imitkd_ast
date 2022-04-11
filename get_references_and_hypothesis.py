import argparse

def read_file(filename):
    with open(filename) as f:
        data = f.read().split("\n")[:-1]
    return data


def get_references(data):
    references = []
    for line in data:
        if line.startswith("T-"):
            references.append(" ".join(line.split()[1:]))
    return references


def get_output_hypothesis(data):
    output_hypos = []
    for line in data:
        if line.startswith("D-"):
            output_hypos.append(" ".join(line.split()[2:]))
    return output_hypos




def extract_hypos_and_source(filenames):
    filename_list = read_file(filenames)
    for filename in filename_list:
        print(filename)
        filename = filename.split()[-1]
        data = read_file(filename)
        references = get_references(data)
        hypos = get_output_hypothesis(data)
        if "/" in filename:
            filename = filename.split("/")[-1]
        hypo_file = "result_folder/" +filename.split(".txt")[0] + "_hypothesis.txt"
        source_file = "result_folder/" + filename.split(".txt")[0] + "_sources.txt"
        with open(hypo_file, "w") as f:
            f.write("\n".join(hypos))
        with open(source_file, "w") as f:
            f.write("\n".join(references))





parser = argparse.ArgumentParser()
parser.add_argument("log_file_list", help="name of fairseq-generate log file")
args = parser.parse_args()
fairseq_file = args.log_file_list
extract_hypos_and_source(fairseq_file)

