import pandas as pd


def read_file(filename):
    with open(filename) as f:
        data = f.read().split("\n")[:-1]
    return data


def get_references_and_hypotheses(data):
    references2hypos = {}
    for line in data:
        if line.startswith("T-"):
            reference = " ".join(line.split()[1:])
        if line.startswith("D-"):
            output_hypo = " ".join(line.split()[2:])
            references2hypos[reference] = output_hypo
    return references2hypos


def create_generated_dateset(reference2hypos, dataset, reference2transcripts=None):
    dataframe = pd.read_csv(dataset, sep="\t")
    # get values for quicker access
    data_values = dataframe.values
    # dataframe has to have form (id, audio, n_frames, tgt_text, speaker, src_text)
    for i, row in enumerate(data_values):
        try:
            if len(row) > 6:
                data_values[i][6] = reference2hypos[row[6]]
                if reference2transcripts:
                    data_values[i][8] = reference2transcripts[row[8]]
            else:
                data_values[i][3] = reference2hypos[row[3]]
                if reference2transcripts:
                    data_values[i][5] = reference2transcripts[row[5]]
        except KeyError:
            print(row)
    if len(row) == 6:
        new_data_frame = pd.DataFrame(data_values, columns=["id", "audio", "n_frames", "tgt_text", "speaker", "src_text"])
    else:
        new_data_frame = pd.DataFrame(data_values,
                                      columns=["Untitled", "Untitled:1", "Untitled:2", "id", "audio", "n_frames", "tgt_text", "speaker", "src_text"])
    return new_data_frame


def run(generation_file, dataset_file, audiotranscriptsfile=None):
    reference2hypos = get_references_and_hypotheses(read_file(generation_file))
    if audiotranscriptsfile:
        print(audiotranscriptsfile)
        audio_transcripts = get_references_and_hypotheses(read_file(audiotranscriptsfile))
        dataset = create_generated_dateset(reference2hypos, dataset_file, audio_transcripts)
    else:
        dataset = create_generated_dateset(reference2hypos, dataset_file)
    if audiotranscriptsfile:
        dataset.to_csv(f"{dataset_file.split('.tsv')[0]}_wmt19_generated_from_transcripts.tsv", index=False, sep="\t")
    else:
        dataset.to_csv(f"{dataset_file.split('.tsv')[0]}_wmt19_generated.tsv", index=False, sep="\t")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="create dataset from wmt19 transformer output")
    parser.add_argument('-o', '--output', help="files containing the output of fairseq-generate", required=True)
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-a', '--audiotranscripts', required=False, default=None)
    args = parser.parse_args()
    if args.audiotranscripts:
        run(args.output, args.dataset, args.audiotranscripts)
    run(args.output, args.dataset)
