import os
import sacrebleu

directory = os.fsencode("result_folder_detok")

bleu = sacrebleu.BLEU
result_commands = []
for i, file in enumerate(os.listdir(directory)):
    filename = os.fsdecode(file)
    # print(filename)
    if filename.endswith("sources.txt.detok"):
        source = "result_folder_detok/" + filename
        hypo = (
            "result_folder_detok/"
            + filename.split("sources.txt.detok")[0]
            + "hypothesis.txt.detok"
        )
        command = f"sacrebleu {source} -i {hypo} -m bleu ter chrf --chrf-word-order 2 > result_scores/{filename.split('_sources.txt.detok')[0] + 'detokenized_bleu_score'} &"
        result_commands.append(command)
    if i > 0 and i % 42 == 0:
        result_commands.append("sleep 20s")

with open("sacrebleu_commands.sh", "w") as f:
    f.write("\n".join(result_commands))
