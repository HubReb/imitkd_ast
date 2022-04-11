import os
import sacrebleu

directory = os.fsencode("result_folder_detok")
    
bleu = sacrebleu.BLEU
result_commands = []
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     # print(filename)
     if filename.endswith("sources.txt.detok"): 
         source = "result_folder_detok/" + filename
         hypo = "result_folder_detok/" + filename.split("sources.txt.detok")[0] + "hypothesis.txt.detok"
         command = f"sacrebleu {source} -i {hypo}  > result_scores/{filename.split('_sources.txt.detok')[0] + '_bleu_score'} &"
         result_commands.append(command)

with open("sacrebleu_commands.sh", "w") as f:
    f.write("\n".join(result_commands))
         
