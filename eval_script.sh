# !/usr/bin/env bash


python get_references_and_hypothesis.py $1


cd result_folder
for file in $(ls );
do
    cat $file | sacremoses -l de -q detokenize > ../result_folder_detok/$file.detok &
    sleep 0.01s
done

sleep 2m

python get_sacrebleu_scores.py

bash sacrebleu_commands.sh

