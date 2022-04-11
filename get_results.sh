python get_references_and_hypothesis.py file_names.txt


cd result_folder
for file in $(ls );
do
    cat $file | sacremoses -l de -q detokenize > ../result_folder_detok/$file.detok &
done

sleep 2m
