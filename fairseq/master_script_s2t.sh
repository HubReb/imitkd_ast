
python master_script_s2t.py  --bytes-per-token 2 --model ../berard_st/checkpoint_best.pt --save-data  ../../../fm/fm/fm/MUSTC/en-de/  --num-shards 1 --dstore-mmap dstores/MUST_s2t/index_only --num-for-training 1000000 --code-size 64 --ncentroids 4096 --train-index dstores/MUST_s2t/index_only.4096.index.trained --save-job --merge-dstore-job --train-index-job  --config config_st.yaml --max-tokens 4096  > job_steps.sh 

python master_script_s2t.py  --bytes-per-token 2 --model ../berard_st/checkpoint_best.pt --save-data data-bin/MUST/  --num-shards 1 --dstore-mmap dstores/MUST_s2t/index_only --num-for-training 1000000 --code-size 64 --ncentroids 4096 --train-index dstores/MUST_s2t/index_only.4096.index.trained  --faiss-index dstores/MUST_s2t/index_only.4096.index --write-merged-index dstores/MUST_s2t/index_only.4096.index --corpus-identifiers med --add-keys-job --merge-index-job  --config config_st.yaml --max-tokens 4096 >> job_steps.sh

