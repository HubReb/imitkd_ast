# knn_ast_KD_nmt

## general information

Implementation of [Imitation-based Knowledge Distillation](https://github.com/asappresearch/imitkd) from the paper ["Autoregressive Knowledge Distillation through Imitation Learning"](https://arxiv.org/abs/2009.07253) for Automatic Speech Translation (AST).
Instead of an AST expert, The expert model is a trained Neural Machine Translation (NMT) model.

The implementation is entirely based the [fairseq framework](https://github.com/facebookresearch/fairseq), specifically on the [speech-to-text module](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text).
For usage of the fairseq framework please see the (fairseq documentation)[https://fairseq.readthedocs.io/en/latest/].


In order for ImitKD to work, several changes were made to the fairseq framework:
* training loop was changed
* several files were added to the criterions
    * [imitKD.py](fairseq/fairseq/criterions/imitKD.py) (ImitKD-optimal)
    * [imitKD_CE.py](fairseq/fairseq/criterions/imitKD_CE.py) (ImitKD-full)
    * [imitKD_ast.py](fairseq/fairseq/criterions/imitKD_ast.py) (ImitKD-full for AST expert and AST student)
    * [imitKD_ast_pure_kd.py](fairseq/fairseq/criterions/imitKD_ast_pure_kd.py) (word-level KD for AST expert and AST student)
    * [kd_expert_copy.py](fairseq/fairseq/criterions/kd_expert_copy.py) (word-level KD for NMT expert and AST student)
    * [imitKD_pipeline_nmt_training.py](fairseq/fairseq/criterions/imitKD_pipeline_nmt_training.py) (ImitKD-full training for NMT component in ASR-NMT cascade)
* other criterions were added as proof-of-word but are not recommended for usage


The best way to run experiments with generated transcripts is to:
1. use the ASR model to transcribe the speech data
2. use the NMT expert model to translate those transcripts if you want to use generated target translations
3. run `create_wmt19_generated_dataset.py` to create a new dataset of generated trancripts:
    ``python create_wmt19_generated_dataset.py -o ${fairseq-generate log file of NMT expert's translations} -a ${fairseq-generate log file of ASR model's transcripts} -d ${AST dataset file}``
4. use the new dataset just as the original datasets 





## data pre-processing

Data pre-processing was done as explained in fairseq speech-to-text module. Note that if you create your own datasets and want to use a NMT expert, you need to process the target transcripts and translations in the speech translation/recognition dataset the same way you processed the data for the NMT task.

Scripts to extract source transcripts and target translations from the csv Datafiles created by the speech-to-text pre-processing are included.


For COVOST2 and MUST-C:
1.  adapt the file paths in [get_src_to_st.py](fairseq/get_src_to_st.py) to fit your setup and simply run `python get_src_to_st.py`.
2. adapt the file paths in [get_source_text.py](fairseq/examples/speech_to_text/get_source_text.py) to your setup and run `python get_source_text.py`
3. the extracted data files are saved in `${dataset_name}/${split_name}`
4. process the extracted text data the same you did for your NMT expert, e.g. by adapting [prepare-rest.sh](fairseq/examples/speech_to_text/prepare-rest.sh)
5. run `python get_source_text.py` again
6. adapt the configuration files to point to your NMT expert's vocabulary and BPE.

## model training

Model training is done as is specified in the fairseq framework.
For instance, to train a small AST transformer model with `imit_kd` and a NMT expert run:


`fairseq-train ${COVOST_ROOT}/en --config-yaml config_st_en_de.yaml --train-subset train_processed --valid-subset dev_processed  --num-workers 8 --max-tokens 50000  --max-update 30000   --task speech_to_text --criterion imit_kd --report-accuracy --arch s2t_transformer_s  
--optimizer adam --lr 0.002 --lr-scheduler inverse_sqrt --seed 1 --clip-norm 10.0 --expert ${PATH_TO_EXPERT_MODEL} --expert-vocab-tgt ${PATH_TO_EXPERT_MODEL_DICTIONARY}  --expert-vocab-src ${PATH_TO_EXPERT_MODEL_SRC_DICTIONARY} --path  ${PATH_TO_EXPERT_MODEL_DICTIONARY} 
 --save-dir ${ST_SAVE_DIR}  --bpe-codes ${PATH_TO_BPE} --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8  --patience 10 --load-pretrained-encoder-from ${ASR_MODEL} --encoder-freezing-updates 1000`
 
 
__**Important**: Training such model requires at least 40 GB of RAM and a GPU with at least 20 GB of VRAM, 48GB is better suited.__
