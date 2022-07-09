# knn_ast_KD_nmt



## General information

Implementation of [Imitation-based Knowledge Distillation](https://github.com/asappresearch/imitkd) from the paper ["Autoregressive Knowledge Distillation through Imitation Learning"](https://arxiv.org/abs/2009.07253) for Automatic Speech Translation (AST).
Instead of an AST expert, The expert model is a trained Neural Machine Translation (NMT) model.

The implementation is entirely based the [fairseq framework](https://github.com/facebookresearch/fairseq), specifically on the [speech-to-text module](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text).
For usage of the fairseq framework please see the [fairseq documentation](https://fairseq.readthedocs.io/en/latest/).


In order for ImitKD to work, several changes were made to the fairseq framework:
* training loop was changed
* [dataset class](fairseq/fairseq/data/audio/speech_to_text_dataset.py) for speech-to-text was changed 
* several files were added to the criterions
    * [imitKD.py](fairseq/fairseq/criterions/imitKD.py) (ImitKD-optimal)
    * [imitKD_CE.py](fairseq/fairseq/criterions/imitKD_CE.py) (ImitKD-full)
    * [imitKD_ast.py](fairseq/fairseq/criterions/imitKD_ast.py) (ImitKD-full for AST expert and AST student)
    * [imitKD_ast_pure_kd.py](fairseq/fairseq/criterions/imitKD_ast_pure_kd.py) (word-level KD for AST expert and AST student)
    * [kd_expert_copy.py](fairseq/fairseq/criterions/kd_expert_copy.py) (word-level KD for NMT expert and AST student)
    * [imitKD_pipeline_nmt_training.py](fairseq/fairseq/criterions/imitKD_pipeline_nmt_training.py) (ImitKD-full training for NMT comonent in ASR-NMT cascade)
* other criterions were added as proof-of-word but are not recommended for usage


## Setup

The repo contains an `environment.yml` that specifies the required dependencies and python-version.
Simply create a new conda environment from the environment.yml by running `conda env create -f environment.yml`.
Then change into the fairseq directory and install fairseq:

```
cd fairseq
pip install .
```


If you want to develop locally without reinstall fairseq after every change run:
```
pip install --editable .
```

Installing fairseq is required to have access to the changes made to the framework in this repo.


## Datasets

| Dataset                                              | hours of speech    | total number of samples |
| :--------------------------------------------------- | -----------------: | ----------------------: |
| MUST-C (en-de)                                       | 408                | 234K                    |
| COVOST2 (en-de)                                      | 430                | 319K                    |
| Europarl-ST (en-de)                                  | 83                 | 35.5                    |
| Europarl-ST (en-de)                                  | 173                | 72.4K                   |




## Data pre-processing

Data pre-processing was done as explained in fairseq speech-to-text module. Note that if you create your own datasets and want to use a NMT expert, you need to process the target transcripts and translations in the speech translation/recognition dataset the same way you processed the data for the NMT task.

Scripts to extract source transcripts and target translations from the csv Datafiles created by the speech-to-text pre-processing are included.


For COVOST2 and MUST-C:
1.  adapt the file paths in [get_src_to_st.py](fairseq/get_src_to_st.py) to fit your setup and simply run `python get_src_to_st.py`.
2. adapt the file paths in [get_source_text.py](fairseq/examples/speech_to_text/get_source_text.py) to your setup and run `python get_source_text.py`
3. the extracted data files are saved in `${dataset_name}/${split_name}`
4. process the extracted text data the same you did for your NMT expert, e.g. by adapting [prepare-rest.sh](fairseq/examples/speech_to_text/prepare-rest.sh)
5. run `python get_source_text.py` again
6. adapt the configuration files to point to your NMT expert's vocabulary and BPE.

## Model training and evaluation

Model training and evaluation is done as is specified in the fairseq framework.
For instance, to train a small AST transformer model with `imit_kd` and a NMT expert run:

```
fairseq-train ${COVOST_ROOT}/en --config-yaml config_st_en_de.yaml --train-subset train_processed --valid-subset dev_processed  --num-workers 8 --max-tokens 50000  --max-update 30000   --task speech_to_text --criterion imit_kd --report-accuracy --arch s2t_transformer_s  \
--optimizer adam --lr 0.002 --lr-scheduler inverse_sqrt --seed 1 --clip-norm 10.0 --expert ${PATH_TO_EXPERT_MODEL} --expert-vocab-tgt ${PATH_TO_EXPERT_MODEL_DICTIONARY}  --expert-vocab-src ${PATH_TO_EXPERT_MODEL_SRC_DICTIONARY} --path  ${PATH_TO_EXPERT_MODEL_DIRECTORY} \
 --save-dir ${ST_SAVE_DIR}  --bpe-codes ${PATH_TO_BPE} --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8  --patience 10 --load-pretrained-encoder-from ${ASR_MODEL} --encoder-freezing-updates 1000`
 ```
 
__**Important**: Training such a model requires at least 40 GB of RAM and a GPU with at least 20 GB of VRAM, 48GB is better suited.__


## A note on replacing gold transcripts with generated transcripts



The best way to run experiments with generated transcripts is to:
1. use the ASR model to transcribe the speech data
2. use the NMT expert model to translate those transcripts if you want to use generated target translations
3. run `create_wmt19_generated_dataset.py` to create a new dataset of generated trancripts:
```
    python create_wmt19_generated_dataset.py -o ${fairseq-generate log file of NMT expert's translations} -a ${fairseq-generate log file of ASR model's transcripts} -d ${AST dataset file}
```
4. use the new dataset just like the original datasets 



Alternatively, you may run the respective proof-of-work fairseq criterions that generate the transcripts during training. Note that this significantly increases the required VRAM and training time. Creating a synthetic dataset to sample from instead is *highly* recommended.



## Results

The following BLEU scores were reported for ImitKD training and the two baselines, word-level knowledge distillation (KD) and negative log-likelihood training (NLL) for ground truth targets.
We experimented with replacing the audio transcripts given in the dataset with transcripts generated by an Automatic Speech Recognition (ASR) model for both KD and ImitKDT.
These are given as KDT and ImitKD.T

### MUST-C

| MODEL: Training method                               | dev               | test    |
| :--------------------------------------------------- | ----------------: | ------: |
| AST RNN: NLL                                         | 14.6              | 14.1    |
| AST RNN: KD                                          | 17.1              | 17.2    |
| AST RNN: KDT                                         | 16.9              | 15.9    |
| AST RNN: ImitKD-full                                 | 15.7              | 14.9    |
| AST RNN: ImitKDT-full                                | 16.3              | 15.1    |
| AST transformer: NLL                                 | 19.5              | 19.4    |
| AST transformer: KD                                  | 22.2              | 22.3    |
| AST transformer: KDT                                 | 22.5              | 22.6    |
| AST transformer: ImitKD-full                         | 23.2              | 23.3    |
| AST transformer: ImitKDT-full                        | 23.5              | 23.5    |

### COVOST2

| MODEL: Training method                               | dev               | test    |
| :--------------------------------------------------- | ----------------: | ------: |
| AST RNN: NLL                                         | 13.6              | 10.0    |
| AST RNN: KD                                          | 14.6              | 11.1    |
| AST RNN: KDT                                         | 14.1              | 10.6    |
| AST RNN: ImitKD-full                                 | 13.1              | 10.1    |
| AST RNN: ImitKDT-full                                | 12.8              | 9.7     |
| AST transformer: NLL                                 | 18.4              | 14.6    |
| AST transformer: wold-level knowledge distillation   | 21.3              | 17.7    |
| AST transformer: ImitKD-full                         | 21.8              | 18.4    |
| AST transformer: ImitKDT-full                        | 21.8              | 18.5    |

### Europarl-ST: clean training set

| MODEL: Training method                               | dev               | test    |
| :--------------------------------------------------- | ----------------: | ------: |
| AST RNN: NLL                                         | 13.8              | 14.4    |
| AST RNN: KD                                          | 17.4              | 17.8    |
| AST RNN: KDT                                         | 17.5              | 18.0    |
| AST RNN: ImitKD-full                                 | 17.0              | 17.1    |
| AST RNN: ImitKDT-full                                | 17.4              | 17.5    |


### Europarl-ST: clean+noisy training set

| MODEL: Training method                               | dev               | test    |
| :--------------------------------------------------- | ----------------: | ------: |
| AST RNN: NLL                                         | 17.5              | 17.3    |
| AST RNN: KD                                          | 11.5              | 12.0    |
| AST RNN: KDT                                         | 18.3              | 18.2    |
| AST RNN: ImitKD-full                                 | 12.0              | 12.3    |
| AST RNN: ImitKDT-full                                | 16.7              | 16.6    |



## Conclusions

ImitKD for AST with a NMT expert results in more performant models than can be achieved for NLL training (except COVOST2 RNNs).
But AST transformers are required to efficiently utilize the expert and consequently outperform KD.

Moreover, we found the audio transcripts in the dataset can be replaced with transcripts generated by an ASR model if the target translations are not replaced: For samples with incorrect transcripts, the expert views the prefix that is determined by the target translations as an incorrect translation of the input transcript as it cannot determine that its input is incorrect:
The expert lattempts to continue the student hypothesis in a manner that turns the "incorrect" translation of the generated transcript into a correct one.
As the above tables show the models trained with generated transcript achieve comparable performance.


## References

Lin, A.,  Wohlwend, J.,  Chen, H., Lei, T. : [Autoregressive Knowledge Distillation through Imitation Learning](https://arxiv.org/abs/2009.07253)

Mattia A. Di Gangi, Roldano Cattoni, Luisa Bentivogli, Matteo Negri, and Marco Turchi. [MuST-C: a Multilingual Speech Translation Corpus](https://aclanthology.org/N19-1202/).

Changhan Wang, Anne Wu, and Juan Miguel Pino. [Covost 2: A massively multilingual speech-to-text translation corpus. CoRR, abs/2007.10310, 2020a.]( https://arxiv.org/abs/2007.10310)
Javier Iranzo-Sánchez, Joan Albert Silvestre-Cerdà, Javier Jorge, Nahuel Roselló, Adrià Giménez, Albert Sanchis, Jorge Civera, and Alfons Juan. [Europarl-st: A multilingual corpus for speech translation of parliamentary debates. In ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8229–8233, 2020. doi: 10.1109/ICASSP40776.2020.9054626](https://doi.org/10.48550/arXiv.1911.03167)

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. [fairseq: A fast, extensible toolkit for sequence modeling](https://aclanthology.org/N19-4009/)

Changhan Wang and Yun Tang and Xutai Ma and Anne Wu and Dmytro Okhonko and Juan Pino: [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://aclanthology.org/2020.aacl-demo.6.pdf)

Yuchen Liu, Hao Xiong, Zhongjun He, Jiajun Zhang, Hua Wu, Haifeng Wang, and Chengqing Zong: [End-to-end speech translation with knowledge distillation.](https://www.isca-speech.org/archive/pdfs/interspeech_2019/liu19d_interspeech.pdf)

