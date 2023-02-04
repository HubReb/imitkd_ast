# imitkd_ast



## General information

Implementation of [Imitation-based Knowledge Distillation](https://github.com/asappresearch/imitkd) from the paper ["Autoregressive Knowledge Distillation through Imitation Learning"](https://arxiv.org/abs/2009.07253) for Automatic Speech Translation (AST).
Instead of an AST expert,  a trained Neural Machine Translation (NMT) model is used as oracle.

The implementation is entirely based the [fairseq framework](https://github.com/facebookresearch/fairseq), specifically on the [speech-to-text module](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text).
For usage of the fairseq framework please see the [fairseq documentation](https://fairseq.readthedocs.io/en/latest/).
At the moment, this repository also contains code from the [Nearest Neighbor Machine Translation](https://github.com/bpnayak/knnmt), though it is not relevant to the results reported here.

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
Simply create a new conda environment from the environment.yml by running:
```bash
conda env create -f environment.yml
```
Note that this process may take several minutes.
Then activate the environment, change into the fairseq directory and install fairseq:

```bash
conda activate train
cd fairseq
pip install .
```


If you want to develop locally without reinstalling fairseq after every change run:
```bash
pip install --editable .
```

Installing fairseq like this is required to have access to the changes made to the framework in this repo.

Download and compile [fastBPE](https://github.com/glample/fastBPE). 

## Datasets

| Dataset                                         | hours of speech | total number of samples |
|:------------------------------------------------|----------------:|------------------------:|
| MUST-C (en-de)                                  |             408 |                    234K |
| COVOST2 (en-de)                                 |             430 |                    319K |
| Europarl-ST (en-de)                             |              83 |                    35.5 |
| Europarl-ST (en-de) + Europarl-ST (en-de) noisy |             173 |                   72.4K |




## Data pre-processing

Audo data pre-processing was done as explained in fairseq speech-to-text module. Note that if you create your own datasets and want to use a NMT expert, you need to process the target transcripts and translations in the speech translation/recognition dataset the same way you processed the data for the NMT task.

Scripts to extract source transcripts and target translations from the csv Datafiles created by the speech-to-text pre-processing are included.

The easiest way to create datasets with source transcripts and target translations is given in the speech-to-text modules [README](fairseq/examples/speech_to_text/README.md).


For COVOST2 and MUST-C:
1. Run [get_source_text.py](fairseq/examples/speech_to_text/get_source_text.py) `python get_source_text.py python -m ${MUSTC_DATA}/en-de/ -c ${COVOST_DATA}/en/`.
2. Process the extracted text data the same you did for your NMT expert, e.g. by adapting [prepare-rest.sh](fairseq/examples/speech_to_text/prepare-rest.sh).
3. Rerun `python get_source_text.py python -m ${MUSTC_DATA}/en-de/ -c ${COVOST_DATA}/en/`
4. (Optional) The source transcripts and translations are extracted and processed during the above step. To  generate the binarized files run 
```bash
fairseq-preprocess --source-lang en --target-lang de     --trainpref ${PROCESSED_DATA}/train --validpref ${PROCESSED_DATA}/dev --testpref ${PROCESSED_DATA}/test  --destdir ${BINARIZED_DATA_DIR}  --workers 21 --srcdict ${WMT19_TRANSFORMERS_DICTIONARY}  --joined-dictionary
```
5. (Optional) Generate the translations of the gold transcripts with the WMT19 transformer to use sequence-level knowledge distillation later on._NOTE_: This may take several minutes up to 2 hours, depending on your hardware.
```bash
fairseq-generate ${BINARIZED_DATA_DIR}\
  --gen-subset train  --path ${WMT19_TRANSFORMER_DIRECTORY}/model1.pt  --max-tokens 5000 --beam 5 --remove-bpe --sacrebleu  > ${OUTPUT_FILE}
```
6. Run `python get_source_text.py` again
7. (Optional) Run [create_wmt19_generated_dataset.py](create_wmt19_generated_dataset.py) to create the dataset consisting of the original transcripts to WMT19 translations of the transcripts:
```bash
python create_wmt19_generated_dataset.py -o ${OUTPUT_FILE} -d {PROCESSED_SPEECH_TO_TEXT_DATASET_FILE}
```
8. Adapt the configuration files (`config_{task}.yaml`) to point to your NMT expert's vocabulary and BPE. The configuration files are in `{MUSTC_DATA}/en-de/` and  `${COVOST_DATA}/en/`.
 

## Model training and evaluation

Model training and evaluation is done as is specified in the fairseq framework.
For instance, to train a small AST transformer model with `imit_kd` and a NMT expert run:

```bash
fairseq-train ${COVOST_ROOT}/en --config-yaml config_st_en_de.yaml --train-subset train_processed --valid-subset dev_processed  --num-workers 8 --max-tokens 50000  --max-update 30000   --task speech_to_text --criterion imit_kd --report-accuracy --arch s2t_transformer_s  \
--optimizer adam --lr 0.002 --lr-scheduler inverse_sqrt --seed 1 --clip-norm 10.0 --expert ${PATH_TO_EXPERT_MODEL} --expert-vocab-tgt ${PATH_TO_EXPERT_MODEL_DICTIONARY}  --expert-vocab-src ${PATH_TO_EXPERT_MODEL_SRC_DICTIONARY} --path  ${PATH_TO_EXPERT_MODEL_DIRECTORY} \
 --save-dir ${ST_SAVE_DIR}  --bpe-codes ${PATH_TO_BPE} --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8  --patience 10 --load-pretrained-encoder-from ${ASR_MODEL} --encoder-freezing-updates 1000`
 ```
 
__**Important**: Training such a model requires at least 40 GB of RAM and a GPU with at least 20 GB of VRAM, 48GB of VRAM are recommended.__

### AggreVaTe 

To use AggreVaTe , e. g. the above transformer or a ImitKD transformer, use:

```bash
fairseq-train ${COVOST_ROOT} \
  --config-yaml config_test_wmt19.yaml --train-subset train_processed  --valid-subset dev_processed --finetune-from-model ${pre-trained_s2t_transformer} \
  --save-dir ${ST_SAVE_DIR} --num-workers 8 --max-tokens 50000  --max-epoch 50 --update-freq 8\
  --task speech_to_text --criterion  aggrevate --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr 1e-6 --lr-scheduler fixed  \
 --expert ${EXPERT} --expert-vocab-tgt ${PATH_TO_EXPERT_MODEL_DICTIONARY} \
  --seed 1 --clip-norm 10.0  --expert-vocab-src ${PATH_TO_EXPERT_MODEL_SRC_DICTIONARY} \
 --expert-vocab-tgt ${PATH_TO_EXPERT_MODEL_SRC_DICTIONARY} --path  ${PATH_TO_EXPERT_MODEL_DIRECTORY} \
  --save-dir ${ST_SAVE_DIR} --bpe-codes ${PATH_TO_BPE} \
  --patience 10 --sample-action-prob 0.5  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --eval-bleu \
  --eval-bleu-detok moses   \
 --tensorboard-logdir ${LOG_DIR} --log-file ${LOG_FILE}
```

In the above command, training is warm-started by loading a pre-trained model (`--finetune-from-model`). 
Remove the flag to cold-start training.
If cold-starting, it is strongly recommended to initialze the new model with a pre-trained ASR encoder with the `--load-pretrained-encoder-from` flag.
Because the expert policy is not mixed in to generate the student's translations, training in this manner is not efficient.

`--sample-action-prob` defines the probability with which a random uniform action is taken. Otherwise, the model's most probable action is taken.


uniform random action is taken with probability 0.5. 
Otherwise, the model's best action at timestep t is taken.
The best model with respect to BLEU on the development set is taken.
Omit ` --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --eval-bleu --eval-bleu-detok moses` to choose the best model with respect to NLL on the validation set.


## A note on replacing gold transcripts with generated transcripts



The best way to run experiments with generated transcripts is to:
1. use the ASR model to transcribe the speech data as demonstrated in the speech-to-text module examples
2. extract the generated hypotheses with
```bash
python create_wmt19_generated_dataset.py -d ${speech-to-text dataset} -a ${LOGGED_ASR_MODEL_OUTPUT_ON_DATASET}
```
3. use the NMT expert model to translate those transcripts if you want to use generated target translations
```bash
${FASTBPE}/fast applybpe ${DATADIR}/output.en ${DATADIR}/${EXTRACTED_TRANSCRIPTS} ${BPECODES} ${WMT19_VOCAB}

# binarize data - easiest to simply use the already extracted references here
cp ${PROCESSED_DATA}/train.de ${DATADIR}/output.de
fairseq-preprocess --source-lang en --target-lang de     --trainpref ${DATA_DIR}/train  --destdir ${BINARIZED_DATA_DIR}  --workers 21 --srcdict ${WMT19_TRANSFORMERS_DICTIONARY}  --joined-dictionary
fairseq-generate ${BINARIZED_DATA_DIR}  --gen-subset train --path ${WMT19_TRANSFORMER_DIRECTORY}/model1.pt  --batch-size 32 --beam 5 --remove-bpe --sacrebleu  > ${LOG_FILE_TRANSCRIPT_TRANSLATIONS}
```
4. run `create_wmt19_generated_dataset.py` to create a new dataset of generated trancripts:
```bash
    python create_wmt19_generated_dataset.py -o ${fairseq-generate log file of NMT expert's translations} -a ${fairseq-generate log file of ASR model's transcripts} -d ${AST dataset file}
```
5. use the new dataset just like the original datasets 



Alternatively, you may run the respective proof-of-work fairseq criterions that generate the transcripts during training. Note that this significantly increases the required VRAM and training time. Creating a synthetic dataset to sample from instead is *highly* recommended.



## Results

The following BLEU scores were reported for ImitKD training and the two baselines, word-level knowledge distillation (KD) and negative log-likelihood training (NLL) for ground truth targets.
We experimented with replacing the audio transcripts given in the dataset with transcripts generated by an Automatic Speech Recognition (ASR) model for both KD and ImitKDT.
These are given as KDT and ImitKD.T

### MUST-C

| MODEL: Training method        |  dev | test |
|:------------------------------|-----:|-----:|
| AST RNN: NLL                  | 14.6 | 14.1 |
| AST RNN: KD                   | 17.9 | 17.2 |
| AST RNN: KDT                  | 16.9 | 15.9 |
| AST RNN: ImitKD-full          | 15.7 | 14.9 |
| AST RNN: ImitKDT-full         | 16.3 | 15.1 |
| AST transformer: NLL          | 19.5 | 19.4 |
| AST transformer: KD           | 22.2 | 22.3 |
| AST transformer: KDT          | 22.5 | 22.6 |
| AST transformer: ImitKD-full  | 23.2 | 23.3 |
| AST transformer: ImitKDT-full | 23.5 | 23.5 |

### COVOST2

| MODEL: Training method        |  dev | test |
|:------------------------------|-----:|-----:|
| AST RNN: NLL                  | 13.6 | 10.0 |
| AST RNN: KD                   | 14.6 | 11.1 |
| AST RNN: KDT                  | 14.1 | 10.6 |
| AST RNN: ImitKD-full          | 13.1 | 10.1 |
| AST RNN: ImitKDT-full         | 12.8 |  9.7 |
| AST transformer: NLL          | 18.4 | 14.2 |
| AST transformer: KD           | 21.3 | 17.7 |
| AST transformer: KDT          | 21.7 | 18.0 |
| AST transformer: ImitKD-full  | 21.8 | 18.4 |
| AST transformer: ImitKDT-full | 21.8 | 18.5 |

### Europarl-ST: clean training set

| MODEL: Training method |  dev | test |
|:-----------------------|-----:|-----:|
| AST RNN: NLL           | 13.8 | 14.4 |
| AST RNN: KD            | 17.4 | 17.8 |
| AST RNN: KDT           | 17.5 | 18.0 |
| AST RNN: ImitKD-full   | 17.0 | 17.1 |
| AST RNN: ImitKDT-full  | 17.0 | 17.0 |


### Europarl-ST: clean+noisy training set

| MODEL: Training method |  dev | test |
|:-----------------------|-----:|-----:|
| AST RNN: NLL           | 17.5 | 17.3 |
| AST RNN: KD            | 11.5 | 12.0 |
| AST RNN: KDT           | 18.3 | 18.2 |
| AST RNN: ImitKD-full   | 12.0 | 12.3 |
| AST RNN: ImitKDT-full  | 16.6 | 16.6 |



### BLEU calculation

Due to the pre-processing applied to the text data, fairseq-generate calculates BLEU on the tokenized hypotheses and reference translations. 
While this is fine if only the models in this repository are compared, it does not provide a meaningful comparison to commonly reported detokenized BLEU.


The scripts to detokenize the translations are provided.
__Important__: fairseq requires an older sacrebleu version than fairseq. The easiest method is to create a second conda environment for the evaluation of results.
The configuration is given in eval_environment.yml.
Simply run:
```bash
conda env create -f environment.yml
```

Create a file that list the logs created by running fairseq-generate (*important*: Do not run fairseq-generate with the --quiet flag. If you do, fairseq-generates only saves the detokenized BLEU score to the log file).
Write each file name to a new line, e.g.:

```bash
baseline_mustc_nll.log
baseline_mustc_kd.log
```

Then run 

```bash
bash eval_script.sh ${name of file that contains the list of file names}
```

The detokenized BLEU scores can be found in the folder `result_scores`. 
The name of each file in this folder is `${file_name}_detokenized_bleu_score`.


## Value of synthetic transcripts

Synthetic transcripts can be used as a frozen dataset. Their real value, however, is the possibility to generate them "on the fly". For each iteration, the synthetic transcripts can be generated by an ASR model instead of only generating them once and using them as a frozen dataset. 

In each iteration, the ASR model transcibes the audio data *differently* - especially if the ASR model halluzinates - and thus the expert produces different optimal next tokens or full output distributions, depending on the chosen algorithm. This - essentially noise - makes it harder for the student model to overfit to the training data (e. g. when simply word-level KD is used) and leads to better results.
This has a negligable effect on the results achieved with the WMT19 expert's *vocabulary*.

But it allows the usage of a far smaller, _worse_ performing  expert with a far _smaller_ vocabulary (8K to 32K byte pair vocabulary size, thus easier and faster to train, but more likely to overfit and harder for such an expert to predict the next token if the synthetic transcript is entirely wrong) than the WMT19 expert and still achieve comparable results to models trained with the WMT19 expert, whereas using the original dataset results in far worse performing student models.

Thus, it allows a more effective KD than can be done with the original data.
Results for the CoVoST 2 dataset are listed below.


| MODEL: Training method                                |  dev | test |
|:------------------------------------------------------|-----:|-----:|
| on text data: CoVoST 2 NMT transformer: NLL (8K BPE)  |    - | 29.6 |
| on text data: WMT19 transformer: NLL (32K BPE)        | 40.2 | 38.5 |
| AST transformer: NLL (8K BPE)                         | 20.0 | 15.8 |
| AST transformer: KD (8K BPE)                          | 17.0 | 14.2 |
| AST transformer: KDT (8K BPE, frozen transcripts)     | 15.8 | 12.9 |
| AST transformer: KDT (8K BPE, on-the-fly transcripts) | 22.1 | 18.2 |


## A word on phasing out the reference translations

Instead of phasing out the references as done in ImitKD(T), it is also possible to interpolate the two losses by using a weighted sum for ImitKD(T):

$$
L = \beta \cdot L_{KD} + (1 - \beta) \cdot L_{ImitKD} 

$$

with
$$
\beta \rightarrow 0
$$
and using the Imitation-based loss only to correct the student hypotheses. 
This leads to better results, but requires more GPU VRAM.


## Conclusions

ImitKD for AST with a NMT expert results in more performant models than can be achieved for NLL training (except COVOST2 RNNs).
But AST transformers are required to efficiently utilize the expert and outperform KD.

Moreover, we found the audio transcripts in the dataset can be replaced with transcripts generated by an ASR model if the target translations are not replaced: For samples with incorrect transcripts, the expert views the prefix that is determined by the target translations as an incorrect translation of the input transcript as it cannot determine that its input is incorrect:
The expert lattempts to continue the student hypothesis in a manner that turns the "incorrect" translation of the generated transcript into a correct one.
As the above tables show, the models trained with generated transcript achieve comparable performance.


## References

Lin, A.,  Wohlwend, J.,  Chen, H., Lei, T. : [Autoregressive Knowledge Distillation through Imitation Learning](https://arxiv.org/abs/2009.07253)

Mattia A. Di Gangi, Roldano Cattoni, Luisa Bentivogli, Matteo Negri, and Marco Turchi. [MuST-C: a Multilingual Speech Translation Corpus](https://aclanthology.org/N19-1202/).

Changhan Wang, Anne Wu, and Juan Miguel Pino. [Covost 2: A massively multilingual speech-to-text translation corpus. CoRR, abs/2007.10310, 2020a.]( https://arxiv.org/abs/2007.10310)
Javier Iranzo-Sánchez, Joan Albert Silvestre-Cerdà, Javier Jorge, Nahuel Roselló, Adrià Giménez, Albert Sanchis, Jorge Civera, and Alfons Juan. [Europarl-st: A multilingual corpus for speech translation of parliamentary debates. In ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8229–8233, 2020. doi: 10.1109/ICASSP40776.2020.9054626](https://doi.org/10.48550/arXiv.1911.03167)

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. [fairseq: A fast, extensible toolkit for sequence modeling](https://aclanthology.org/N19-4009/)

Changhan Wang and Yun Tang and Xutai Ma and Anne Wu and Dmytro Okhonko and Juan Pino: [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://aclanthology.org/2020.aacl-demo.6.pdf)

Yuchen Liu, Hao Xiong, Zhongjun He, Jiajun Zhang, Hua Wu, Haifeng Wang, and Chengqing Zong: [End-to-end speech translation with knowledge distillation.](https://www.isca-speech.org/archive/pdfs/interspeech_2019/liu19d_interspeech.pdf)

Khandelwal, Urvashi and Fan, Angela and Jurafsky, Dan and Zettlemoyer, Luke and Lewis, Mike: [Nearest Neighbor Machine Translation](https://arxiv.org/pdf/2010.00710.pdf). In International Conference on Learning Representations (ICLR) - 2021

