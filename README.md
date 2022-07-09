# knn_ast_KD_nmt

## general information

Implementation of [Imitation-based Knowledge Distillation][https://github.com/asappresearch/imitkd] from the paper ["Autoregressive Knowledge Distillation through Imitation Learning"][https://arxiv.org/abs/2009.07253] for Automatic Speech Translation (AST).
Instead of an AST expert, The expert model is a trained Neural Machine Translation model.

The implementation is entirely based the [fairseq framework][https://github.com/facebookresearch/fairseq], specifically on the [speech-to-text module][https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text].
For usage of the fairseq framework please see the [fairseq documentation][https://fairseq.readthedocs.io/en/latest/].


In order for ImitKD to work, several changes were made to the fairseq framework:
* training loop was changed for $$\beta$$ calculation
* several files were added to the criterions
    * imitKD.py (ImitKD-optimal)
    * imitKD_CE.py (ImitKD-full)
    * imitKD_ast.py (ImitKD-full for AST expert and AST student)
    * imitKD_ast_pure_kd.py (word-level KD for AST expert and AST student)
    * kd_expert_copy.py (word-level KD for NMT expert and AST student)
    * imitKD_pipeline_nmt_training.py (ImitKD-full training for NMT component in ASR-NMT cascade)
* other criterions were added as proof-of-word but are not recommended for usage


The best way to run experiments with generated transcripts is to:
    1. use the ASR model to transcribe the speech data
    2. use the NMT expert model to translate those transcripts if you want to use generated target translations
    3. run `create_wmt19_generated_dataset.py` to create a new dataset of generated trancripts:
        ``python create_wmt19_generated_dataset.py -o ${fairseq-generate log file of NMT expert's translations} -a ${fairseq-generate log file of ASR model's transcripts} -d ${AST dataset file}``
    4. use the new dataset just as the original datasets 







[1] https://github.com/asappresearch/imitkd
[2] https://arxiv.org/abs/2009.07253
[3] https://github.com/facebookresearch/fairseq
[4] https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text
[5] https://fairseq.readthedocs.io/en/latest/
