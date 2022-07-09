#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

fileen=mustc_processed/train_text_en.txt
tmp=mustc_processed_text

cat $fileen | \
  perl $NORM_PUNC $l | \
  perl $REM_NON_PRINT_CHAR | \
  perl $TOKENIZER -threads 16 -a -l en  >> ${tmp}/train.tok.en


for split in dev test
do
  fileen=mustc_processed/${split}_text_en.txt

  cat $fileen| \
    perl $TOKENIZER -threads 16 -a -l en >> ${tmp}/${split}.tok.en
done


filede=mustc_processed/train_text.txt
tmp=mustc_processed_text

cat $filede | \
  perl $NORM_PUNC $l | \
  perl $REM_NON_PRINT_CHAR | \
  perl $TOKENIZER -threads 16 -a -l de  >> ${tmp}/train.tok.de


for split in dev test
do
  filede=mustc_processed/${split}_text.txt

  cat $filede | \
    perl $TOKENIZER -threads 16 -a -l de  >> ${tmp}/${split}.tok.de
done

fileen=covost/train_text_en.txt
tmp=covost_processed_text

cat $fileen | \
  perl $NORM_PUNC $l | \
  perl $REM_NON_PRINT_CHAR | \
  perl $TOKENIZER -threads 16 -a -l en  >> ${tmp}/train.tok.en


for split in dev test
do
  fileen=covost/${split}_text_en.txt

  cat $fileen | \
    perl $TOKENIZER -threads 16 -a -l en >> ${tmp}/${split}.tok.en
done




filede=covost/train_text.txt
tmp=covost_processed_text

cat $filede | \
  perl $NORM_PUNC $l | \
  perl $REM_NON_PRINT_CHAR | \
  perl $TOKENIZER -threads 16 -a -l de  >> ${tmp}/train.tok.de


for split in dev test
do
  filede=covost/${split}_text.txt

  cat $filede | \
    perl $TOKENIZER -threads 16 -a -l de >> ${tmp}/${split}.tok.de
done





"""
filede=libri_processed/train_text.txt
tmp=libri_processed_text

cat $filede | \
  perl $NORM_PUNC $l | \
  perl $REM_NON_PRINT_CHAR | \
  perl $TOKENIZER -threads 16 -a -l de  >> ${tmp}/train.tok.de

"""
