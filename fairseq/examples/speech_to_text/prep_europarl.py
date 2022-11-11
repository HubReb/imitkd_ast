#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Tuple, Optional

import pandas as pd
import torchaudio
import torch

from data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from fairseq.data.audio.audio_utils import get_waveform
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

import soundfile as sf

log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


class EUROPARL(Dataset):
    """
    Create a Dataset for Europarl. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "test"]
    LANGUAGES = ["en", "de", "es", "fr", "it", "nl", "pl", "pt", "ro"]

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
        target_language: Optional[str] = None,
    ) -> None:
        assert split in self.SPLITS
        assert source_language in self.LANGUAGES
        self.asr_task = target_language is None
        _root = Path(root) / source_language
        wav_root = _root / "audios"
        print(_root, wav_root)
        assert _root.is_dir() and wav_root.is_dir()
        if self.asr_task:
            import warnings
            warnings.simplefilter("always")
            warnings.warn("WARNING: EUROPARL-ST has no single asr part - we use the en-de alignments to get en asr")
            txt_root = _root / "de"
            with open(txt_root / f"{split}/segments.tok.en") as f:
                utterances = [r.strip() for r in f.read().split("\n")[:-1]]
        else:
            assert target_language in self.LANGUAGES
            txt_root = _root / f"{target_language}/"
            with open(txt_root / f"{split}/segments.tok.{source_language}") as f:
                utterances = [r.strip() for r in f.read().split("\n")[:-1]]
            assert txt_root.is_dir()
            # Load target utterances
            with open(txt_root / f"{split}/segments.tok.{target_language}") as f:
                translations = [r.strip() for r in f.read().split("\n")[:-1]]
            assert len(translations) == len(utterances)
        segments = [{} for _ in utterances]
        for i, u in enumerate(utterances):
            segments[i][f"{source_language}"] = u
            segments[i][f"{target_language}"] = (
                None if self.asr_task else translations[i]
            )
        # audio files
        with open(txt_root / f"{split}/segments.lst") as f:
            audio_data = [line.split() for line in f.read().split("\n")[:-1]]
        # speakers
        with open(txt_root / f"{split}/speakers.lst") as f:
            speakers_list = [r.strip() for r in f.read().split("\n")[:-1]]
        # Gather info
        self.data = []
        speaker_id = 0
        old_audiofile = audio_data[0][0]
        for index, audio_datapoint in enumerate(audio_data):
            audio_filename, start_point, end_point = audio_datapoint
            if audio_filename != old_audiofile:
                speaker_id += 1
                old_audiofile = audio_filename
            duration = float(end_point) - float(start_point)
            audio_filename += ".wav"
            wav_path = wav_root / audio_filename
            # stimmt
            # torchaudio old version - uncomment for cluster usage
            # sample_rate = torchaudio.info(wav_path.as_posix())[0].rate
            # get info with soundfile - comment out on cluster
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            frame_offset = int(sample_rate * float(start_point))
            num_frames = int(duration * sample_rate)
            utterance = segments[index][f"{source_language}"]
            target_sentence = (
                None if self.asr_task else segments[index][f"{target_language}"]
            )
            _id = f"{wav_path.stem}_{index}"
            self.data.append(
                (
                    wav_path.as_posix(),
                    utterance,
                    target_sentence,
                    speakers_list[speaker_id],
                    _id,
                    frame_offset,
                    num_frames,
                )
            )

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str, str]:
        (
            wav_path,
            src_utt,
            tgt_utt,
            spk_id,
            utt_id,
            frame_offset,
            num_frames,
        ) = self.data[n]
        # waveform, sample_rate = torchaudio.load(
            # wav_path, offset=frame_offset, num_frames=num_frames
        # )
        # comment this out for cluster - adopted from newer version of fairseq to deal with changed torchaudio API
        waveform, sample_rate = get_waveform(wav_path, frames=num_frames, start=frame_offset)
        waveform = torch.from_numpy(waveform)
        if waveform.shape[0] == 2:
            if torch.sum(waveform[1]) == 0:  # empty second channel
                waveform = waveform[0, :].unsqueeze(0)
            elif torch.sum(waveform[1]) == 0:  # empty first channel
                waveform = waveform[0, :].unsqueeze(0)
            else:  # add both channels and divide by number of channels
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform, sample_rate, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")
        # Extract features
    feature_root = root / "fbank80"
    feature_root.mkdir(exist_ok=True)
    for split in EUROPARL.SPLITS:
        print(f"Fetching split {split}...")
        dataset = EUROPARL(
            root.as_posix(), split, args.src_lang, args.tgt_lang
        )
        print("Extracting log mel filter bank features...")
        for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
            extract_fbank_features(
                waveform, sample_rate, feature_root / f"{utt_id}.npy"
            )
    # Pack features into ZIP
    zip_path = root / "fbank80.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    task = f"st_{args.src_lang}_{args.tgt_lang}" if args.tgt_lang else f"asr_{args.src_lang}"
    for split in EUROPARL.SPLITS:
        is_train_split = split.startswith("train")
        if args.il:
            MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "src_text", "speaker"]
        else:
            MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = EUROPARL(
            args.data_root, split, args.src_lang, args.tgt_lang
        )
        if args.il:
            for wav, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
                manifest["id"].append(utt_id)
                manifest["audio"].append(zip_manifest[utt_id])
                duration_ms = int(wav.size(1) / sr * 1000)
                manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
                manifest["tgt_text"].append(tgt_utt)
                manifest["src_text"].append(src_utt)
                manifest["speaker"].append(speaker_id)
        else:
            for wav, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
                manifest["id"].append(utt_id)
                manifest["audio"].append(zip_manifest[utt_id])
                duration_ms = int(wav.size(1) / sr * 1000)
                manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
                manifest["tgt_text"].append(src_utt if args.tgt_lang is None else tgt_utt)
                manifest["speaker"].append(speaker_id)
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        if args.il:
            df = filter_manifest_df(df, is_train_split=is_train_split, is_il=True)
            save_df_to_tsv(df, root / f"{split}_{task}_with_source_text.tsv")
        else:
            df = filter_manifest_df(df, is_train_split=is_train_split)
            save_df_to_tsv(df, root / f"{split}_{task}.tsv")
    # Generate vocab
    v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    if args.il:
        spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{task}_with_source_text"
    else:
        spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{task}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
        )
    # Generate config YAML
    gen_config_yaml(
        root,
        spm_filename_prefix + ".model",
        yaml_filename=f"config_{task}.yaml",
        specaugment_policy="lb",
    )
    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    parser.add_argument("--tgt-lang", "-t", type=str)
    parser.add_argument("--il", "-il", default=False, action="store_true")
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
