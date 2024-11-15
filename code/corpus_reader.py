import glob
import os
import pickle
import random
from typing import Literal

import config as cfg
import librosa
import numpy as np
import pandas as pd


class CorpusReader:
    def __init__(self, sd):
        self.sd = sd
        self.df_corpus = None
        self.df_features = None
        self.df_features_norm = None
        self.df_filenames = pd.DataFrame()
        self.window_length = 1024
        self.hop_length = 512

    def read_corpus_from_files(self):
        self.sd.logger.info("Loading the dataset...")
        list_paths = glob.glob(cfg.DATASET_PATH_FSD50K)
        list_paths = random.sample(list_paths, cfg.CORPUS_SIZE)  ####
        len_paths = len(list_paths)
        list_audio = []
        list_notes = []
        list_filenames = []
        skipped_count = 0
        for i, file in enumerate(list_paths):
            try:
                print(f"Loading {str(i).zfill(5)}/{len_paths}", end="\r")

                y, sr = librosa.load(file)
                len_y = len(y)
                if (
                    len_y < cfg.SAMPLES_THRESHOLD_LOW
                    or len_y > cfg.SAMPLES_THRESHOLD_HIGH
                ):
                    continue
                y /= np.max(np.abs(y))

                rms = librosa.feature.rms(
                    y=y, frame_length=1024, hop_length=512
                )[0]
                spectral_bandwidth = librosa.feature.spectral_bandwidth(
                    y=y, sr=sr, win_length=1024, hop_length=512
                )[0]
                flux = librosa.onset.onset_strength(y=y, sr=sr)
                mfcc = librosa.feature.mfcc(
                    y=y, n_mfcc=1, win_length=1024, hop_length=512
                )[0]
                sc = librosa.feature.spectral_centroid(
                    y=y, win_length=1024, hop_length=512
                )[0]
                sf = librosa.feature.spectral_flatness(
                    y=y, win_length=1024, hop_length=512
                )[0]

                notes = [
                    # os.path.basename(file),
                    # sr,
                    # len_y,
                    np.mean(rms),
                    np.mean(spectral_bandwidth),
                    np.mean(flux),
                    np.mean(mfcc),
                    np.mean(sc),
                    np.mean(sf),
                ]
                audio = [
                    y,
                ]
                filename = os.path.basename(file)
                list_notes.append(notes)
                list_audio.append(audio)
                list_filenames.append(filename)
            except Exception as e:
                self.sd.logger.warning(f"\nSkipping {file}, error loading: {e}")
                skipped_count += 1
                continue
        self.sd.logger.info(
            f"Loaded {len(list_audio)} files, skipped {skipped_count}."
        )

        self.df_features = pd.DataFrame(
            list_notes,
            columns=[
                "rms",
                "spectral_bandwidth",
                "flux",
                "mfcc",
                "sc",
                "sf",
            ],
        ).reset_index(drop=True)
        self.df_samples = pd.DataFrame(
            list_audio,
            columns=[
                "signal",
            ],
        ).reset_index(drop=True)
        self.df_filenames = pd.DataFrame(
            list_filenames,
            columns=[
                "filename",
            ],
        ).reset_index(drop=True)
        self.df_filenames["sample_id"] = self.df_filenames.index

        print(self.df_features.describe())
        print(self.df_samples.describe())
        print(self.df_filenames.describe())

    def read_corpus_from_saved(self):
        with open("features.pkl", "rb") as f:
            self.df_features_norm = pickle.load(f)
        print(self.df_features_norm.describe())
        input()
        with open("samples.pkl", "rb") as f:
            self.df_samples = (
                None  # pickle.load(f)  # temp, samples file is corrupted
            )
        with open("filenames.pkl", "rb") as f:
            self.df_filenames = pickle.load(f)
            print(self.df_filenames.describe())
            input()

    def save_corpus(self):
        self.sd.logger.info("Saving corpus...")
        with open("features_raw.pkl", "wb") as file:
            pickle.dump(self.df_features, file)
        with open("features.pkl", "wb") as file:
            pickle.dump(self.df_features_norm, file)
        with open("samples.pkl", "wb") as file:
            pickle.dump(self.df_samples, file)
        with open("filenames.pkl", "wb") as file:
            pickle.dump(self.df_filenames, file)
        self.sd.logger.info(f"Corpus saved with size: {len(self.df_features)}")

    def normalize_corpus(self):
        self.df_features_norm = self.df_features.copy(deep=True)
        for col in [
            "rms",
            "spectral_bandwidth",
            "flux",
            "mfcc",
            "sc",
            "sf",
        ]:
            min_val = self.df_features[col].min()
            max_val = self.df_features[col].max()
            self.df_features_norm[col] = (self.df_features[col] - min_val) / (
                max_val - min_val
            )

    def prepare(self, source: Literal["files", "checkpoint"], save=False):
        if source == "checkpoint":
            self.read_corpus_from_saved()
        elif source == "files":
            self.read_corpus_from_files()
            self.normalize_corpus()
        else:
            raise Exception(f"{source} is not excepted as a source")
        if save:
            self.save_corpus()

    def get_samples(self):
        return self.df_samples

    def get_as_population(self, population_size=cfg.POPULATION_SIZE):
        # keep: ["rms","spectral_bandwidth","flux","mfcc","sc","sf",]
        df_corpus = (
            self.df_features_norm[
                [
                    "rms",
                    "spectral_bandwidth",
                    "flux",
                    "mfcc",
                    "sc",
                    "sf",
                ]
            ]
            .assign(score=-1)
            .assign(pop=None)
            .assign(sample_id=None)
            .assign(id=None)
            .loc[:, cfg.DATA_FIELDS_CORPUS_LABEL]
            .set_axis(cfg.DATA_FIELDS_CORPUS, axis=1)
        )
        df_corpus.sample_id = self.df_filenames.sample_id

        df_population = df_corpus.sample(population_size).reset_index(drop=True)
        return df_corpus, df_population
