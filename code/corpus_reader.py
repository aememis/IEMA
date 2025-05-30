import glob
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
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
        len_paths = len(list_paths)
        dict_audio = {}
        dict_notes = {}
        list_filenames = []
        skipped_count = 0

        def process_file(i, file):
            try:
                print(f"Loading {str(i).zfill(5)}/{len_paths}", end="\r")

                y, sr = librosa.load(file)
                len_y = len(y)
                if (
                    len_y < cfg.SAMPLES_THRESHOLD_LOW
                    or len_y > cfg.SAMPLES_THRESHOLD_HIGH
                ):
                    return
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
                dict_notes[filename] = notes
                list_filenames.append(filename)
            except Exception as e:
                self.sd.logger.warning(f"\nSkipping {file}, error loading: {e}")
                return

        with ThreadPoolExecutor(max_workers=16) as executor:
            list(
                executor.map(
                    lambda args: process_file(*args),
                    enumerate(list_paths, 1),
                )
            )

        self.sd.logger.info(f"Loaded {len(dict_notes)} files")

        list_notes = [dict_notes[filename] for filename in list_filenames]

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
        self.df_samples = None
        self.df_filenames = pd.DataFrame(
            list_filenames,
            columns=[
                "filename",
            ],
        ).reset_index(drop=True)
        self.df_filenames["sample_id"] = self.df_filenames.index

        self.sd.logger.info("Compiled samples")

    def read_corpus_from_saved(self):
        with open("precomputed/features.pkl", "rb") as f:
            self.df_features_norm = pickle.load(f)
            self.sd.logger.info("LEN DF")
            self.sd.logger.info(len(self.df_features_norm))
        with open("precomputed/samples.pkl", "rb") as f:
            # currently not supported as it's not needed for the research
            self.df_samples = None  # pickle.load(f)
        with open("precomputed/filenames.pkl", "rb") as f:
            self.df_filenames = pickle.load(f)

        df_metacoll = pd.read_csv(cfg.FSD50K_METADATA_COLLECTION_PATH)
        df_metacoll_valid = df_metacoll[
            df_metacoll["mids"].notnull() & (df_metacoll["mids"] != "")
        ]
        valid_filenames = df_metacoll_valid["fname"].astype(str).to_list()
        self.df_filenames = self.df_filenames[
            self.df_filenames["filename"]
            .apply(lambda x: os.path.splitext(x)[0])
            .astype(str)
            .isin(valid_filenames)
        ]
        self.df_features_norm = self.df_features_norm[
            self.df_features_norm.index.isin(self.df_filenames.index)
        ]
        self.df_filenames = self.df_filenames.reset_index(drop=True)
        self.df_features_norm = self.df_features_norm.reset_index(drop=True)
        self.sd.logger.info(
            f"Dropped {len(df_metacoll) - len(df_metacoll_valid)} invalid files"
        )

    def save_corpus(self):
        self.sd.logger.info("Saving corpus...")
        with open("precomputed/features_raw.pkl", "wb") as file:
            pickle.dump(self.df_features, file)
        with open("precomputed/features.pkl", "wb") as file:
            pickle.dump(self.df_features_norm, file)
        # currently not supported as it's not needed for the research
        # with open("precomputed/samples.pkl", "wb") as file:
        #     pickle.dump(self.df_samples, file)
        with open("precomputed/filenames.pkl", "wb") as file:
            pickle.dump(self.df_filenames, file)
        self.sd.logger.info(f"Corpus saved with size: {len(self.df_features)}")

    def normalize_corpus(self):
        self.sd.logger.info("Normalizing corpus...")
        self.df_features_norm = self.df_features.copy(deep=True)
        for col in [
            "rms",
            "spectral_bandwidth",
            "flux",
            "mfcc",
            "sc",
            "sf",
        ]:
            self.sd.logger.info(f"Normalizing {col}")
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
