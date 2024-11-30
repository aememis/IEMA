import pickle
import random

import config as cfg
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import sounddevice


class CorpusGauss:
    def __init__(self, sd) -> None:
        self.sd = sd
        self.sd.logger.info("Initializing CorpusGauss...")
        self.list_samples = None
        self.df_samples = None
        self.df_features = None
        self.length_seconds = 2
        self.length_sample = self.length_seconds * cfg.SAMPLE_RATE

    def generate_gauss_corpus(self):
        """Generate a Gaussian noise corpus."""
        self.sd.logger.info("Generating gaussian samples...")
        # generate samples with gaussian distribution
        # self.list_samples = np.random.normal(
        #     size=(cfg.CORPUS_SIZE, self.length_sample)
        # )

        # generate samples randomly, not gaussian
        self.list_samples = np.random.rand(cfg.CORPUS_SIZE, self.length_sample)
        self.list_samples = np.float32(self.list_samples)
        self.sd.logger.info(
            f"Created samples of shape: {self.list_samples.shape}"
        )

    # def extract_features(self):
    #     self.sd.logger.info("Extracting features...")
    #     list_notes = []
    #     list_audio = []
    #     for i in range(len(self.list_samples)):
    #         print(f"Processing {str(i).zfill(5)}/{cfg.CORPUS_SIZE}", end="\r")

    #         y = self.list_samples[i]
    #         sr = cfg.SAMPLE_RATE
    #         len_y = len(y)
    #         y /= np.max(np.abs(y))

    #         rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    #         spectral_bandwidth = librosa.feature.spectral_bandwidth(
    #             y=y, sr=sr, win_length=1024, hop_length=512
    #         )[0]
    #         flux = librosa.onset.onset_strength(y=y, sr=sr)
    #         mfcc = librosa.feature.mfcc(
    #             y=y, n_mfcc=1, win_length=1024, hop_length=512
    #         )[0]
    #         sc = librosa.feature.spectral_centroid(
    #             y=y, win_length=1024, hop_length=512
    #         )[0]
    #         sf = librosa.feature.spectral_flatness(
    #             y=y, win_length=1024, hop_length=512
    #         )[0]

    #         notes = [
    #             sr,
    #             len_y,
    #             np.mean(rms),
    #             np.mean(spectral_bandwidth),
    #             np.mean(flux),
    #             np.mean(mfcc),
    #             np.mean(sc),
    #             np.mean(sf),
    #         ]
    #         audio = [
    #             y,
    #         ]

    #         list_notes.append(notes)
    #         list_audio.append(audio)
    #     self.sd.logger.info(f"Processed {len(list_notes)} arrays.")

    #     ###
    #     ###
    #     ###
    #     ### random features gaussian
    #     list_notes = np.random.normal(size=(cfg.CORPUS_SIZE, 6))
    #     list_notes = np.float32(list_notes)
    #     # insert first two columns as sr and length
    #     # len_y
    #     list_notes = np.insert(list_notes, 0, 44100, axis=1)
    #     # sr
    #     list_notes = np.insert(list_notes, 0, 22050, axis=1)
    #     print(list_notes.shape)
    #     input(list_notes)
    #     ###
    #     ###
    #     ###

    #     self.df_features = pd.DataFrame(
    #         list_notes,
    #         columns=[
    #             "sr",
    #             "length",
    #             "rms",
    #             "spectral_bandwidth",
    #             "flux",
    #             "mfcc",
    #             "sc",
    #             "sf",
    #         ],
    #     ).reset_index(drop=True)

    #     self.df_samples = pd.DataFrame(
    #         list_audio,
    #         columns=[
    #             "signal",
    #         ],
    #     ).reset_index(drop=True)

    def normalize_features(self):
        self.sd.logger.info("Normalizing features...")
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

    def save_features_data(self):
        self.sd.logger.info("Saving corpus...")
        with open("gaussian_features_raw.pkl", "wb") as file:
            pickle.dump(self.df_features, file)
        with open("gaussian_features_norm.pkl", "wb") as file:
            pickle.dump(self.df_features_norm, file)
        with open("gaussian_samples.pkl", "wb") as file:
            pickle.dump(self.df_samples, file)
        self.sd.logger.info("Corpus saved.")

    def read_corpus_from_saved(self):
        self.sd.logger.info("Reading corpus from saved files...")
        with open("gaussian_features_norm.pkl", "rb") as f:
            self.df_features_norm = pickle.load(f)
        with open("gaussian_samples.pkl", "rb") as f:
            self.df_samples = pickle.load(f)
        self.sd.logger.info(
            f"Corpus read with shape {self.df_features_norm.shape}."
        )

    def describe_features_data(self):
        self.sd.logger.info("Describing corpus...")
        self.sd.logger.info(self.df_features_norm.describe())
        self.df_features_norm.iloc[:, 2:].boxplot()
        input(self.df_features_norm)
        plt.show()

    # def get_random_features(self):
    #     self.sd.logger.info("Generating random features...")

    #     ###
    #     ###
    #     ###
    #     ### random features gaussian
    #     list_notes = np.random.normal(size=(cfg.CORPUS_SIZE, 6))
    #     list_notes = np.float32(list_notes)
    #     # insert first two columns as sr and length
    #     # len_y
    #     list_notes = np.insert(list_notes, 0, 44100, axis=1)
    #     # sr
    #     list_notes = np.insert(list_notes, 0, 22050, axis=1)
    #     print(list_notes.shape)
    #     input(list_notes)
    #     ###
    #     ###
    #     ###

    #     self.df_features = pd.DataFrame(
    #         list_notes,
    #         columns=[
    #             "sr",
    #             "length",
    #             "rms",
    #             "spectral_bandwidth",
    #             "flux",
    #             "mfcc",
    #             "sc",
    #             "sf",
    #         ],
    #     ).reset_index(drop=True)

    #     self.df_samples = None

    # def prepare(self, from_saved=True):
    #     if from_saved:
    #         self.read_corpus_from_saved()
    #     else:
    #         self.get_random_features()
    #         self.normalize_features()
    #         self.save_features_data()
    #         self.describe_features_data()

    def generate_gauss_features(self):
        """Generate a Gaussian noise features."""
        self.list_samples = np.random.normal(size=(cfg.CORPUS_SIZE, 6))
        self.list_samples = np.random.rand(cfg.CORPUS_SIZE, 6)
        self.list_samples = np.float32(self.list_samples)
        self.df_features = pd.DataFrame(
            self.list_samples,
            columns=[
                "rms",
                "spectral_bandwidth",
                "flux",
                "mfcc",
                "sc",
                "sf",
            ],
        )
        self.sd.logger.info(
            "Created gaussian noise features of shape: "
            f"{self.df_features.shape}"
        )

    def prepare_gauss_features(self, from_saved=True):
        if from_saved:
            self.read_corpus_from_saved()
        else:
            self.generate_gauss_features()
            self.normalize_features()
            self.save_features_data()
            # self.describe_features_data()

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
        df_population = df_corpus.sample(
            population_size,
        ).reset_index(drop=True)
        return df_corpus, df_population
