import pickle

import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        self.list_samples = np.random.normal(
            size=(cfg.CORPUS_SIZE, self.length_sample)
        )
        self.list_samples = np.float32(self.list_samples)
        self.sd.logger.info(
            f"Created samples of shape: {self.list_samples.shape}"
        )

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
        with open("precomputed/gaussian_features_raw.pkl", "wb") as file:
            pickle.dump(self.df_features, file)
        with open("precomputed/gaussian_features_norm.pkl", "wb") as file:
            pickle.dump(self.df_features_norm, file)
        with open("precomputed/gaussian_samples.pkl", "wb") as file:
            pickle.dump(self.df_samples, file)
        self.sd.logger.info("Corpus saved.")

    def read_corpus_from_saved(self):
        self.sd.logger.info("Reading corpus from saved files...")
        with open("precomputed/gaussian_features_norm.pkl", "rb") as f:
            self.df_features_norm = pickle.load(f)
        with open("precomputed/gaussian_samples.pkl", "rb") as f:
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
