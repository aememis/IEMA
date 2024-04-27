import json
import os
import datetime
import logging
import math
import pickle
import random
import time
from typing import Literal

import config as cfg
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from umap import UMAP


class SharedData:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.logger = self._create_logger(self.timestamp)
        self.outpur_dir = "output/" + self.timestamp
        if not os.path.exists(self.outpur_dir):
            os.makedirs(self.outpur_dir)

        note = input("Enter a note for this run: ")
        self.metadata = {
            "start_timestamp": self.timestamp,
            "sample_rate": cfg.SAMPLE_RATE,
            "number_of_iterations": cfg.NUMBER_OF_ITERATIONS,
            "number_of_paths": cfg.NUMBER_OF_PATHS,
            "k": cfg.K,
            "corpus_method": cfg.CORPUS_METHOD,
            "corpus_size": cfg.CORPUS_SIZE,
            "projection_method": cfg.PROJECTION_METHOD,
            "population_size": cfg.POPULATION_SIZE,
            "selection_threshold_divisor": cfg.SELECTION_THRESHOLD_DIVISOR,
            "mutation_rate": cfg.MUTATION_RATE,
            "mutation_scale": cfg.MUTATION_STRENGTH,
            "elitist_selection": cfg.ELITIST_SELECTION,
            "sample_legth_threshold": (
                cfg.SAMPLES_THRESHOLD_LOW,
                cfg.SAMPLES_THRESHOLD_HIGH,
            ),
            "note": note,
        }
        path_metadata_output = os.path.join(
            self.outpur_dir, f"{self.timestamp}_metadata.json"
        )
        with open(path_metadata_output, "w") as file:
            json.dump(self.metadata, file)
        self.logger.info(self.metadata)

        # data
        self.x = None
        self.y = None
        self.z = None
        self.norm_x = None
        self.norm_y = None
        self.list_pos = None
        self.current_zone = -1
        self.zones = None
        self.angle_center = -1
        self.distance_center = -1
        self.qom = -1
        self.top = -1
        self.dispersion = -1
        self.lhpos = [0, 0, 0]
        self.rhpos = [0, 0, 0]
        self.headpos = [0, 0, 0]
        self.backpos = [0, 0, 0]
        self.data_buffer = np.zeros((100, 3))
        self.df_population = None
        self.df_dead = None
        self.df_tsne = None
        self.knn_distances = None
        self.knn_indices = None
        self.index_lookup = {}
        self.closest_point_index = None
        self.index_closest_in_corpus = None
        self.sounds = None
        self.selected_params = None
        self.musical_params = None
        self.cluster_labels = None
        self.start_time = None
        self.current_population = 0
        self.df_analysis_population = pd.DataFrame()
        self.df_analysis_features = pd.DataFrame()
        self.df_analysis_samples = pd.DataFrame()

    def _create_logger(self, timestamp):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Set the logging level
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s  %(message)s")

        # Create a file handler
        log_filename = f"log_{timestamp}.log"
        file_handler = logging.FileHandler(
            os.path.join(timestamp, log_filename), encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Create a console (stream) handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger


class Operator:
    def __init__(self, sd):
        self.sd = sd

    def init_from_corpus(self, df_corpus, df_population, df_samples):
        self.sd.logger.info("Initiating from corpus...")
        self.sd.current_population += 1
        df_population["pop"] = self.sd.current_population
        self.sd.df_population = df_population.copy(deep=True)
        self.sd.df_corpus = df_corpus.copy(deep=True)
        # self.sd.df_samples = df_samples.copy(deep=True)
        self.sd.logger.info(f"Corpus had {len(self.sd.df_corpus.index)} samples")
        self.sd.logger.info(
            f"Initiated from corpus with {len(self.sd.df_population.index)} individuals"
        )

    def get_or_create_paths(self, source: Literal["file", "generate"] = "file"):
        """Random path generator. Returns a list of random points within the 3D
        projection space.
        """
        if source == "file":
            with open("paths.pkl", "rb") as file:
                paths = pickle.load(file)
        elif source == "generate":
            # dimension_x = cfg.MOCAP_WIDTH_PROJECTION
            # dimension_y = cfg.MOCAP_HEIGHT_PROJECTION
            # dimension_z = cfg.MOCAP_HEIGHT_Z_PROJECTION

            paths = []
            for _ in range(cfg.NUMBER_OF_PATHS):
                path = []
                for j in range(cfg.SELECTION_THRESHOLD_DIVISOR):
                    x = random.uniform(-1, 1)
                    y = random.uniform(-1, 1)
                    z = random.uniform(-1, 1)
                    path.append([x, y, z])
                paths.append(path)
            paths = np.array(paths)
            with open("paths.pkl", "wb") as file:
                pickle.dump(paths, file)
        else:
            raise ValueError("Invalid source argument.")

        # self.draw_path(path)

        return paths

    def normalize_projection_data(self, df_proj):
        proj_min_x, proj_min_y, proj_min_z = df_proj.min(axis=0)
        proj_max_x, proj_max_y, proj_max_z = df_proj.max(axis=0)

        df_proj_norm = pd.DataFrame()
        df_proj_norm[0] = (
            (df_proj.iloc[:, 0] - proj_min_x) / (proj_max_x - proj_min_x) * 2
        )
        df_proj_norm[1] = (
            (df_proj.iloc[:, 1] - proj_min_y) / (proj_max_y - proj_min_y) * 2
        )
        df_proj_norm[2] = (
            (df_proj.iloc[:, 2] - proj_min_z) / (proj_max_z - proj_min_z) * 2
        )
        df_proj_norm.iloc[:, 0] -= 1
        df_proj_norm.iloc[:, 1] -= 1
        df_proj_norm.iloc[:, 2] -= 1

        # plt.figure()
        # plt.scatter(df_proj_norm.iloc[:, 0], df_proj_norm.iloc[:, 1], s=1.5)
        # plt.savefig("norm_proj.png")

        return df_proj_norm

    def draw_projection(self, path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for i in range(len(path)):
            x, y, z = path[i]
            ax.scatter(x, y, z, c="r", marker="o")
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")
        # plt.show()

    def umap_project(self):
        self.sd.logger.info("UMAP projecting the population...")
        umap_model = UMAP(
            n_components=3,
            # random_state=42,  # Optional: for reproducible results
            # n_neighbors=15,  # Default is 15, controls the balance between local vs global structure
            min_dist=0.1,
            # * 2
            # * self.sd.current_population,  # Default is 0.1, controls how tightly UMAP is allowed to pack points together
        )  #### ????
        umap_data = umap_model.fit_transform(self.sd.df_population.iloc[:, :-2])
        self.sd.logger.info("Done UMAP projection.")
        # self.draw_projection(umap_data)

        return pd.DataFrame(umap_data)

    def tsne_project(self):
        tsne = TSNE(
            n_components=3,
            # random_state=42,
            # perplexity=10,
            # early_exaggeration=8,
        )
        tsne_data = tsne.fit_transform(self.sd.df_population.iloc[:, :-2])
        self.sd.logger.info("Done TSNE projection.")
        return pd.DataFrame(tsne_data)

    def fit_knn(self):
        self.sd.knn = NearestNeighbors(n_neighbors=cfg.K)
        self.sd.knn.fit(self.sd.df_tsne.to_numpy())

    def get_and_rate_selected(self, path):
        """Rate the selected individuals based on the path."""

        for point in path:
            user_point = np.array([point])
            distances, indices = self.sd.knn.kneighbors(user_point)
            list_closest_indices = indices[0]
            closest_index_proj = list_closest_indices[0]

            # rate the closest individual
            self.sd.df_population.loc[closest_index_proj, "score"] = 1

    def apply_selection(self, threshold):
        df_top = self.sd.df_population.nlargest(threshold, "score", keep="first")
        # self.sd.df_dead = pd.concat(
        #     [self.sd.df_dead, self.sd.df_population], ignore_index=True
        # ).reset_index(drop=True)
        # self.sd.df_population.drop(self.sd.df_population.index, inplace=True)
        return df_top.copy(deep=True).reset_index(drop=True)

    def _cross_parts(self, arr1, arr2):
        crossover_point = np.random.randint(1, len(arr1) - 1)
        offspring1 = np.concatenate((arr1[:crossover_point], arr2[crossover_point:]))
        offspring2 = np.concatenate((arr2[:crossover_point], arr1[crossover_point:]))
        return [offspring1, offspring2]

    def apply_crossover(self, df):
        nd_new_values = []
        for i in range(len(df.index)):
            for j in range(i + 1, len(df.index), 1):
                arr1 = df.iloc[i].values[:-2]
                arr2 = df.iloc[j].values[:-2]
                list_new_values = self._cross_parts(arr1, arr2)
                list_new_values_reshaped = np.append(
                    list_new_values, [[None, self.sd.current_population]] * 2, axis=1
                )
                nd_new_values.extend(list_new_values_reshaped)
        df_new_values = pd.DataFrame(nd_new_values, columns=cfg.DATA_FIELDS_CORPUS)
        # df = pd.concat([df, df_new_values], ignore_index=True)
        df_new_values["score"] = -1
        return df_new_values

    def apply_mutation(self, df):
        list_new_datapoints = []
        for i in range(len(df.index)):
            if np.random.random() < cfg.MUTATION_RATE:
                new_datapoint = df.iloc[i].copy(deep=True)
                for j in range(len(new_datapoint[:-2])):
                    if np.random.random() < cfg.MUTATION_RATE:
                        new_datapoint[j] += np.random.uniform(
                            -cfg.MUTATION_STRENGTH, cfg.MUTATION_STRENGTH
                        )
                list_new_datapoints.append(new_datapoint)
        if len(list_new_datapoints) > 0:
            df_new_datapoints = pd.DataFrame(
                list_new_datapoints, columns=cfg.DATA_FIELDS_CORPUS
            )
            df = pd.concat([df, df_new_datapoints])
        return df

    def apply_gaussian_mutation(self, df):
        list_new_datapoints = []
        for i in range(len(df.index)):
            new_datapoint = df.iloc[i].copy(deep=True)
            for j in range(len(new_datapoint[:-2])):
                if np.random.random() < cfg.MUTATION_RATE:
                    new_value = new_datapoint[j] + np.random.normal(
                        0, cfg.MUTATION_STRENGTH
                    )
                    new_datapoint[j] = np.clip(new_value, -1, 1)

            list_new_datapoints.append(new_datapoint)
        if list_new_datapoints:
            df_new_datapoints = pd.DataFrame(list_new_datapoints, columns=df.columns)
            df = pd.concat([df, df_new_datapoints])
        return df

    def draw_all(self, df_proj_norm, path):
        # draw projection and the selected individuals along with the path, in different color and 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            df_proj_norm.iloc[:, 0],
            df_proj_norm.iloc[:, 1],
            df_proj_norm.iloc[:, 2],
            s=1.5,
        )
        # plt.show()
        for i in range(len(path)):
            x, y, z = path[i]
            ax.scatter(x, y, z, c="r", marker="o", s=1.75)
        for i in range(len(self.sd.df_population)):
            if self.sd.df_population.loc[i, "score"] == 1:
                x, y, z = df_proj_norm.iloc[i, :]
                ax.scatter(x, y, z, c="g", marker="o", s=1.75)
        self.sd.logger.info(
            f"Scored unique individuals: {sum(self.sd.df_population['score'] == 1)}"
        )
        plt.show()

    def operate(self):
        paths = self.get_or_create_paths(source="generate")
        for path_id, path in enumerate(paths):
            for iteration in range(cfg.NUMBER_OF_ITERATIONS):

                if cfg.PROJECTION_METHOD == "tsne":
                    df_proj = self.tsne_project()
                elif cfg.PROJECTION_METHOD == "umap":
                    df_proj = self.umap_project()
                elif cfg.PROJECTION_METHOD == "pca":
                    raise NotImplementedError("PCA projection not implemented.")
                else:
                    raise ValueError("Invalid projection method specified.")

                df_proj_norm = self.normalize_projection_data(df_proj)

                # sd.df_tsne = df_tsne_norm
                self.sd.df_tsne = df_proj_norm  # temp assinging to tsne for convenience
                # !!!

                self.fit_knn()

                # ####
                # self.sd.logger.info("Calculating Euclidian distances with the corpus")
                # self.sd.index_lookup = {}
                # for i in self.sd.df_tsne.index:
                #     """Find the closest point in the corpus for each individual in the
                #     population.
                #     """
                #     distances = np.linalg.norm(
                #         self.sd.df_corpus.iloc[:, :-2] - self.sd.df_population.iloc[i, :-2],
                #         axis=1,
                #     )
                #     closest_index = np.argmin(distances)
                #     self.sd.index_lookup[i] = closest_index
                # self.sd.logger.info(f"Done calculating the distances with the corpus")

                # self.sd.logger.info("Done projecting.")
                # self.sd.logger.info(
                #     f"Selection threshold divisor: {cfg.SELECTION_THRESHOLD_DIVISOR}"
                # )
                # ####

                self.get_and_rate_selected(path)

                # self.draw_all(df_proj_norm, path)

                # time.sleep(2)  ####

                self.sd.current_population += 1

                df_top = self.apply_selection(cfg.SELECTION_THRESHOLD_DIVISOR)
                self.sd.logger.info(
                    f"Selected, populating from {len(df_top.index)} individuals..."
                )
                df = self.apply_crossover(df_top)
                self.sd.logger.info(f"Crossover resulted {len(df.index)} individuals.")
                # df = self.apply_mutation(df)
                df = self.apply_gaussian_mutation(df)
                self.sd.logger.info(f"Mutation resulted {len(df.index)} individuals.")

                # sample the population to preserve the population size
                if cfg.ELITIST_SELECTION:
                    sample_size = cfg.POPULATION_SIZE - len(df_top.index)
                else:
                    sample_size = cfg.POPULATION_SIZE
                df = df.sample(
                    sample_size,
                    # random_state=42,
                ).reset_index(
                    drop=True
                )  ####

                # elitist selection, bring the top-scoring individuals to
                # the next population
                if cfg.ELITIST_SELECTION:
                    df = pd.concat([df, df_top], ignore_index=True)

                # save df for analysis
                df_save_features = df.copy(deep=True)
                df_save_features["iteration"] = iteration
                df_save_features["path_id"] = path_id
                df_save_features["projection_method"] = cfg.PROJECTION_METHOD
                df_save_features["corpus_method"] = cfg.CORPUS_METHOD
                self.sd.df_analysis_features = pd.concat(
                    [self.sd.df_analysis_features, df_save_features]
                ).reset_index(drop=True)

                # get the closest individuals from the corpus
                # self.sd.index_lookup = {}
                list_closest_indices = []
                for i in df.index:
                    """Find the closest point in the corpus for each individual in the
                    population.
                    """
                    distances = np.linalg.norm(
                        self.sd.df_corpus.iloc[:, :-2] - df.iloc[i, :-2],
                        axis=1,
                    )
                    list_closest_indices.append(np.argmin(distances))
                self.sd.logger.info(f"Done calculating the distances with the corpus")

                df_closest_in_corpus = self.sd.df_corpus.iloc[list_closest_indices, :]
                # df_samples_closest_in_corpus = self.sd.df_samples.iloc[
                #     list_closest_indices, :
                # ]

                self.sd.df_population = df_closest_in_corpus.copy(
                    deep=True
                ).reset_index(drop=True)
                self.sd.df_population["score"] = -1
                self.sd.df_population["pop"] = self.sd.current_population

                self.sd.logger.info(
                    f"New population with {len(self.sd.df_population.index)}"
                )

                # save population for analysis
                df_save_population = self.sd.df_population.copy(deep=True)
                df_save_population["iteration"] = iteration
                df_save_population["path_id"] = path_id
                df_save_population["projection_method"] = cfg.PROJECTION_METHOD
                df_save_population["corpus_method"] = cfg.CORPUS_METHOD
                self.sd.df_analysis_population = pd.concat(
                    [self.sd.df_analysis_population, df_save_population]
                ).reset_index(drop=True)

                # # save samples for analysis
                # df_analysis_samples = df_samples_closest_in_corpus.copy(deep=True)
                # df_analysis_samples["iteration"] = iteration
                # df_analysis_samples["path_id"] = path_id
                # df_analysis_samples["projection_method"] = cfg.PROJECTION_METHOD
                # df_analysis_samples["corpus_method"] = cfg.CORPUS_METHOD
                # self.sd.df_analysis_samples = pd.concat(
                #     [self.sd.df_analysis_samples, df_analysis_samples]
                # ).reset_index(drop=True)pd.DataFrame()
                self.sd.logger.info("- - - - - - - - - - - - - - - - - - - - - - -")
                self.sd.logger.info(
                    f"Finished iteration {iteration} for path {path_id}"
                )
                self.sd.logger.info("---------------------------------------------")

        # save analysis dataframes

        path_features_output = os.path.join(
            self.sd.outpur_dir, f"{self.sd.timestamp}_analysis_features.pkl"
        )
        with open(path_features_output, "wb") as file:
            pickle.dump(self.sd.df_analysis_features, file)
        path_population_output = os.path.join(
            self.sd.outpur_dir, f"{self.sd.timestamp}_analysis_population.pkl"
        )
        with open(path_population_output, "wb") as file:
            pickle.dump(self.sd.df_analysis_population, file)
        path_samples_output = os.path.join(
            self.sd.outpur_dir, f"{self.sd.timestamp}_analysis_samples.pkl"
        )
        # with open(path_samples_output, "wb") as file:
        #     pickle.dump(self.sd.df_analysis_samples, file)
