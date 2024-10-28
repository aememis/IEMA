import traceback
from scipy.io.wavfile import write
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

import networkx as nx
from Individual import Individual


class SharedData:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.output_dir = "output/" + self.timestamp
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = self._create_logger(self.timestamp)

        self.id_counter = 0
        self.G = nx.DiGraph()

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
            "crossover_rate": cfg.CROSSOVER_RATE,
            "elitist_selection": cfg.ELITIST_SELECTION,
            "sample_legth_threshold": (
                cfg.SAMPLES_THRESHOLD_LOW,
                cfg.SAMPLES_THRESHOLD_HIGH,
            ),
            "k_closest_in_corpus": cfg.K_CLOSEST_IN_CORPUS,
            "note": note,
        }
        path_metadata_output = os.path.join(
            self.output_dir, f"{self.timestamp}_metadata.json"
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
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s  %(message)s"
        )

        # Create a file handler
        log_filename = f"log_{timestamp}.log"
        log_filepath = os.path.join(self.output_dir, log_filename)
        if not os.path.exists(log_filepath):
            open(log_filepath, "w").close()
        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
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
        ids = np.arange(len(df_population))
        df_population["id"] = ids
        # self.sd.id_counter = len(df_corpus)
        self.sd.id_counter = df_population.id.max() + 1
        self.sd.df_population = df_population.copy(deep=True)
        self.sd.df_corpus = df_corpus.copy(deep=True)

        for pop_i in self.sd.df_population.index:
            self.sd.G.add_node(
                self.sd.df_population.iloc[pop_i, -1],
                pop=self.sd.current_population,
                sample_id=self.sd.df_population.iloc[pop_i, -2],
            )

        # nodes_container = [
        #     (i, {"pop": self.sd.current_population}) for i in df_population.id
        # ]
        # self.sd.G.add_nodes_from(nodes_container)
        # for node in reversed(list(self.sd.G.nodes(data=True))):
        #     # input(node)
        #     print(node[1]["pop"])

        # self.sd.df_samples = df_samples.copy(deep=True) # temp, samples corrupted
        self.sd.logger.info(
            f"Corpus had {len(self.sd.df_corpus.index)} samples"
        )
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
            # min_dist=0.1,
            # * 2
            # * self.sd.current_population,  # Default is 0.1, controls how tightly UMAP is allowed to pack points together
        )  #### ????
        umap_data = umap_model.fit_transform(self.sd.df_population.iloc[:, :6])
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
        tsne_data = tsne.fit_transform(self.sd.df_population.iloc[:, :6])
        self.sd.logger.info("Done TSNE projection.")
        return pd.DataFrame(tsne_data)

    def fit_knn(self):
        self.sd.knn = NearestNeighbors(n_neighbors=cfg.K)
        self.sd.knn.fit(self.sd.df_tsne.to_numpy())

    def get_and_rate_selected(self, path):
        """Rate the selected individuals based on the path."""

        list_closest_index_proj = []
        for point in path:
            user_point = np.array([point])
            distances, indices = self.sd.knn.kneighbors(user_point)
            list_closest_indices = indices[0]
            closest_index_proj = list_closest_indices[0]
            list_closest_index_proj.append(closest_index_proj)

        # rate the closest individual
        self.sd.df_population.loc[list_closest_index_proj, "score"] = 1
        return list_closest_index_proj

    def apply_selection(self, threshold):
        # df_top = self.sd.df_population.nlargest(threshold, "score", keep="first")
        df_top = self.sd.df_population[self.sd.df_population["score"] == 1]
        self.sd.logger.info(f"{len(df_top.index)} rated individual found.")
        # self.sd.df_dead = pd.concat(
        #     [self.sd.df_dead, self.sd.df_population], ignore_index=True
        # ).reset_index(drop=True)
        # self.sd.df_population.drop(self.sd.df_population.index, inplace=True)
        return df_top.copy(deep=True).reset_index(drop=True)

    def _cross_parts(self, arr1, arr2):
        crossover_point = np.random.randint(1, len(arr1) - 1)
        offspring1 = np.concatenate(
            (arr1[:crossover_point], arr2[crossover_point:])
        )
        offspring2 = np.concatenate(
            (arr2[:crossover_point], arr1[crossover_point:])
        )
        return [offspring1, offspring2]

    def apply_crossover_multiple_per_individual(self, df):
        nd_new_values = []
        for i in range(len(df.index)):
            for j in range(i + 1, len(df.index), 1):
                if np.random.random() < cfg.CROSSOVER_RATE:
                    arr1 = df.iloc[i].values[:6]
                    arr2 = df.iloc[j].values[:6]
                    list_new_values = self._cross_parts(arr1, arr2)
                    list_new_values_reshaped = np.append(
                        list_new_values,
                        np.array(
                            [[None, self.sd.current_population, (i, j)]] * 2,
                            dtype=object,
                        ),
                        axis=1,
                    )
                    nd_new_values.extend(list_new_values_reshaped)
        if len(nd_new_values) > 0:
            df_new_values = pd.DataFrame(
                nd_new_values, columns=cfg.DATA_FIELDS_CORPUS
            )
            # df = pd.concat([df, df_new_values], ignore_index=True)
            df_new_values["score"] = -1
            return df_new_values
        else:
            self.sd.logger.warning("No crossover applied, that's so weird!")
            return df

    def apply_crossover_unique_pairs(self, df):
        nd_new_values = []
        indices = df.index.to_numpy()
        np.random.shuffle(indices)
        # augment the indices to have more pairs
        # indices = np.append(indices, np.random.choice(indices, len(indices)))
        # np.random.shuffle(indices)
        if len(indices) % 2 != 0:
            self.sd.logger.warning(
                "Odd number of individuals, one will be reused for crossover."
            )
            indices = np.append(indices, np.random.choice(indices, 1))
        ids_relation = []
        for i in range(0, len(indices), 2):
            if i + 1 < len(indices):
                # if np.random.random() < cfg.CROSSOVER_RATE:
                arr1 = df.iloc[indices[i]].values[:6]
                arr2 = df.iloc[indices[i + 1]].values[:6]
                list_new_values = self._cross_parts(arr1, arr2)
                list_new_values_reshaped = np.append(
                    list_new_values,
                    np.array(
                        [
                            [
                                None,  # score
                                self.sd.current_population,  # pop
                                None,  # sample_id
                                None,  # id
                            ]
                        ]
                        * 2,
                        dtype=object,
                    ),
                    axis=1,
                )
                new_ids = np.arange(self.sd.id_counter, self.sd.id_counter + 2)
                list_new_values_reshaped[:, -1] = new_ids  # assign new ids
                ids_relation.append((df.loc[indices[i], "id"], new_ids[0]))
                ids_relation.append((df.loc[indices[i], "id"], new_ids[1]))
                ids_relation.append((df.loc[indices[i + 1], "id"], new_ids[0]))
                ids_relation.append((df.loc[indices[i + 1], "id"], new_ids[1]))
                self.sd.id_counter += 2
                nd_new_values.extend(list_new_values_reshaped)

        if len(nd_new_values) > 0:
            df_new_values = pd.DataFrame(
                nd_new_values, columns=cfg.DATA_FIELDS_CORPUS
            )
            df_new_values["score"] = -1
            return df_new_values, ids_relation
        else:
            self.sd.logger.warning("No crossover applied, this's so weird.")
            raise Exception()
            # input("!!!")
            # return df

    def apply_mutation(self, df):
        list_new_datapoints = []
        for i in range(len(df.index)):
            if np.random.random() < cfg.MUTATION_RATE:
                new_datapoint = df.iloc[i].copy(deep=True)
                for j in range(len(new_datapoint[:6])):
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

    def apply_gaussian_mutation_OLD(self, df):
        """Apply Gaussian mutation to the population. This is deprecated, it is appending
        new individuals to the population instead of mutating the existing ones.
        """
        list_new_datapoints = []
        for i in range(len(df.index)):
            new_datapoint = df.iloc[i].copy(deep=True)
            for j in range(len(new_datapoint[:6])):
                if np.random.random() < cfg.MUTATION_RATE:
                    new_value = new_datapoint[j] + np.random.normal(
                        0, cfg.MUTATION_STRENGTH
                    )
                    new_datapoint[j] = np.clip(new_value, -1, 1)

            list_new_datapoints.append(new_datapoint)
        if list_new_datapoints:
            df_new_datapoints = pd.DataFrame(
                list_new_datapoints, columns=df.columns
            )
            df = pd.concat([df, df_new_datapoints])
        return df

    def apply_gaussian_mutation(self, df):
        for i in range(len(df.index)):
            for j in range(len(df.columns) - 3):
                if np.random.random() < cfg.MUTATION_RATE:
                    new_value = df.iloc[i, j] + np.random.normal(
                        0, cfg.MUTATION_STRENGTH
                    )
                    df.iloc[i, j] = np.clip(new_value, -1, 1)
        return df

    def draw_all(self, path, list_closest_index_proj):
        # draw projection and the selected individuals along with the path, in different color and 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            self.sd.df_tsne.iloc[:, 0],
            self.sd.df_tsne.iloc[:, 1],
            self.sd.df_tsne.iloc[:, 2],
            s=1.5,
        )

        # plot the path points
        for i in range(len(path)):
            x, y, z = path[i]
            ax.scatter(x, y, z, c="r", marker="o", s=1.75)

        # plot the selected individuals
        for i in range(len(self.sd.df_population)):
            if self.sd.df_population.loc[i, "score"] == 1:
                x, y, z = self.sd.df_tsne.iloc[i, :]
                ax.scatter(x, y, z, c="g", marker="o", s=1.75)

        # plot the the error lines
        list_distances = []
        for i in range(len(path)):
            x, y, z = path[i]
            user_point = np.array([[x, y, z]])
            distances, indices = self.sd.knn.kneighbors(user_point)
            closest_index_proj = indices[0][0]
            list_distances.append(distances[0][0])
            x, y, z = self.sd.df_tsne.iloc[closest_index_proj, :]
            ax.plot(
                [path[i][0], x], [path[i][1], y], [path[i][2], z], c="b", lw=0.5
            )

        # add legend
        ax.scatter([], [], [], c="r", marker="o", s=1.75, label="Path points")
        ax.scatter(
            [], [], [], c="g", marker="o", s=1.75, label="Selected individuals"
        )
        ax.plot([], [], [], c="b", lw=0.5, label="Error")
        ax.legend()

        # stats
        ax.text2D(
            0.05,
            0.95,
            f"Scored unique individuals: {sum(self.sd.df_population['score'] == 1)}",
            transform=ax.transAxes,
        )
        ax.text2D(
            0.05,
            0.90,
            f"Mean error: {np.mean(list_distances):.3f}",
            transform=ax.transAxes,
        )
        self.sd.logger.info(
            f"Scored unique individuals: {sum(self.sd.df_population['score'] == 1)}"
        )
        plt.show()

    def reseed_from_corpus(self, df):
        """Reseed the population with random samples from the corpus."""
        if len(df.index) < cfg.POPULATION_SIZE:
            self.sd.logger.info(
                "Reseeding the population with "
                f"{cfg.POPULATION_SIZE - len(df.index)} samples from the corpus."
            )
            indices = np.random.choice(
                self.sd.df_corpus.index, cfg.POPULATION_SIZE - len(df.index)
            )
            df_new = (
                self.sd.df_corpus.iloc[indices, :]
                .copy(deep=True)
                .reset_index(drop=True)
            )
            df_new["score"] = -1
            df_new["pop"] = self.sd.current_population
            assert not df["id"].max() > self.sd.id_counter, "noliy"
            df_new["id"] = np.arange(len(df_new.index)) + self.sd.id_counter
            self.sd.id_counter = df_new["id"].max() + 1
            df = pd.concat([df, df_new])  # , ignore_index=True)
            df.reset_index(drop=True, inplace=True)
            self.sd.logger.info(
                f"Population size after reseeding: {len(df.index)}"
            )
        return df

    def visualize_evo_graph(self):
        layers = {}
        for node in self.sd.G.nodes(data=True):
            if node[1]["pop"] in layers:
                layers[node[1]["pop"]].append(node[0])
            else:
                layers[node[1]["pop"]] = [node[0]]
        pos = nx.multipartite_layout(self.sd.G, subset_key=layers)
        nx.draw(
            self.sd.G,
            pos,
            labels=nx.get_node_attributes(self.sd.G, "sample_id"),
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            node_size=200,
            font_size=16,
        )
        plt.title("Graph with Multiple Parents")
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
                self.sd.df_tsne = (
                    df_proj_norm  # temp assinging to tsne for convenience
                )
                # !!!

                self.fit_knn()

                # print("ID COUNTER 0")
                # input(self.sd.id_counter)

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

                list_closest_index_proj = self.get_and_rate_selected(path)

                # if iteration % 10 == 0:
                #     self.draw_all(path, list_closest_index_proj)

                # time.sleep(2)  ####

                # for pop_i in self.sd.df_population.index:
                #     self.sd.G.add_node(
                #         self.sd.df_population.iloc[pop_i, -1],
                #         pop=self.sd.current_population,
                #         sample_id=self.sd.df_population.iloc[pop_i, -2],
                #     )

                # self.sd.current_population += 1

                # print("unique sample_ids in popu")
                # print(self.sd.df_population.iloc[:, -2].unique())
                # print(len(self.sd.df_population.iloc[:, -2].unique()))
                # print(len(self.sd.df_population.iloc[:, -2]))
                # input()
                # print("unique ids in popu")
                # print(self.sd.df_population.iloc[:, -1].unique())
                # print(len(self.sd.df_population.iloc[:, -1].unique()))
                # print(len(self.sd.df_population.iloc[:, -1]))
                # input()

                df_top = self.apply_selection(cfg.SELECTION_THRESHOLD_DIVISOR)

                # print("unique sample_ids in selected")
                # print(df_top.iloc[:, -2].unique())
                # print(len(df_top.iloc[:, -2].unique()))
                # print(len(df_top.iloc[:, -2]))
                # input()

                self.sd.logger.info(
                    f"Selected, populating from {len(df_top.index)} individuals..."
                )
                df_recombined, ids_relation = self.apply_crossover_unique_pairs(
                    df_top
                )

                show_graph = False
                if show_graph and iteration % 10 == 0:
                    self.visualize_evo_graph()

                # print("size of df_recombined")
                # print(len(df_recombined.index))
                # print("ID COUNTER AFTER CROSSOVER")
                # input(self.sd.id_counter)

                len_top = len(df_top.index)
                len_recombined = len(df_recombined.index)
                # assert len_top + len_top % 2 == len_recombined, (
                #     "Bad population size after unique pairwise crossover: "
                #     + f"{len_top} vs {len_recombined}"
                # )
                self.sd.logger.info(
                    f"Crossover resulted {len_recombined} individuals."
                )

                # df_mutated = self.apply_mutation(df_recombined)
                df = self.apply_gaussian_mutation(df_recombined)

                # print("size of df mutation")
                # print(len(df.index))
                # print("ID COUNTER AFTER MUTATION")
                # input(self.sd.id_counter)

                self.sd.logger.info(
                    f"Mutation resulted {len(df.index)} individuals."
                )

                # sample the population to preserve the population size
                # if cfg.ELITIST_SELECTION:
                #     sample_size = min(
                #         cfg.POPULATION_SIZE - len(df_top.index), len(df.index)
                #     )
                # else:
                #     sample_size = min(cfg.POPULATION_SIZE, len(df.index))
                # df = df.sample(
                #     sample_size,
                #     # random_state=42,
                # ).reset_index(
                #     drop=True
                # )  ####

                # elitist selection, bring the top-scoring individuals to
                # the next population
                if (
                    cfg.ELITIST_SELECTION
                ):  ### IF ENABLING, CHECK IF THESE ARE ASSIGNED WITH A CORRECT ID, PROB NOT
                    sample_size = cfg.POPULATION_SIZE - len(df.index)
                    df = df.sample(sample_size).reset_index(drop=True)
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

                # for ind in self.sd.df_corpus.index:
                #     if any(self.sd.df_corpus.iloc[ind, :] == 0):
                #         audio_signal = self.sd.df_samples.iloc[ind, 0]
                #         audio_signal = np.array(audio_signal)
                #         write("audio.wav", 44100, np.int16(audio_signal * 32767))
                #         print(f"Audio signal saved.")
                #         input(self.sd.df_corpus.iloc[ind, :])

                # get the closest individuals from the corpus
                # self.sd.index_lookup = {}
                list_closest_indices = []
                for i in df.index:
                    """Find the closest point in the corpus for each individual in the
                    population.
                    """
                    distances = self.sd.df_corpus.iloc[:, :6] - df.iloc[i, :6]
                    distances_norm = np.linalg.norm(
                        distances.astype(float),
                        axis=1,
                    )
                    closest_indices = np.argsort(distances_norm)[
                        : cfg.K_CLOSEST_IN_CORPUS
                    ]
                    # list_closest_indices.append(np.argmin(distances_norm))
                    list_closest_indices.extend(closest_indices)
                self.sd.logger.info(
                    f"Done calculating the distances with the corpus"
                )
                df_closest_in_corpus = self.sd.df_corpus.iloc[
                    list_closest_indices, :
                ].copy(deep=True)
                df_closest_in_corpus["id"] = df["id"].values
                assert len(df_closest_in_corpus["id"].unique()) == len(
                    df_closest_in_corpus["id"]
                ), AssertionError(
                    "There are duplicate ids in the closest individuals "
                    "from the corpus."
                )
                assert len(list_closest_indices) == len(
                    df_closest_in_corpus.index
                ), AssertionError(
                    "The number of closest individuals from the corpus "
                    "does not match the number of indices."
                )

                # print("dupe samples in corpus:")
                # print(len(df_closest_in_corpus[df_closest_in_corpus.duplicated()]))
                # print(df_closest_in_corpus[df_closest_in_corpus.duplicated()])
                # input()

                # print("size of before reseed")
                # print(len(df_closest_in_corpus.index))
                # print("ID COUNTER BEFORE RESEED")
                # input(self.sd.id_counter)

                # reseed the population with samples from the corpus
                df_reseed = self.reseed_from_corpus(df_closest_in_corpus)

                # print("size of after reseed")
                # print(len(df_reseed.index))
                # print("ID COUNTER AFTER RESEED")
                # input(self.sd.id_counter)

                assert all(
                    i in df_closest_in_corpus["id"].values
                    for i in np.array(ids_relation)[:, 1]
                ), "!!! Not all ids in the relation are in the closest individuals from the corpus."

                # print("size of ids relation")
                # print(len(ids_relation))
                # print("size of reseeded individuals")
                # print(len(df_recombined.index))
                assert len(ids_relation) / 2 == len(
                    df_recombined.index
                ), "!!! The number of ids in the relation does not match the number of reseeded individuals."

                self.sd.df_population = df_reseed.copy(deep=True).reset_index(
                    drop=True
                )
                self.sd.df_population["score"] = -1
                self.sd.df_population["pop"] = self.sd.current_population
                # self.sd.df_population["id"] = (
                #     np.arange(len(self.sd.df_population)) + self.sd.id_counter
                # )
                # self.sd.id_counter = self.sd.df_population["id"].max()

                self.sd.logger.info(
                    f"New population with {len(self.sd.df_population.index)}"
                )

                for pop_i in self.sd.df_population.index:
                    self.sd.G.add_node(
                        self.sd.df_population.iloc[pop_i, -1],
                        pop=self.sd.current_population,
                        sample_id=self.sd.df_population.iloc[pop_i, -2],
                    )

                print("current population")
                print(self.sd.current_population)
                if self.sd.current_population == 150:
                    print(len(self.sd.df_population.index))
                    input(150)

                try:
                    nodes_added = set()
                    for new_i in range(len(ids_relation)):
                        if not ids_relation[new_i][1] in nodes_added:
                            self.sd.G.add_node(
                                ids_relation[new_i][1],
                                pop=self.sd.current_population,
                            )
                            nodes_added.add(ids_relation[new_i][1])
                        self.sd.G.add_edge(
                            ids_relation[new_i][0], ids_relation[new_i][1]
                        )
                except Exception as e:
                    print(e)
                    # print(list_new_values_reshaped)
                    # print(indices[i], indices[i + 1])
                    # print(df.iloc[indices[i], :], df.iloc[indices[i + 1], :])
                    # print(df.iloc[indices[i], -1], df.iloc[indices[i + 1], -1])
                    print(traceback.format_exc())
                    input()

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
                self.sd.logger.info(
                    "- - - - - - - - - - - - - - - - - - - - - - -"
                )
                self.sd.logger.info(
                    f"Finished iteration {iteration} for path {path_id}"
                )
                self.sd.logger.info(
                    "---------------------------------------------"
                )

        # save the graph
        path_graph_output = os.path.join(
            self.sd.output_dir,
            f"{self.sd.timestamp}_analysis_evo_graph.gpickle",
        )
        with open(path_graph_output, "wb") as file:
            pickle.dump(self.sd.G, file)

        # save analysis dataframes
        path_features_output = os.path.join(
            self.sd.output_dir, f"{self.sd.timestamp}_analysis_features.pkl"
        )
        with open(path_features_output, "wb") as file:
            pickle.dump(self.sd.df_analysis_features, file)
        path_population_output = os.path.join(
            self.sd.output_dir, f"{self.sd.timestamp}_analysis_population.pkl"
        )
        with open(path_population_output, "wb") as file:
            pickle.dump(self.sd.df_analysis_population, file)
        # path_samples_output = os.path.join(
        #     self.sd.output_dir, f"{self.sd.timestamp}_analysis_samples.pkl"
        # )
        # with open(path_samples_output, "wb") as file:
        #     pickle.dump(self.sd.df_analysis_samples, file)
