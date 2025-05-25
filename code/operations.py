import json
import logging
import os
import pickle
import traceback
from datetime import datetime

import config as cfg
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from umap import UMAP
from sklearn.manifold import Isomap


class SharedData:
    """SharedData is a container for all data and state shared across different
    components and threads, including the operator and evaluation logic.
    It manages configuration, logging, metadata, and runtime variables.
    """

    def __init__(self, session_timestamp, run_id, path_id):
        self.start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_timestamp = session_timestamp
        self.run_id = run_id
        self.path_id = path_id
        self.output_dir = (
            f"output/{session_timestamp}/"
            f"run_{str(run_id).zfill(3)}/"
            f"path_{str(path_id).zfill(3)}"
        )
        self.output_prefix = ""
        self.session_dir = f"output/{session_timestamp}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        logger_name = self.output_dir
        if not logging.getLogger(logger_name).hasHandlers():
            self.logger = self._create_logger(logger_name)
        else:
            self.logger = logging.getLogger(logger_name)

        self.id_counter = 0
        self.G = nx.DiGraph()

        self.metadata = {
            "start_timestamp": self.start_timestamp,
            "session_timestamp": self.session_timestamp,
            "run_id": self.run_id,
            "path_id": self.path_id,
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
            "augment_ratio_crossover": cfg.AUGMENT_RATIO_CROSSOVER,
            "unique_closest_in_corpus": cfg.UNIQUE_CLOSEST_IN_CORPUS,
            "note": cfg.NOTE,
        }
        path_metadata_output = os.path.join(
            self.output_dir,
            f"{self.output_prefix}run_config.json",
        )
        with open(path_metadata_output, "w") as file:
            json.dump(self.metadata, file, indent=4)

        self.logger.info(f"Metadata: {self.metadata}")

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

    def _create_logger(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s  %(message)s"
        )

        # Create a file handler
        log_filename = "session.log"
        log_filepath = os.path.join(self.session_dir, log_filename)
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
        self.sd.id_counter = df_population.id.max() + 1
        self.sd.df_population = df_population.copy(deep=True)
        self.sd.df_corpus = df_corpus.copy(deep=True)

        for pop_i in self.sd.df_population.index:
            self.sd.G.add_node(
                self.sd.df_population.iloc[pop_i, -1],
                pop=self.sd.current_population,
                sample_id=self.sd.df_population.iloc[pop_i, -2],
            )

        # currently not supported as it's not needed for the research
        # self.sd.df_samples = df_samples.copy(deep=True)
        self.sd.logger.info(
            f"Corpus had {len(self.sd.df_corpus.index)} samples"
        )
        self.sd.logger.info(
            "Initiated from corpus with "
            f"{len(self.sd.df_population.index)} individuals"
        )

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

    def isomap_project(self):
        self.sd.logger.info("Isomap projecting the population...")
        isomap = Isomap(
            n_components=3,
        )
        isomap_data = isomap.fit_transform(self.sd.df_population.iloc[:, :6])
        self.sd.logger.info("Done Isomap projection.")
        return pd.DataFrame(isomap_data)

    def umap_project(self):
        self.sd.logger.info("UMAP projecting the population...")
        umap_model = UMAP(
            n_components=3,
        )
        umap_data = umap_model.fit_transform(self.sd.df_population.iloc[:, :6])
        self.sd.logger.info("Done UMAP projection.")

        return pd.DataFrame(umap_data)

    def tsne_project(self):
        tsne = TSNE(
            n_components=3,
            perplexity=10,
        )
        tsne_data = tsne.fit_transform(self.sd.df_population.iloc[:, :6])
        self.sd.logger.info("Done TSNE projection.")
        return pd.DataFrame(tsne_data)

    def pca_project(self):
        self.sd.logger.info("PCA projecting the population...")
        pca = PCA(n_components=3)
        pca_data = pca.fit_transform(self.sd.df_population.iloc[:, :6])
        self.sd.logger.info("Done PCA projection.")
        return pd.DataFrame(pca_data)

    def fit_knn(self):
        self.sd.knn = NearestNeighbors(n_neighbors=cfg.K)
        self.sd.knn.fit(self.sd.df_tsne.to_numpy())

    def get_and_rate_selected(self, path):
        """Rate the selected individuals based on the path."""

        list_closest_index_proj = []

        # calculate the closest
        for point in path:
            user_point = np.array([point])
            distances, indices = self.sd.knn.kneighbors(user_point)
            list_closest_indices = indices[0]
            closest_index_proj = list_closest_indices[0]
            list_closest_index_proj.append(closest_index_proj)

        # rate the closest individual
        self.sd.df_population.loc[list_closest_index_proj, "score"] = 1
        return list_closest_index_proj

    def get_and_rate_selected_random(self, path):
        """Rate the selected individuals based on the path."""

        # select random individuals
        list_closest_index_proj = np.random.choice(
            np.arange(len(self.sd.df_tsne.index)),
            len(path),
            replace=False,
        )

        # rate the closest individual
        self.sd.df_population.loc[list_closest_index_proj, "score"] = 1
        return list_closest_index_proj

    def apply_selection(self, threshold):
        df_top = self.sd.df_population[self.sd.df_population["score"] == 1]
        self.sd.logger.info(f"{len(df_top.index)} rated individual found.")
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
            df_new_values["score"] = -1
            return df_new_values
        else:
            self.sd.logger.warning("No crossover applied, this is so weird!")
            return df

    def apply_crossover_unique_pairs(self, df):
        nd_new_values = []
        indices = df.index.to_numpy()
        np.random.shuffle(indices)

        # augment the indices to have more pairs
        if cfg.AUGMENT_RATIO_CROSSOVER > 0:
            indices = np.append(
                indices,
                np.random.choice(
                    indices, int(len(indices) * cfg.AUGMENT_RATIO_CROSSOVER)
                ),
            )

        if len(indices) % 2 != 0:
            self.sd.logger.warning(
                "Odd number of individuals, one will be reused for crossover."
            )
            indices = np.append(indices, np.random.choice(indices, 1))
        ids_relation = []
        for i in range(0, len(indices), 1):  # 2
            if i + 1 < len(indices):
                if np.random.random() < cfg.CROSSOVER_RATE:
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
                    new_ids = np.arange(
                        self.sd.id_counter, self.sd.id_counter + 2
                    )
                    list_new_values_reshaped[:, -1] = new_ids  # assign new ids
                    ids_relation.append((df.loc[indices[i], "id"], new_ids[0]))
                    ids_relation.append((df.loc[indices[i], "id"], new_ids[1]))
                    ids_relation.append(
                        (df.loc[indices[i + 1], "id"], new_ids[0])
                    )
                    ids_relation.append(
                        (df.loc[indices[i + 1], "id"], new_ids[1])
                    )
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
        """Apply Gaussian mutation to the population.

        This is deprecated, it is appending new individuals to the population
        instead of mutating the existing ones.
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
                    new_value = df.iloc[i, j] + (
                        cfg.MUTATION_STRENGTH * np.random.choice([-1, 1])
                    )
                    df.iloc[i, j] = np.clip(np.int64(new_value), -1, 1)
        return df

    def draw_all(self, path):
        # draw projection and the selected individuals along with the path,
        # in different color and 3D
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
            (
                "Scored unique individuals: "
                f"{sum(self.sd.df_population['score'] == 1)}"
            ),
            transform=ax.transAxes,
        )
        ax.text2D(
            0.05,
            0.90,
            f"Mean error: {np.mean(list_distances):.3f}",
            transform=ax.transAxes,
        )
        self.sd.logger.info(
            "Scored unique individuals: "
            f"{sum(self.sd.df_population['score'] == 1)}"
        )
        plt.show()

    def reseed_from_corpus(self, df):
        """Reseed the population with random samples from the corpus."""
        if len(df.index) < cfg.POPULATION_SIZE:
            self.sd.logger.info(
                f"Reseeding with {cfg.POPULATION_SIZE - len(df.index)} "
                "samples from the corpus."
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
            df = pd.concat([df, df_new])
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

    def operate(self, path_id, path):
        for iteration in range(cfg.NUMBER_OF_ITERATIONS):
            if cfg.PROJECTION_METHOD == "tsne":
                df_proj = self.tsne_project()
            elif cfg.PROJECTION_METHOD == "umap":
                df_proj = self.umap_project()
            elif cfg.PROJECTION_METHOD == "pca":
                df_proj = self.pca_project()
            elif cfg.PROJECTION_METHOD == "isomap":
                df_proj = self.isomap_project()
            else:
                raise ValueError("Invalid projection method specified.")

            df_proj_norm = self.normalize_projection_data(df_proj)

            # using 'tsne' name for convenience,
            # however, it can be any projection method as selected in config
            self.sd.df_tsne = df_proj_norm

            self.fit_knn()

            list_closest_index_proj = self.get_and_rate_selected_random(path)

            df_top = self.apply_selection(cfg.SELECTION_THRESHOLD_DIVISOR)

            self.sd.logger.info(
                f"Selected, populating from {len(df_top.index)} "
                "individuals..."
            )
            df_recombined, ids_relation = self.apply_crossover_unique_pairs(
                df_top
            )

            show_graph = False
            if show_graph and iteration % 10 == 0:
                self.visualize_evo_graph()

            len_top = len(df_top.index)
            len_recombined = len(df_recombined.index)
            self.sd.logger.info(
                f"Crossover resulted {len_recombined} individuals."
            )

            df = self.apply_gaussian_mutation(df_recombined)

            self.sd.logger.info(
                f"Mutation resulted {len(df.index)} individuals."
            )

            # elitist selection, bring the top-scoring individuals to
            # the next population
            # currently not supported, as it is not needed for the research
            if cfg.ELITIST_SELECTION == "true":
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

            # get the closest individuals from the corpus
            list_closest_indices = []
            for i in df.index:
                """Find the closest point in the corpus for each
                individual in the population.
                """
                distances = self.sd.df_corpus.iloc[:, :6] - df.iloc[i, :6]
                distances_norm = np.linalg.norm(
                    distances.astype(float),
                    axis=1,
                )
                closest_indices = np.argsort(distances_norm)[
                    :1  # cfg.K_CLOSEST_IN_CORPUS
                ]

                if cfg.UNIQUE_CLOSEST_IN_CORPUS:
                    if not self.sd.df_analysis_population.empty:
                        used_sample_ids = set(
                            self.sd.df_analysis_population.sample_id.values
                        )
                        for i_closest in range(len(self.sd.df_corpus.index)):
                            candidate_sample_id = self.sd.df_corpus.iloc[
                                closest_indices, :
                            ].sample_id.values[0]
                            if candidate_sample_id not in used_sample_ids:
                                break
                            closest_indices = np.argsort(distances_norm)[
                                i_closest : i_closest + 1
                            ]
                        else:
                            self.sd.logger.warning(
                                "No unique closest individual left."
                            )
                            raise ValueError(
                                "No unique closest individual left."
                            )

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

            # reseed the population with samples from the corpus
            df_reseed = self.reseed_from_corpus(df_closest_in_corpus)

            assert all(
                i in df_closest_in_corpus["id"].values
                for i in np.array(ids_relation)[:, 1]
            ), (
                "!!! Not all ids in the relation are in the "
                "closest individuals from the corpus."
            )

            assert len(ids_relation) / 2 == len(df_recombined.index), (
                "!!! The number of ids in the relation does not "
                "match the number of reseeded individuals."
            )

            self.sd.df_population = df_reseed.copy(deep=True).reset_index(
                drop=True
            )
            self.sd.df_population["score"] = -1
            self.sd.current_population += 1
            self.sd.df_population["pop"] = self.sd.current_population

            self.sd.logger.info(
                f"New population with {len(self.sd.df_population.index)}"
            )

            for pop_i in self.sd.df_population.index:
                self.sd.G.add_node(
                    self.sd.df_population.iloc[pop_i, -1],
                    pop=self.sd.current_population,
                    sample_id=self.sd.df_population.iloc[pop_i, -2],
                )

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
                print(traceback.format_exc())
                input("ERROR")

            # save population for analysis
            df_save_population = self.sd.df_population.copy(deep=True)
            df_save_population["iteration"] = iteration
            df_save_population["path_id"] = path_id
            df_save_population["projection_method"] = cfg.PROJECTION_METHOD
            df_save_population["corpus_method"] = cfg.CORPUS_METHOD
            self.sd.df_analysis_population = pd.concat(
                [self.sd.df_analysis_population, df_save_population]
            ).reset_index(drop=True)

            self.sd.logger.info("- - - - - - - - - - - - - - - - - - - - - - -")
            self.sd.logger.info(
                f"Session: {self.sd.session_timestamp} "
                f"Run: {self.sd.run_id} "
                f"Path: {self.sd.path_id}"
            )
            self.sd.logger.info(f"Finished iteration {iteration}")
            self.sd.logger.info("---------------------------------------------")

        # save the graph
        path_graph_output = os.path.join(
            self.sd.output_dir,
            f"{self.sd.output_prefix}analysis_evo_graph.gpickle",
        )
        with open(path_graph_output, "wb") as file:
            pickle.dump(self.sd.G, file)

        # save analysis dataframes
        path_features_output = os.path.join(
            self.sd.output_dir, f"{self.sd.output_prefix}analysis_features.pkl"
        )
        with open(path_features_output, "wb") as file:
            pickle.dump(self.sd.df_analysis_features, file)
        path_population_output = os.path.join(
            self.sd.output_dir,
            f"{self.sd.output_prefix}analysis_population.pkl",
        )
        with open(path_population_output, "wb") as file:
            pickle.dump(self.sd.df_analysis_population, file)
