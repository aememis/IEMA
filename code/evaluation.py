import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


class Evaluation:
    def __init__(self, record_timestamp):
        self.record_timestamp = record_timestamp
        print(f"Running: {self.record_timestamp}")

    def load_data(self):
        print("Loading data...")

        # load the audio features
        with open("features.pkl", "rb") as f:  # gaussian_features_norm.pkl
            df_features = pickle.load(f)
        self.nd_features = df_features.values

        # load the evolution history
        with open(
            f"output/{self.record_timestamp}/{self.record_timestamp}_analysis_population.pkl",
            "rb",
        ) as f:
            self.df_iterations = pickle.load(f)

    def calculate_functional_variance(self):
        print("Calculating functional variance...")
        self.all_pops_var = []
        for pop in self.df_iterations["pop"].unique():
            df_gen = self.df_iterations.loc[
                self.df_iterations["pop"] == pop,
                ["p1", "p2", "p3", "p4", "p5", "p6"],
            ]
            var = df_gen.var(axis=0)
            self.all_pops_var.append(np.mean(var))

        # plot
        plt.plot(self.all_pops_var)
        plt.xlabel("Population")
        plt.ylabel("Variance")
        plt.title("Variance of each population")
        plt.ylim(0, 0.2)
        plt.savefig(f"output/{self.record_timestamp}/population_diversity.png")
        # plt.show()

    def load_phylo_data(self):
        # load the graph to nx
        ts = self.record_timestamp
        with open(
            f"output/{ts}/{ts}_analysis_evo_graph.gpickle",
            "rb",
        ) as f:
            self.G = pickle.load(f)

        # G.number_of_nodes()

        # get a copy of the graph
        self.G_plot = self.G.copy()

        # make bidirectional by adding the reverse edges
        for u, v in self.G_plot.edges():
            self.G_plot.add_edge(v, u)

    def calculate_pairwise_distances(self):
        # get max population
        self.max_pop = self.df_iterations["pop"].max()

        all_mpd_pops = []
        all_pops_unique = (
            pd.Series([node[1]["pop"] for node in self.G_plot.nodes(data=True)])
            .sort_values()
            .unique()
        )

        # calculate for the last population
        for pop in [self.max_pop]:  # DEL sorted(all_pops_unique):
            leaf_nodes = [
                n for n in self.G_plot.nodes(data=True) if n[1]["pop"] == pop
            ]
            G_upto_pop = self.G_plot.subgraph(
                [
                    n[0]
                    for n in self.G_plot.nodes(data=True)
                    if n[1]["pop"] <= pop
                ]
            )

            # for each unique pair of leaf nodes,
            # calculate the shortest path between them
            pairwise_distances = {}
            no_path = []
            for i in range(len(leaf_nodes)):
                for j in range(i + 1, len(leaf_nodes)):
                    try:
                        dist = nx.shortest_path_length(
                            G_upto_pop, leaf_nodes[i][0], leaf_nodes[j][0]
                        )
                        if dist > 2:
                            pairwise_distances[(i, j)] = dist + 1
                    except nx.NetworkXNoPath:
                        no_path.append((i, j))
                        # print(f"No path between {leaf_nodes[i][0]} and {leaf_nodes[j][0]}")

            pairwise_distances_values = np.array(
                list(pairwise_distances.values())
            )

            mpd_pop = (
                pairwise_distances_values.sum()
                * 2
                / (len(leaf_nodes) * (len(leaf_nodes) - 1))
            )
            all_mpd_pops.append(mpd_pop)
            # print(
            #     f"Population {str(pop).zfill(3)} "
            #     f"Mean Pairwise Distances: {mpd_pop.round(2)} ",
            # )

        self.final_mpd = all_mpd_pops[-1]

        # plot
        plt.clf()
        plt.plot(all_mpd_pops)
        plt.xlabel("Population")
        plt.ylabel("Mean Pairwise Distance")
        plt.title("Mean Pairwise Distances for Each Population")
        plt.savefig(
            f"output/{self.record_timestamp}/mean_pairwise_distance.png"
        )
        # plt.show()

    def calculate_root_contribution_index(self):
        # get the roots based on the condition that they have no incoming edges
        roots = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]

        # calculate the number of leaves that they lead to, for each root
        root_contributions = []
        for root in roots:
            descs = nx.descendants(self.G, root)
            # leaf_descs = [n for n in descs if G.nodes[n]["pop"] == max_pop]
            root_age = self.max_pop - self.G.nodes[root]["pop"]
            contribution = len(descs) / root_age
            root_contributions.append(contribution)

        root_contributions = np.array(root_contributions)

        self.final_rci = root_contributions.sum() / self.G.number_of_nodes()

    def calculate_phylo_diversity_novelty_index(self):
        print("Calculating phylogenetic diversity and novelty index...")
        self.load_phylo_data()
        self.calculate_pairwise_distances()
        self.calculate_root_contribution_index()
        self.final_pdni = self.final_mpd / self.final_rci

    def calculate_categorical_diversity_novelty_index(self):
        print("Calculating categorical diversity and novelty index...")
        # load the filenames of the audio files
        with open("filenames.pkl", "rb") as f:
            self.df_filenames = pickle.load(f)
        self.df_filenames.filename = self.df_filenames.filename.apply(
            os.path.splitext
        ).str[0]

        # Load the metadata file
        self.df_metacoll = pd.read_csv(
            r"D:\datasets\FSD50K\FSD50K.metadata\collection\collection_dev.csv"
        )
        self.df_metacoll.mids = self.df_metacoll.mids.str.split(",")
        self.df_metacoll.fname = self.df_metacoll.fname.astype(str)

        # Merge the metadata with the filenames
        self.df_iter_w_filename = self.df_iterations.merge(
            self.df_filenames, on="sample_id", how="left"
        )
        self.df_iter_w_filename.filename = (
            self.df_iter_w_filename.filename.astype(str)
        )

        # Merge with the metedata
        self.df_ontology_lookup = self.df_iter_w_filename.merge(
            self.df_metacoll, left_on="filename", right_on="fname", how="left"
        )
        self.df_ontology_lookup["mids_first"] = (
            self.df_ontology_lookup.mids.str[0]
        )

        # Calculate
        all_pops_cat = []
        for pop in self.df_ontology_lookup["pop"].unique():
            cats = self.df_ontology_lookup.loc[
                self.df_ontology_lookup["pop"] == pop, "mids_first"
            ]
            all_pops_cat.append(len(set(cats)) / len(cats))

        # plot
        plt.clf()
        plt.plot(all_pops_cat)
        plt.xlabel("Population")
        plt.ylabel("Mean Category Diversity")
        plt.title("Mean category diversity for each population")
        plt.ylim(0, 1)
        plt.savefig(f"output/{self.record_timestamp}/category_diversity.png")
        # plt.show()

    def calculate_coverage_metrics(self):
        print("Calculating coverage metrics...")

        # Calculate dataset coverage
        n_of_unique_files = self.df_ontology_lookup.fname.unique().shape[0]
        n_of_files = self.df_metacoll.fname.astype(str).unique().shape[0]
        self.final_dataset_coverage = n_of_unique_files / n_of_files * 100

        # Calculate category coverage
        n_of_unique_categories = (
            self.df_ontology_lookup["mids_first"].unique().shape[0]
        )
        n_of_categories = (
            self.df_metacoll["mids"]
            .apply(lambda x: x[0] if x is not np.nan else np.nan)
            .astype(str)
            .unique()
            .shape[0]
        )
        self.final_category_coverage = (
            n_of_unique_categories / n_of_categories * 100
        )

    def run(self):
        self.load_data()
        self.calculate_functional_variance()
        self.calculate_phylo_diversity_novelty_index()
        self.calculate_categorical_diversity_novelty_index()
        self.calculate_coverage_metrics()

        print(f"Functional variance plotted")
        print(f"Phylo diversity novelty index: {self.final_pdni}")
        print(f"Dataset coverage: {self.final_dataset_coverage}")
        print(f"Category coverage: {self.final_category_coverage}")


def main():
    list_record_timestamp = [
        "20241104_213011",
    ]
    for record_timestamp in list_record_timestamp:
        evaluation = Evaluation(record_timestamp)
        evaluation.run()


if __name__ == "__main__":
    main()
