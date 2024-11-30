import json
import os
import pickle
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from dataset import FSD50K


class Evaluation:
    def __init__(self, target_dir):
        self.target_dir = target_dir
        self.evaluation_results = {}
        self.dataset = FSD50K()

    def load_data(self):
        print("Loading data...")

        # load the audio features
        with open("features.pkl", "rb") as f:  # gaussian_features_norm.pkl
            df_features = pickle.load(f)
        self.nd_features = df_features.values

        # load the evolution history
        with open(
            f"{self.target_dir}/analysis_population.pkl",
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

        self.evaluation_results["functional_variance"] = self.all_pops_var

        # plot
        plt.clf()
        plt.plot(self.all_pops_var)
        plt.xlabel("Population")
        plt.ylabel("Variance")
        plt.title("Variance of each population")
        plt.ylim(0, 0.2)
        plt.savefig(f"{self.target_dir}/functional_variance.png")
        # plt.show()

    def load_phylo_data(self):
        # load the graph to nx
        with open(
            f"{self.target_dir}/analysis_evo_graph.gpickle",
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
        print("\tCalculating pairwise distances...")
        all_mpd_pops = []
        all_pops_unique = (
            pd.Series([node[1]["pop"] for node in self.G_plot.nodes(data=True)])
            .sort_values()
            .unique()
        )

        # calculate for the last population, final version of the tree
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

            pairwise_distances_values = np.array(
                list(pairwise_distances.values())
            )

            mpd_pop = (
                pairwise_distances_values.sum()
                * 2
                / (len(leaf_nodes) * (len(leaf_nodes) - 1))
            )
            all_mpd_pops.append(mpd_pop)

        return all_mpd_pops[-1]  # for the final version of the tree

    def calculate_root_contribution_index(self):
        print("\tCalculating root contribution index...")
        # get the roots based on the condition that they have no incoming edges
        roots = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]

        # calculate the number of leaves that they lead to, for each root
        root_contributions = []
        for root in roots:
            descs = nx.descendants(self.G, root)
            if len(descs) == 0:
                root_contributions.append(0)
                continue
            # leaf_descs = [n for n in descs if G.nodes[n]["pop"] == max_pop]
            root_age = self.max_pop - self.G.nodes[root]["pop"]
            contribution = len(descs) / root_age
            root_contributions.append(contribution)

        root_contributions = np.array(root_contributions)

        return root_contributions.sum() / self.G.number_of_nodes()

    def calculate_phylo_diversity_novelty_index(self):
        print("Calculating phylogenetic diversity and novelty index...")

        # get max population
        self.max_pop = self.df_iterations["pop"].max()

        self.final_rci = self.calculate_root_contribution_index()
        self.final_mpd = self.calculate_pairwise_distances()
        self.final_pdni = self.final_mpd / self.final_rci

        self.evaluation_results["phylo_diversity_novelty_index"] = {
            "phylo_diversity_novelty_index": self.final_pdni,
            "root_contribution_index": self.final_rci,
            "mean_pairwise_distance": self.final_mpd,
        }

        # plot single point
        plt.clf()
        plt.scatter(0, self.final_rci, label="Root Contribution Index")
        plt.scatter(0, self.final_mpd, label="Mean Pairwise Distance")
        plt.scatter(
            0,
            self.final_pdni,
            label="Phylogenetic Diversity-Novelty Index",
            c="r",
            s=100,
        )
        plt.title("Phylogenetic Diversity Novelty Index")
        plt.grid(axis="y", color="grey", linestyle="--", linewidth=0.5)
        for i, txt in enumerate(
            [self.final_rci, self.final_mpd, self.final_pdni]
        ):
            plt.annotate(
                f"{txt:.2f}",
                (0, txt),
                textcoords="offset points",
                xytext=(20, 0),
                ha="center",
                va="center",
            )
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(
            f"{self.target_dir}/phylo_diversity_novelty_index.png",
            bbox_inches="tight",
        )
        # plt.show()

    def category_change_rate(self):
        """The change in categories from the
        previous generation to the current one. Calculated by summing the dis-
        tances between the categories of individuals and their parents, and av-
        eraged across all individuals in the current generation.Indicates
        whether the algorithm is exploring new categories or sticking to
        established ones. Calculated for each generation
        """
        print("\tCalculating category change rate...")
        all_pops_ccr = [None]
        for pop in sorted(self.df_iterations["pop"].unique()):
            if pop == 0 or pop == 1 or pop == 2:
                #### /\ FIX THIS, pop should start from 1
                continue
            df_gen = self.df_iterations.loc[
                self.df_iterations["pop"] == pop, ["id", "sample_id"]
            ]
            df_prev_gen = self.df_iterations.loc[
                self.df_iterations["pop"] == pop - 1, ["id", "sample_id"]
            ]
            pop_ccr = []
            for i, row in df_gen.iterrows():
                node_id = row["id"]
                node_sample_id = row["sample_id"]
                node_cats = (
                    self.df_ontology_lookup.loc[
                        self.df_ontology_lookup["sample_id"] == node_sample_id,
                        "mids",
                    ]
                    .explode()
                    .unique()
                )

                assert pop == self.G.nodes()[node_id]["pop"], "pop mismatch"

                parents = list(self.G_plot.predecessors(node_id))
                parents = [p for p in parents if p in df_prev_gen["id"].values]
                if len(parents) == 0:
                    continue

                parents_sample_ids = df_prev_gen.loc[
                    df_prev_gen["id"].isin(parents), "sample_id"
                ].values
                parent_cats = (
                    self.df_ontology_lookup.loc[
                        self.df_ontology_lookup["sample_id"].isin(
                            parents_sample_ids
                        ),
                        "mids",
                    ]
                    .explode()
                    .unique()
                )
                distances = self.dataset.get_distances(node_cats, parent_cats)

                if len(distances) == 0:
                    raise ValueError(
                        "Distances between dataset categories "
                        "could not be calculated"
                    )

                ccr = np.mean(distances)
                pop_ccr.append(ccr)

            if len(pop_ccr) == 0:
                input("pop_ccr is empty")

            mean_pop_ccr = np.mean(pop_ccr)
            all_pops_ccr.append(mean_pop_ccr)

        return all_pops_ccr

    def categorical_diversity(self):
        print("\tCalculating categorical diversity...")

        # Calculate the categorical diversity
        all_pops_cd = []
        for pop in self.df_ontology_lookup["pop"].unique():
            cats = self.df_ontology_lookup.loc[
                self.df_ontology_lookup["pop"] == pop, "mids"
            ]
            len_pop = len(cats)
            cats_unique = cats.explode().unique()
            cd = len(cats_unique) / len_pop
            all_pops_cd.append(cd)

        # plot
        plt.clf()
        plt.plot(all_pops_cd)
        plt.xlabel("Population")
        plt.ylabel("Mean Category Diversity")
        plt.title("Mean category diversity for each population")
        plt.ylim(0, 1)
        plt.savefig(f"{self.target_dir}/category_diversity.png")
        # plt.show()

        return all_pops_cd

    def load_ontology_data(self):
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

    def calculate_categorical_diversity_novelty_index(self):
        print("Calculating categorical diversity novelty index...")

        self.all_pops_cd = self.categorical_diversity()
        self.all_pops_ccr = self.category_change_rate()

        self.evaluation_results["categorical_diversity_novelty_index"] = {
            "categorical_diversity_novelty_index": None,
            "categorical_diversity": self.all_pops_cd,
            "category_change_rate": self.all_pops_ccr,
        }

    def calculate_coverage_metrics(self):
        print("Calculating coverage metrics...")

        # Calculate dataset coverage
        n_of_unique_files = self.df_ontology_lookup.fname.unique().shape[0]
        n_of_files = self.df_metacoll.fname.astype(str).unique().shape[0]
        self.final_dataset_coverage = n_of_unique_files / n_of_files * 100

        self.evaluation_results["dataset_coverage"] = (
            self.final_dataset_coverage
        )

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

        self.evaluation_results["category_coverage"] = (
            self.final_category_coverage
        )

        # plot two points with labels
        plt.clf()
        plt.scatter(0, self.final_dataset_coverage, label="Dataset Coverage")
        plt.scatter(0, self.final_category_coverage, label="Category Coverage")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title("Coverage Metrics")
        plt.grid(axis="y", color="grey", linestyle="--", linewidth=0.5)
        for i, txt in enumerate(
            [self.final_dataset_coverage, self.final_category_coverage]
        ):
            plt.annotate(
                f"{txt:.2f}",
                (0, txt),
                textcoords="offset points",
                xytext=(20, 0),
                ha="center",
                va="center",
            )
        plt.savefig(
            f"{self.target_dir}/data_and_category_coverage.png",
            bbox_inches="tight",
        )
        # plt.show()

    def calculate(self):
        print(f"Running: {self.target_dir}")

        self.load_data()

        self.calculate_functional_variance()

        self.load_phylo_data()
        self.calculate_phylo_diversity_novelty_index()

        self.load_ontology_data()
        self.calculate_categorical_diversity_novelty_index()

        self.calculate_coverage_metrics()

        # save the results
        with open(
            f"{self.target_dir}/evaluation_results.pkl",
            "wb",
        ) as f:
            pickle.dump(self.evaluation_results, f)

        # save as readable json
        with open(
            f"{self.target_dir}/evaluation_results.json",
            "w",
        ) as f:
            json.dump(self.evaluation_results, f)

        print("Done evaluation.")
        print("--------------------------------------\n")


# end of class


def main():
    print("Starting evaluation...")
    session_timestamp = "20241117_202424"  # test
    target_dirs = []
    for root, dirs, files in os.walk(f"output\\{session_timestamp}"):
        for target_dir in dirs:
            if re.match(r"path_[0-9]{3}", target_dir):
                target_dirs.append(os.path.join(root, target_dir))

    for target_dir in target_dirs:
        evaluation = Evaluation(target_dir)
        evaluation.calculate()

    print(session_timestamp)
    print("Done evaluation!")


if __name__ == "__main__":
    main()
