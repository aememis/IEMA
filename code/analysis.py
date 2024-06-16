import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform


class Analyzer:
    def __init__(self, record_timestamp):
        self.record_timestamp = record_timestamp
        self.path_population = (
            f"output/{self.record_timestamp}"
            + f"/{self.record_timestamp}_analysis_population.pkl"
        )
        with open(self.path_population, "rb") as f:
            self.df_gens = pickle.load(f)
        self.output_dir = f"output/{self.record_timestamp}"

    def calculate_saturation(self, df_iteration):
        saturation = df_iteration.std().mean()
        return saturation

    def calculate_diversity(self, df_iteration):
        distances = squareform(pdist(df_iteration.values))
        mean_distance = np.mean(distances)
        return mean_distance

    def calculate_entropy(self, df_iteration):
        entropies = []
        for column in df_iteration.columns:
            value_counts = df_iteration[column].value_counts(normalize=True)
            value_counts = value_counts[value_counts > 0]
            entropy = -np.sum(value_counts * np.log2(value_counts))
            entropies.append(entropy)
        mean_entropy = np.mean(entropies)
        return mean_entropy

    def calculate_convergence(self, df_population):
        print("Calculating convergence ...")
        max_iteration = df_population.iteration.max()
        dict_convergence = {}
        for it in sorted(df_population.iteration.unique()):
            print(f"Calculating convergence for iteration: {it}", end="\r")
            if it == max_iteration:
                break
            df_pop1 = df_population[df_population.iteration == it]
            df_pop2 = df_population[df_population.iteration == it + 1]
            pop1_points = df_pop1.iloc[:, :6].to_numpy()
            pop2_points = df_pop2.iloc[:, :6].to_numpy()
            distances = np.linalg.norm(pop1_points[:, np.newaxis] - pop2_points, axis=2)
            mean_distance = np.mean(distances)
            dict_convergence[it] = mean_distance
            print(f"Convergence for iteration {it}: {mean_distance}")
        return dict_convergence

    def calculate_convergence2(self, df_population):
        print("Calculating convergence ...")
        max_iteration = df_population.iteration.max()
        dict_convergence = {}
        for it in sorted(df_population.iteration.unique()):
            print(f"Calculating convergence for iteration: {it}")
            if it == max_iteration:
                break
            df_pop1 = df_population[df_population.iteration == it]
            df_pop2 = df_population[df_population.iteration == it + 1]
            distances = []
            for i1 in df_pop1.index:
                for i2 in df_pop2.index:
                    distance = np.linalg.norm(i1 - i2)
                    distances.append(distance)
            mean_distance = np.mean(distances)
            dict_convergence[it] = mean_distance
        return dict_convergence

    def analyze(self):
        """Measure diversity and saturation for each iteration and path_id. Save
        the results in a DataFrame to be plotted and saved."""
        analysis_results = []
        for iteration in self.df_gens.iteration.unique():
            for path_id in self.df_gens.path_id.unique():
                # print(f"Analyzing iteration: {iteration}")
                df_iteration = self.df_gens[
                    (self.df_gens.iteration == iteration)
                    & (self.df_gens.path_id == path_id)
                ]
                df_iteration = df_iteration[["p1", "p2", "p3", "p4", "p5", "p6"]]
                # df_iteration = df_iteration.drop_duplicates()

                diversity = self.calculate_diversity(df_iteration)
                saturation = self.calculate_saturation(df_iteration)
                entropy = self.calculate_entropy(df_iteration)
                analysis_results.append(
                    [iteration, path_id, diversity, saturation, entropy]
                )

        self.df_result = pd.DataFrame(
            analysis_results,
            columns=["iteration", "path_id", "diversity", "saturation", "entropy"],
        )
        path_results = os.path.join(
            self.output_dir, f"{self.record_timestamp}_results.csv"
        )
        self.df_result.to_csv(path_results, index=False)

        # calculate convergence
        dict_convergence = self.calculate_convergence(self.df_gens)
        # dict_convergence = self.calculate_convergence2(self.df_gens)
        self.df_convergence = pd.DataFrame.from_dict(
            dict_convergence, orient="index", columns=["convergence"]
        )

    def plot(self):
        self.df_plot = self.df_result.copy(deep=True)
        self.df_plot["iteration"] = self.df_plot["iteration"].astype(int)
        self.df_plot["path_id"] = self.df_plot["path_id"].astype(int)

        # plot diversity
        self.df_plot_diversity = self.df_plot.pivot(
            index="iteration",
            columns="path_id",
            values="diversity",
        )
        fig, ax = plt.subplots()
        ax.plot(self.df_plot_diversity, alpha=0.5)
        ax.plot(self.df_plot_diversity.median(axis=1), color="black", linestyle="-")
        plt.title("Diversity")
        plt.xlabel("Iteration")
        plt.ylabel("Diversity")
        plt.ylim(0, 0.75)
        path_plot = os.path.join(
            self.output_dir, f"{self.record_timestamp}_diversity.png"
        )
        plt.savefig(path_plot)
        plt.clf()

        # plot saturation
        self.df_plot_saturation = self.df_plot.pivot(
            index="iteration",
            columns="path_id",
            values="saturation",
        )
        fig, ax = plt.subplots()
        ax.plot(self.df_plot_saturation, alpha=0.5)
        ax.plot(self.df_plot_saturation.median(axis=1), color="black", linestyle="-")
        plt.title("Saturation")
        plt.xlabel("Iteration")
        plt.ylabel("Saturation")
        plt.ylim(0, 0.75)
        path_plot = os.path.join(
            self.output_dir, f"{self.record_timestamp}_saturation.png"
        )
        plt.savefig(path_plot)
        plt.clf()

        # plot entropy
        self.df_plot_entropy = self.df_plot.pivot(
            index="iteration",
            columns="path_id",
            values="entropy",
        )
        fig, ax = plt.subplots()
        ax.plot(self.df_plot_entropy, alpha=0.5)
        ax.plot(self.df_plot_entropy.median(axis=1), color="black", linestyle="-")
        plt.title("Entropy")
        plt.xlabel("Iteration")
        plt.ylabel("Entropy")
        path_plot = os.path.join(
            self.output_dir, f"{self.record_timestamp}_entropy.png"
        )
        plt.savefig(path_plot)
        plt.clf()

        # # plot convergence
        fig, ax = plt.subplots()
        ax.plot(self.df_convergence)
        plt.title("Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Convergence")
        plt.ylim(0, 1)
        path_plot = os.path.join(
            self.output_dir, f"{self.record_timestamp}_convergence.png"
        )
        plt.savefig(path_plot)
        plt.clf()


if __name__ == "__main__":
    record_timestamp = sorted(os.listdir("output"))[-1]
    print(f"Analyzing {record_timestamp} ...")
    analysis = Analyzer(record_timestamp)
    analysis.analyze()
    analysis.plot()
