import json
from datetime import datetime

import config as cfg
from corpus_gauss import CorpusGauss
from corpus_reader import CorpusReader
from evaluation import Evaluation
from operations import Operator, SharedData
from path import Path


def read_runs_configs():
    with open("runs_config.json", "r") as file:
        data = json.load(file)
    return data


def run():
    timestamps = []
    paths = Path.get_or_create_paths(source="generate")
    for path_id, path in enumerate(paths):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamps.append(timestamp)
        sd = SharedData(timestamp)
        sd.logger.info("Starting...")

        if cfg.CORPUS_METHOD == "gaussian":
            corpus = CorpusGauss(sd)
            corpus.prepare_gauss_features(from_saved=False)
            df_corpus, df_population = corpus.get_as_population()
            df_samples = corpus.get_samples()
        elif cfg.CORPUS_METHOD == "read":
            corpus = CorpusReader(sd)
            corpus.prepare(source="checkpoint", save=False)
            df_corpus, df_population = corpus.get_as_population()
            df_samples = corpus.get_samples()

        op = Operator(sd)
        op.init_from_corpus(df_corpus, df_population, df_samples)
        op.operate(path_id, path)


def evaluate(timestamp):
    evaluation = Evaluation(timestamp)
    evaluation.run()


def update_config(run_config):
    for key, value in run_config.items():
        setattr(cfg, key, value)


def main():
    print("Starting...")
    run_configs = read_runs_configs()
    run_timestamps = []
    for run_id, run_config in run_configs.items():
        update_config(run_config)
        run_timestamp = run()
        run_timestamps.append(run_timestamp)
    for run_timestamp in run_timestamps:
        evaluate(run_timestamp)
    print(run_timestamps)
    input("Done!")


if __name__ == "__main__":
    main()
