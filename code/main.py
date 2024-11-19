import json
import os
from datetime import datetime

import config as cfg
from corpus_gauss import CorpusGauss
from corpus_reader import CorpusReader
from evaluation import Evaluation
from operations import Operator, SharedData
from path import Path
import re


def read_run_configs():
    with open("run_configs.json", "r") as file:
        data = json.load(file)
    return data


def run(session_timestamp, run_id):
    paths = Path.get_or_create_paths(source="file")
    assert cfg.NUMBER_OF_PATHS == len(paths), (
        "Number of paths does not match the config. "
        f"The config has '{cfg.NUMBER_OF_PATHS}' paths but "
        f"the file has '{len(paths)}' paths."
    )
    for path_id, path in enumerate(paths, 1):
        sd = SharedData(session_timestamp, run_id, path_id)
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


def run_tests(session_timestamp):
    os.makedirs(f"output/{session_timestamp}")
    run_configs = read_run_configs()
    for run_id, run_config in run_configs.items():
        update_config(run_config)
        run(session_timestamp, run_id)
    print(run_configs)


def evaluate(session_timestamp):
    target_dirs = []
    for root, dirs, files in os.walk(f"output\\{session_timestamp}"):
        for target_dir in dirs:
            if re.match(r"path_[0-9]{3}", target_dir):
                target_dirs.append(os.path.join(root, target_dir))

    for target_dir in target_dirs:
        evaluation = Evaluation(target_dir)
        evaluation.calculate()


def update_config(run_config):
    for key, value in run_config.items():
        setattr(cfg, key, value)


def main():
    print("Starting...")
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tests(session_timestamp)
    evaluate(session_timestamp)
    input("Done!")


if __name__ == "__main__":
    main()
