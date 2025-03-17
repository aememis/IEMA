import multiprocessing
import concurrent.futures
import json
import os
import re
from datetime import datetime

import config as cfg
from corpus_gauss import CorpusGauss
from corpus_reader import CorpusReader
from evaluation import Evaluation
from operations import Operator, SharedData
from path import Path


def read_run_configs():
    with open("run_configs.json", "r") as file:
        data = json.load(file)
    return data


def update_config(run_config):
    for key, value in run_config.items():
        setattr(cfg, key, value)


def run(session_timestamp, run_id):
    print(f"Running tests for session '{session_timestamp}', run '{run_id}'")

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


def run_with_config(args):
    session_timestamp, run_id, run_config = args
    update_config(run_config)
    run(session_timestamp, run_id)


def run_tests(session_timestamp):
    print("Running tests...")

    # Create output directory
    os.makedirs(f"output/{session_timestamp}")

    # Read run configs
    run_configs = read_run_configs()

    # Save run configs to session directory
    with open(f"output/{session_timestamp}/run_configs.json", "w") as file:
        json.dump(run_configs, file, indent=4)

    # Run configured tests
    # for run_id, run_config in run_configs.items():
    #     update_config(run_config)
    #     run(session_timestamp, run_id)

    arg_list = [
        (session_timestamp, run_id, run_config)
        for run_id, run_config in run_configs.items()
    ]
    with multiprocessing.Pool(processes=len(run_configs)) as pool:
        pool.map(run_with_config, arg_list)


def evaluate(session_timestamp):
    print("Evaluating results...")

    target_dirs = []
    for root, dirs, files in os.walk(f"output\\{session_timestamp}"):
        for target_dir in dirs:
            if re.match(r"path_[0-9]{3}", target_dir):
                target_dirs.append(os.path.join(root, target_dir))

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(Evaluation(target_dir).calculate)
    #         for target_dir in target_dirs
    #     ]
    #     concurrent.futures.wait(futures)

    for target_dir in target_dirs:
        evaluation = Evaluation(target_dir)
        evaluation.calculate()


def main():
    print("Starting...")
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if cfg.GENERATE_PATHS_ON_START:
        Path.get_or_create_paths(source="generate")

    run_tests(session_timestamp)
    evaluate(session_timestamp)

    input("Done!")


if __name__ == "__main__":
    main()
