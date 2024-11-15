from datetime import datetime

import config as cfg
from corpus_gauss import CorpusGauss
from corpus_reader import CorpusReader
from evaluation import Evaluation
from operations import Operator, SharedData


def run():
    timestamps = []
    for _ in range(cfg.NUMBER_OF_RUNS):
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
        op.operate()
    return timestamps


def evaluate(timestamps):
    for timestamp in timestamps:
        evaluation = Evaluation(timestamp)
        evaluation.run()


def main():
    run_timestamps = run()
    evaluate(run_timestamps)
    print(run_timestamps)
    input("Done!")


if __name__ == "__main__":
    main()
