from datetime import datetime
from Operator import Operator
from Operator import SharedData
from CorpusGauss import CorpusGauss
from CorpusReader import CorpusReader
import config as cfg


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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


if __name__ == "__main__":
    main()
