RUN_IN_PARALLEL = True
PLOT_IN_EVALUATION = False
NUMBER_OF_RUNS = 1

NUMBER_OF_ITERATIONS = 50  # 150  # 500
NUMBER_OF_PATHS = 5
PROJECTION_METHOD = "tsne"  # "tsne", "umap", "isomap"
CORPUS_METHOD = "gaussian"  # "gaussian", "read"
READ_CORPUS_SOURCE = "checkpoint"  # "files", "checkpoint"
SAMPLE_RATE = 22050
POPULATION_SIZE = 200
CORPUS_SIZE = 40000
MUTATION_STRENGTH = 0.05
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8
ELITIST_SELECTION = "false"
AUGMENT_RATIO_CROSSOVER = 0.5
NOTE = "N/A"

K = 5
K_CLOSEST_IN_CORPUS = 1
UNIQUE_CLOSEST_IN_CORPUS = True

NUMBER_OF_CHILDREN = 2
NUMBER_OF_PARENTS = 2

SELECTION_THRESHOLD_DIVISOR = 50

GENERATE_PATHS_ON_START = False
ADAPT_NUMBER_OF_PATHS = True

WHITE = (200, 200, 200)
BLACK = (0, 0, 0)
GREY = (70, 70, 70)

MOCAP_WIDTH_PROJECTION = 3.30
MOCAP_HEIGHT_PROJECTION = 1.65
MOCAP_HEIGHT_Z_PROJECTION = 2.30

WINDOW_HEIGHT_APP = 1080

WINDOW_Y = (
    WINDOW_HEIGHT_APP
    / (MOCAP_HEIGHT_PROJECTION + MOCAP_HEIGHT_Z_PROJECTION)
    * MOCAP_HEIGHT_PROJECTION
)
WINDOW_X = WINDOW_Y / MOCAP_HEIGHT_PROJECTION * MOCAP_WIDTH_PROJECTION
WINDOW_Z = WINDOW_Y / MOCAP_HEIGHT_PROJECTION * MOCAP_HEIGHT_Z_PROJECTION

WINDOW_WIDTH_APP = 1920  # WINDOW_X + WINDOW_Y
FULLSCREEN = True

USER_DOT_SIZE = 6  # Set the size of the grid block
DOT_SIZE = 3
SURR_DOT_SIZE = 5
SURR_DOT_SIZE_2 = 6
MOVE = 5
TARGET_WIDTH_PROJECTION = WINDOW_X  # / 100  # temp
TARGET_HEIGHT_PROJECTION = WINDOW_Y  # / 100  # temp
TARGET_HEIGHT_Z_PROJECTION = WINDOW_Z  # / 100  # temp
TARGET_IP = "localhost"
TARGET_PORT = 8001
NETSEND_DELAY = 0.05
SELECT_DISTANCE_THRESHOLD = 1
NUMBER_OF_IDX_TO_APPLY_SCORE = 1
N_LARGEST_TO_SELECT = 10

USE_SIMULATION = True
SAMPLES_THRESHOLD_LOW = 22050 / 2  # 4
SAMPLES_THRESHOLD_HIGH = 22050 * 20

DATA_FIELDS_VOCALS = [
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "score",
    "pop",
]
DATA_FIELDS_CORPUS = [
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "score",
    "pop",
    "sample_id",
    "id",
]
DATA_FIELDS_CORPUS_LABEL = [
    "rms",
    "spectral_bandwidth",
    "flux",
    "mfcc",
    "sc",
    "sf",
    "score",
    "pop",
    "sample_id",
    "id",
]

DATASET_PATH_ALLIN = (
    "C:\\Users\\emin\\Documents\\mct\\smc24\\data\\allin\\samples\\*.wav"
)
DATASET_PATH_ALLIN2 = (
    "C:\\Users\\emin\\Documents\\mct\\smc24\\data\\allin2\\samples\\*.wav"
)
DATASET_PATH_ALLIN3 = (
    "C:\\Users\\emin\\Documents\\mct\\smc24\\data\\allin3\\mp3\\*.mp3"
)
DATASET_PATH_FSD50K = (
    "D:\\datasets\\FSD50K\\FSD50K.dev_audio_comp\\FSD50K.dev_audio\\*.wav"
)
FSD50K_METADATA_COLLECTION_PATH = (
    r"D:\datasets\FSD50K\FSD50K.metadata\collection\collection_dev.csv"
)
ONTOLOGY_GRAPH_PATH_FSD50K = (
    "analyze_datasets\\fsd50k\\dataset_ontology.gpickle"
)
ONTOLOGY_LAYERS_PATH_FSD50K = "analyze_datasets\\fsd50k\\layers.json"
