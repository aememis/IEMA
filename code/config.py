import math

NUMBER_OF_RUNS = 1

NUMBER_OF_ITERATIONS = 50  # 150  # 500
NUMBER_OF_PATHS = 3  # CURRENTLY IN THE CODE, POINTS ARE SELECTED RANDOMLY
PROJECTION_METHOD = "tsne"  # "umap"
CORPUS_METHOD = "gaussian"  # "gaussian"  # "read"
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

NUMBER_OF_CHILDREN = 2
NUMBER_OF_PARENTS = 2

# SELECTION_THRESHOLD_DIVISOR = (
#     100 * POPULATION_SIZE / (81.65 * math.sqrt(POPULATION_SIZE + 0.375) + 50)
# )
SELECTION_THRESHOLD_DIVISOR = 100  # 160  # 80  # temp

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

# additive synth patch
# DATA_FIELDS = ["carrier", "ratio", "metrodev", "att", "sus", "score", "pop"]

# vocal mocap
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
