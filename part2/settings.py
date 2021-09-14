SHAPE = (81, 81, 3)
SIZE_CROP = 41
FILES_USE_COUNT = False  # true means that building dataset from FILE_COUNT amount of files
FILES_START_INDEX = 0  # start scan files from position INDEX with limit of COUNT
FILES_COUNT = 70
TFL_SIGN = b'1'
NOT_TFL_SIGN = b'0'

# files
DATA_DIR_PATH = r'./dataset'
DATA_BIN_FILE_NAME = r'data.bin'
LABELS_BIN_FILE_NAME = r'labels.bin'
JSON_PATH = r'../gtFine/test/*/*polygons.json'
LEFT_IMG_PATH_DIRECTORY = r'../leftImg8bit_trainvaltest/leftImg8bit'
TEST_DIR_NAME = 'test'
TRAIN_DIR_NAME = 'train'
VAL_DIR_NAME = 'val'

# replace name file extension
FILE_EXT_FROM = 'gtFine_polygons.json'
FILE_EXT_TO = 'leftImg8bit.png'

# Train File
LABELS_TO_NAME = {0: 'No TFL', 1: 'Yes TFL'}
