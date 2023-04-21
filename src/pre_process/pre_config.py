from yacs.config import CfgNode as CN

_C = CN()

# DATA process related configurations.
_C.DATA = CN()

_C.DATA.TRAIN_FILES_2022 = 'data/json/original_data/train-tracks.json'
_C.DATA.TEST_FILES_2022 = 'data/json/original_data/test-queries.json'

_C.DATA.OUTPUT = 'data/dataclean_v1'
_C.DATA.ROOT = 'src/pre_process'

_C.DATA.COLOR_FILENAME = "valid-color_v2.txt"
_C.DATA.VEHICLE_FILENAME = "valid-vehicle_v2.txt"
_C.DATA.REPLACE_FILENAME = "replace_words.txt"
_C.DATA.SIZE_FILENAME = "valid-size.txt"
_C.DATA.DIRECTION_FILENAME = "valid-directions.txt"


def get_default_config():
    return _C.clone()
