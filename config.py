#-*- coding:utf-8 -*-

ENABLE_GPU = True

MODEL_PATH1 = "models"
MODEL_PATH2 = "/work/machine_learning/prisma_style/open-source-proj/gafr/chainer-fast-neuralstyle-models/models"
MODEL_PATH3 = "/work/machine_learning/prisma_style/open-source-proj/yusuketomoto/chainer-fast-neuralstyle/models"
MODEL_PATH_LIST = [MODEL_PATH1, MODEL_PATH2, MODEL_PATH3]


FILE_UPLOAD_PATH = "uploads"
OPT_FOLDER = "opt"
THUMB_FOLDR = "thumb_opt"

PRE_TRAINED_MODELS = [
    "model_trained_by_kevin",
    "composition", "seurat",
    "candy_512_2_49000", "fur_0", "kanagawa", "scream-style",
    "cubist", "hokusai", "kandinsky_e2_crop512", "starry",
    "edtaonisl", "hundertwasser", "kandinsky_e2_full512", "starrynight",
    "brad_1", "600_271_0", "4_600_274", "4_600_274_0", "4_600_274_1"]


DOWN_SCALE = 0.03
DOWN_SCALE_COUNT = 5

MODE_STATIC_IMAGE = 1
MODE_STATIC_ANIM_IMAGE = 2

RET_TIME = "time"
RET_MODE = "mode"
RET_OPT_VIDEO = "video"
RET_OPT_GIF = "gif"
RET_OPT_FILENAME = "file"
RET_OPT_FILENAME_LIST = "file_list"
RET_RESOLUTION = "resolution"
RET_MODEL = "model"

MAX_EDGE = 2048
MAX_EDGE_ANIM = 720

LOG_FODLER = "log"
LOG_FILE_GENERAL = "log.log"
LOG_FILE_STRESS = "stress.log"
