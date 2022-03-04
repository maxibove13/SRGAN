import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"
TRAINING_SET = 'DIV2K'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 16
NUM_WORKERS = 2
HIGH_RES = 96
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3