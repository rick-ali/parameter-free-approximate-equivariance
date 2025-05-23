# Training hyperparameters
INPUT_SIZE = 784
LATENT_DIM = 32
NUM_CLASSES = 10
LEARNING_RATE = 0.003
BATCH_SIZE = 64
NUM_EPOCHS = 3

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 4
VAL_SPLIT = 0.1

# Compute related
ACCELERATOR = "gpu"
DEVICES = 1
PRECISION = 32