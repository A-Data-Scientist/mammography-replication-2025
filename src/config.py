"""
Variables set by the command line arguments dictating which parts of the program to execute.
Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
"""
import os
CBIS_ROOT = r"D:/Projects/Mammography/mammography-replication-2025/data/CBIS-DDSM"
# Constants
RANDOM_SEED = 111
MINI_MIAS_IMG_SIZE = {
    "HEIGHT": 1024,
    "WIDTH": 1024
}
VGG_IMG_SIZE = {
    "HEIGHT": 512,
    "WIDTH": 512
}
RESNET_IMG_SIZE = VGG_IMG_SIZE
INCEPTION_IMG_SIZE = VGG_IMG_SIZE
DENSE_NET_IMG_SIZE = VGG_IMG_SIZE
MOBILE_NET_IMG_SIZE = VGG_IMG_SIZE
XCEPTION_IMG_SIZE = INCEPTION_IMG_SIZE
ROI_IMG_SIZE = {
    "HEIGHT": 224,
    "WIDTH": 224
}

# Variables set by command line arguments/flags
dataset = "CBIS-DDSM"       # The dataset to use.
mammogram_type = "all"      # The type of mammogram (Calc or Mass).
model = "VGG"               # The model to use.
run_mode = "training"       # The type of running mode, either training or testing.
learning_rate = 1e-3        # The learning rate with the pre-trained ImageNet layers frozen.
batch_size = 2              # Batch size.
max_epoch_frozen = 100      # Max number of epochs when original CNN layers are frozen.
max_epoch_unfrozen = 50     # Max number of epochs when original CNN layers are unfrozen.
is_roi = False              # Use cropped version of the images
verbose_mode = False        # Boolean used to print additional logs for debugging purposes.
name = ""                   # Name of experiment.
# is_grid_search = False    # Run the grid search algorithm to determine the optimal hyper-parameters for the model.
preprocess = "none"         # Preprocessing: "none" or "clahe"

# Loss: "weighted_ce" or "focal"
loss_type = "weighted_ce"
focal_alpha = 0.25
focal_gamma = 2.0

# Calibration
calibrate = True
calibration_file_suffix = "_temperature.json"

# Add sensible defaults if not present
split_mode = "patient"
train_frac = 0.70
val_frac   = 0.15
test_frac  = 0.15
