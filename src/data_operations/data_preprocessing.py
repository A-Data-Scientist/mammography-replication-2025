import os

from imutils import paths
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from .patient_splits import group_by_patient, split_patients
import config


def import_minimias_dataset(data_dir: str, label_encoder) -> (np.ndarray, np.ndarray):
    """
    Import the dataset by pre-processing the images and encoding the labels.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param data_dir: Directory to the mini-MIAS images.
    :param label_encoder: The label encoder.
    :return: Two NumPy arrays, one for the processed images and one for the encoded labels.
    """
    # Initialise variables.
    images = list()
    labels = list()

    if not config.is_roi:
        # Loop over the image paths and update the data and labels lists with the pre-processed images & labels.
        print("Loading whole images")
        for image_path in list(paths.list_images(data_dir)):
            images.append(preprocess_image(image_path))
            labels.append(image_path.split(os.path.sep)[-2])  # Extract label from path.
    else:
        # Use the CSV file to get the images and their labels, and crop the images around the specified ROI.
        print("Loading cropped ROI images")
        images, labels = crop_roi_image(data_dir)

    # Convert the data and labels lists to NumPy arrays.
    images = np.array(images, dtype="float32")  # Convert images to a batch.
    labels = np.array(labels)

    # Encode labels.
    labels = encode_labels(labels, label_encoder)

    return images, labels


def import_cbisddsm_training_dataset(label_encoder):
    """
    Import the dataset getting the image paths (downloaded on BigTMP) and encoding the labels.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
    print("Importing CBIS-DDSM training set")
    cbis_ddsm_path = str()
    if config.mammogram_type == "calc":
        cbis_ddsm_path = "../data/CBIS-DDSM/calc-training.csv"
    elif config.mammogram_type == "mass":
        cbis_ddsm_path = "../data/CBIS-DDSM/mass-training.csv"
    else:
        cbis_ddsm_path = "../data/CBIS-DDSM/training.csv"
    df = pd.read_csv(cbis_ddsm_path)
    list_IDs = df['img_path'].values
    labels = encode_labels(df['label'].values, label_encoder)
    return list_IDs, labels

def import_cbisddsm_patientwise_datasets(label_encoder, train=0.70, val=0.15, test=0.15, seed=42):
    """
    Build patient-wise, view-aware train/val/test from CBIS-DDSM CSVs.
    Returns:
      X_train, y_train, X_val, y_val, X_test, y_test  (paths and integer labels)
    Behavior:
      - Combines training.csv and testing.csv to create a single patient pool,
        then splits by patient (leakage-safe).
      - Writes the split manifests to ../data/CBIS-DDSM/splits/patientwise/{train,val,test}.csv
    """

    # Pick CSVs based on mammogram type
    if config.mammogram_type == "calc":
        tr = "../data/CBIS-DDSM/calc-training.csv"
        te = "../data/CBIS-DDSM/calc-test.csv"
    elif config.mammogram_type == "mass":
        tr = "../data/CBIS-DDSM/mass-training.csv"
        te = "../data/CBIS-DDSM/mass-test.csv"
    else:
        tr = "../data/CBIS-DDSM/training.csv"
        te = "../data/CBIS-DDSM/testing.csv"

    df_tr = pd.read_csv(tr)
    df_te = pd.read_csv(te)

    # Combine to form one pool (leakage-safe split across whole dataset)
    df_all = pd.concat([df_tr, df_te], axis=0, ignore_index=True)

    # Expect columns: img_path, label
    paths = df_all["img_path"].astype(str).tolist()
    raw_labels = df_all["label"].values
    # encode to ints (0/1) consistently
    y = encode_labels(raw_labels, label_encoder)

    # Group by patient
    pid_to_idx, pid_major_label = group_by_patient(paths, y)
    pid_list = list(pid_to_idx.keys())

    # Split patients
    p_train, p_val, p_test = split_patients(pid_list, pid_major_label, train=train, val=val, test=test, seed=seed)

    # Flatten indices per split
    def flatten_idxs(pid_set):
        idxs = []
        for pid in pid_set:
            idxs.extend(pid_to_idx[pid])
        return sorted(idxs)

    idx_train = flatten_idxs(p_train)
    idx_val   = flatten_idxs(p_val)
    idx_test  = flatten_idxs(p_test)

    X_train = [paths[i] for i in idx_train]
    y_train = [int(y[i]) for i in idx_train]
    X_val   = [paths[i] for i in idx_val]
    y_val   = [int(y[i]) for i in idx_val]
    X_test  = [paths[i] for i in idx_test]
    y_test  = [int(y[i]) for i in idx_test]

    # Save manifests for reproducibility
    out_dir = "../data/CBIS-DDSM/splits/patientwise"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"img_path": X_train, "label": y_train}).to_csv(os.path.join(out_dir, "train.csv"), index=False)
    pd.DataFrame({"img_path": X_val,   "label": y_val}).to_csv(os.path.join(out_dir, "val.csv"),   index=False)
    pd.DataFrame({"img_path": X_test,  "label": y_test}).to_csv(os.path.join(out_dir, "test.csv"),  index=False)

    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)

def import_cbisddsm_testing_dataset(label_encoder):
    """
    Import the testing dataset getting the image paths (downloaded on BigTMP) and encoding the labels.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
    print("Importing CBIS-DDSM testing set")
    cbis_ddsm_path = str()
    if config.mammogram_type == "calc":
        cbis_ddsm_path = "../data/CBIS-DDSM/calc-test.csv"
    elif config.mammogram_type == "mass":
        cbis_ddsm_path = "../data/CBIS-DDSM/mass-test.csv"
    else:
        cbis_ddsm_path = "../data/CBIS-DDSM/testing.csv"
    df = pd.read_csv(cbis_ddsm_path)
    list_IDs = df['img_path'].values
    labels = encode_labels(df['label'].values, label_encoder)
    return list_IDs, labels


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Pre-processing steps:
        * Load the input image in grayscale mode (1 channel),
        * resize it to fit the CNN model input,
        * transform it to an array format,
        * normalise the pixel intensities.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param image_path: The path to the image to preprocess.
    :return: The pre-processed image in NumPy array format.
    """
    # Resize if using full image.
    if not config.is_roi:
        if config.model == "VGG" or config.model == "Inception":
            target_size = (config.MINI_MIAS_IMG_SIZE['HEIGHT'], config.MINI_MIAS_IMG_SIZE["WIDTH"])
        elif config.model == "VGG-common":
            target_size = (config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE["WIDTH"])
        elif config.model == "MobileNet":
            target_size = (config.MOBILE_NET_IMG_SIZE['HEIGHT'], config.MOBILE_NET_IMG_SIZE["WIDTH"])
        elif config.model == "CNN":
            target_size = (config.ROI_IMG_SIZE['HEIGHT'], config.ROI_IMG_SIZE["WIDTH"])
        image = load_img(image_path, color_mode="grayscale", target_size=target_size)

    # Do not resize if using cropped ROI image.
    else:
        image = load_img(image_path, color_mode="grayscale")

    image = img_to_array(image)
    image /= 255.0
    return image


def encode_labels(labels_list: np.ndarray, label_encoder) -> np.ndarray:
    """
    Encode labels using one-hot encoding.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param label_encoder: The label encoder.
    :param labels_list: The list of labels in NumPy array format.
    :return: The encoded list of labels in NumPy array format.
    """
    labels = label_encoder.fit_transform(labels_list)
    if label_encoder.classes_.size == 2:
        return labels
    else:
        return to_categorical(labels)


def dataset_stratified_split(split: float, dataset: np.ndarray, labels: np.ndarray) -> \
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Partition the data into training and testing splits. Stratify the split to keep the same class distribution in both
    sets and shuffle the order to avoid having imbalanced splits.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param split: Dataset split (e.g. if 0.2 is passed, then the dataset is split in 80%/20%).
    :param dataset: The dataset of pre-processed images.
    :param labels: The list of labels.
    :return: the training and testing sets split in input (X) and label (Y).
    """
    train_X, test_X, train_Y, test_Y = train_test_split(dataset,
                                                        labels,
                                                        test_size=split,
                                                        stratify=labels,
                                                        random_state=config.RANDOM_SEED,
                                                        shuffle=True)
    return train_X, test_X, train_Y, test_Y





def calculate_class_weights(y, label_encoder=None):
    y_np = np.asarray(y).ravel()
    classes = np.unique(y_np)
    # keyword-only args work on all recent sklearn versions
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_np)
    return {int(c): float(wi) for c, wi in zip(classes, w)}


# def calculate_class_weights(y_train, label_encoder):
#     """
#     Calculate class  weights for imbalanced datasets.
#     """
#     if label_encoder.classes_.size != 2:
#         y_train = label_encoder.inverse_transform(np.argmax(y_train, axis=1))

#     # Balanced class weights
#     weights = class_weight.compute_class_weight("balanced",
#                                                 np.unique(y_train),
#                                                 y_train)
#     class_weights = dict(enumerate(weights))

#     # Manual class weights for CBIS-DDSM
#     #class_weights = {0: 1.0, 1:1.5}

#     # No class weights
#     #class_weights = None

#     if config.verbose_mode:
#         print("Class weights: {}".format(str(class_weights)))

#     # return class_weights
#     return None

def make_patientwise_splits(images, labels,
                            train=0.70, val=0.15, test=0.15, seed=42):
    groups = group_by_patient(images, labels)
    return split_patients(groups, train=train, val=val, test=test, seed=seed, stratify_on_label=True)


def crop_roi_image(data_dir):
    """
    Crops the images from the mini-MIAS dataset.
    Function originally written by Shuen-Jen and amended by Adam Jaamour.
    """
    images = list()
    labels = list()

    csv_dir = data_dir
    images_dir = data_dir.split("_")[0] + "_png"

    df = pd.read_csv('/'.join(csv_dir.split('/')[:-1]) + '/data_description.csv', header=None)

    for row in df.iterrows():
        # Skip normal cases.
        if str(row[1][4]) == 'nan':
            continue
        if str(row[1][4]) == '*NOT':
            continue

        # Process image.
        image = preprocess_image(images_dir + '/' + row[1][0] + '.png')

        # Abnormal case: crop around tumour.
        y2 = 0
        x2 = 0
        if row[1][2] != 'NORM':
            y1 = image.shape[1] - int(row[1][5]) - 112
            if y1 < 0:
                y1 = 0
                y2 = 224
            if y2 != 224:
                y2 = image.shape[1] - int(row[1][5]) + 112
                if y2 > image.shape[1]:
                    y2 = image.shape[1]
                    y1 = image.shape[1] - 224
            x1 = int(row[1][4]) - 112
            if x1 < 0:
                x1 = 0
                x2 = 224
            if x2 != 224:
                x2 = int(row[1][4]) + 112
                if x2 > image.shape[0]:
                    x2 = image.shape[0]
                    x1 = image.shape[0] - 224

        # Normal case: crop around centre of image.
        else:
            y1 = int(image.shape[1] / 2 - 112)
            y2 = int(image.shape[1] / 2 + 112)
            x1 = int(image.shape[0] / 2 - 112)
            x2 = int(image.shape[0] / 2 + 112)

        # Get label from CSV file.
        label = "normal"
        if str(row[1][3]) == 'B':
            label = "benign"
        elif str(row[1][3]) == 'M':
            label = "malignant"

        # Append image and label to lists.
        images.append(image[y1:y2, x1:x2, :])
        labels.append(label)

    return images, labels
