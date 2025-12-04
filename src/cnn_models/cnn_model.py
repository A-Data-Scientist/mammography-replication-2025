import json
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, make_scorer
from sklearn.preprocessing import LabelEncoder
from calibration.temperature import (
    fit_temperature_binary, apply_temperature_binary,
    save_temperature, load_temperature
)
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import (
    BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
)
from tensorflow.keras.metrics import (
    BinaryAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers as tfko
import config
from cnn_models.basic_cnn import create_basic_cnn_model
from cnn_models.densenet121 import create_densenet121_model
from cnn_models.inceptionv3 import create_inceptionv3_model
from cnn_models.mobilenet_v2 import create_mobilenet_model
from cnn_models.resnet50 import create_resnet50_model
from cnn_models.vgg19 import create_vgg19_model
from cnn_models.vgg19_common import create_vgg19_model_common
from data_visualisation.csv_report import *
from data_visualisation.plots import *
from data_visualisation.roc_curves import *



class CnnModel:

    def __init__(self, model_name: str, num_classes: int):
        """
        Function to create instantiate a CNN model containing a pre-trained CNN architecture with custom convolution
        layers at the top and fully connected layers at the end.
        :param model_name: The name of the CNN model to use.
        :param num_classes: The number of classes (labels).
        :return: None.
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.history = None
        self.prediction = None

        # Create model.
        if self.model_name == "VGG":
            self._model = create_vgg19_model(self.num_classes)
        elif self.model_name == "VGG-common":
            self._model = create_vgg19_model_common(self.num_classes)
        elif self.model_name == "ResNet":
            self._model = create_resnet50_model(self.num_classes)
        elif self.model_name == "Inception":
            self._model = create_inceptionv3_model(self.num_classes)
        elif self.model_name == "DenseNet":
            self._model = create_densenet121_model(self.num_classes)
        elif self.model_name == "MobileNet":
            self._model = create_mobilenet_model(self.num_classes)
        elif self.model_name == "CNN":
            self._model = create_basic_cnn_model(self.num_classes)
        # Where to save artifacts (model, weights, temperature, etc.)
        self.models_dir = getattr(config, "models_dir", os.path.join(os.getcwd(), "models"))
        os.makedirs(self.models_dir, exist_ok=True)
        # Canonical stem used by everyone (train/test)
        self.artifact_stem = os.path.join(self.models_dir, f"{self.model_name}_{getattr(config, 'dataset', 'unknown')}")
        self.run_tag = (getattr(config, "name", "") or
                f"{config.dataset}_{config.model}_"
                f"{getattr(config,'split_mode','patient')}_"
                f"{getattr(config,'preprocess','none')}_"
                f"{getattr(config,'loss_type','weighted_ce')}"
                f"{'_cal' if getattr(config,'calibrate', False) else ''}")
    
    def get_run_tag(self): return (self.run_tag + "_") if self.run_tag else ""

    def train_model(self, X_train, X_val, y_train, y_val, class_weights) -> None:
        """
        Function to train network in two steps:
            * Train network with initial pre-trained CNN's layers frozen.
            * Unfreeze all layers and retrain with smaller learning rate.
        Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
        :param X_train: training input
        :param X_val: training outputs
        :param y_train: validation inputs
        :param y_val: validation outputs
        :param class_weights: dict containing class weights
        :return: None
        """
        # Training with transfer learning.
        if not self.model_name == "CNN":

            # Freeze pre-trained CNN model layers: only train fully connected layers.
            layer_name = str()
            if self.model_name == "VGG":
                self._model.layers[1].trainable = False
                layer_name = self._model.layers[1].name
            elif self.model_name == "VGG-common" or self.model_name == "Inception" or self.model_name == "ResNet":
                self._model.layers[0].trainable = False
                layer_name = self._model.layers[0].name
            if config.verbose_mode:
                print("Freezing '{}' layers".format(layer_name))

            # Train model with frozen layers (all training with early stopping dictated by loss in validation over 3 runs).
            self.compile_model(config.learning_rate)
            self.fit_model(X_train, X_val, y_train, y_val, class_weights, is_frozen_layers=True)

            # Plot the training loss and accuracy.
            plot_training_results(self.history, "Initial_training", is_frozen_layers=True, prefix=self.get_run_tag())

            # Unfreeze all layers.
            if self.model_name == "VGG":
                self._model.layers[1].trainable = True
            elif self.model_name == "VGG-common" or self.model_name == "Inception" or self.model_name == "ResNet":
                self._model.layers[0].trainable = True
            if config.verbose_mode:
                print("Unfreezing '{}' layers (all layers now unfrozen)".format(layer_name))

            # Train a second time with a smaller learning rate (train over fewer epochs to prevent over-fitting).
            self.compile_model(1e-5)  # Very low learning rate.
            self.fit_model(X_train, X_val, y_train, y_val, class_weights, is_frozen_layers=False)

            # Plot the training loss and accuracy.
            plot_training_results(self.history, "Fine_tuning_training", is_frozen_layers=False, prefix=self.get_run_tag())

        # Small CNN (no transfer learning).
        else:
            self.compile_model(config.learning_rate)
            self.fit_model(X_train, X_val, y_train, y_val, class_weights, is_frozen_layers=True)
            plot_training_results(self.history, "Initial_training",  is_frozen_layers=True,  prefix=self.get_run_tag())

    
    def compile_model(self, learning_rate) -> None:
        # choose optimizer (you can keep adam and set LR after compile as you do)
        opt = "adam"

        if config.dataset in ("CBIS-DDSM", "mini-MIAS-binary"):
            # choose loss based on ablation setting
            loss_type = getattr(config, "loss_type", "weighted_ce")  # "weighted_ce" or "focal"
            if loss_type == "focal":
                alpha = float(getattr(config, "focal_alpha", 0.25))
                gamma = float(getattr(config, "focal_gamma", 2.0))
                print("[debug] focal alpha =", alpha, "gamma =", gamma)
                loss_fn = make_binary_focal_loss(alpha=alpha, gamma=gamma)
            else:
                loss_fn = BinaryCrossentropy()

            self._model.compile(
                optimizer=opt,
                loss=loss_fn,
                metrics=[BinaryAccuracy()]
            )

        elif config.dataset == "mini-MIAS":
            # multiclass path unchanged
            self._model.compile(
                optimizer=opt,
                loss=CategoricalCrossentropy(),
                metrics=[CategoricalAccuracy()]
            )

        # set LR after compile (keeps your existing behavior)
        tf.keras.backend.set_value(self._model.optimizer.learning_rate, float(learning_rate))

    
    
    
    
    # def compile_model(self, learning_rate) -> None:
    #     if config.dataset in ("CBIS-DDSM", "mini-MIAS-binary"):
    #         self._model.compile(optimizer="adam",
    #                             loss=BinaryCrossentropy(),
    #                             metrics=[BinaryAccuracy()])
    #     elif config.dataset == "mini-MIAS":
    #         self._model.compile(optimizer="adam",
    #                             loss=CategoricalCrossentropy(),
    #                             metrics=[CategoricalAccuracy()])

    #     # set LR after compile so we don't pass any foreign optimizer object
    #     tf.keras.backend.set_value(self._model.optimizer.learning_rate, float(learning_rate))


    # def compile_model(self, learning_rate) -> None:
    #     """
    #     Compile the CNN model.
    #     Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    #     :param learning_rate: The initial learning rate for the optimiser.
    #     :return: None
    #     """
    #     if config.dataset == "CBIS-DDSM" or config.dataset == "mini-MIAS-binary":
    #         self._model.compile(optimizer=Adam(learning_rate),
    #                             loss=BinaryCrossentropy(),
    #                             metrics=[BinaryAccuracy()])
    #     elif config.dataset == "mini-MIAS":
    #         self._model.compile(optimizer=Adam(learning_rate),
    #                             loss=CategoricalCrossentropy(),
    #                             metrics=[CategoricalAccuracy()])

    def fit_model(self, X_train, X_val, y_train, y_val, class_weights, is_frozen_layers: bool) -> None:
        """
        Fit the CNN model and plot the training evolution.
        Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
        :param X_train: training input
        :param X_val: training outputs
        :param y_train: validation inputs
        :param y_val: validation outputs
        :param class_weights: dict containing class weights
        :param is_frozen_layers: boolean specifying whether layers are frozen or not
        :return: None
        """
        if is_frozen_layers:
            max_epochs = config.max_epoch_frozen
            patience = int(config.max_epoch_frozen / 10)
        else:
            max_epochs = config.max_epoch_unfrozen
            patience = int(config.max_epoch_unfrozen / 10)

        if config.dataset == "mini-MIAS":
            self.history = self._model.fit(
                x=X_train,
                y=y_train,
                # class_weight=class_weights,
                batch_size=config.batch_size,
                steps_per_epoch=len(X_train) // config.batch_size,
                validation_data=(X_val, y_val),
                validation_steps=len(X_val) // config.batch_size,
                epochs=max_epochs,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                    ReduceLROnPlateau(patience=int(patience / 2))
                ]
            )
        elif config.dataset == "mini-MIAS-binary":
            self.history = self._model.fit(
                x=X_train,
                y=y_train,
                batch_size=config.batch_size,
                steps_per_epoch=len(X_train) // config.batch_size,
                validation_data=(X_val, y_val),
                validation_steps=len(X_val) // config.batch_size,
                epochs=max_epochs,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                    ReduceLROnPlateau(patience=int(patience / 2))
                ]
            )
        elif config.dataset == "CBIS-DDSM":
            self.history = self._model.fit(
                x=X_train,
                validation_data=X_val,
                class_weight=class_weights,
                epochs=max_epochs,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                    ReduceLROnPlateau(patience=int(patience / 2))
                ]
            )

    def make_prediction(self, x):
        """
        Makes a prediction using unseen data.
        Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
        :param x: The input.
        :return: The model predictions (labels, not probabilities).
        """
        if config.dataset == "mini-MIAS" or config.dataset == "mini-MIAS-binary":
            self.prediction = self._model.predict(x=x.astype("float32"), batch_size=config.batch_size)
        elif config.dataset == "CBIS-DDSM":
            self.prediction = self._model.predict(x=x)
        # print(self.prediction)

    def evaluate_model(self, y_true: list, label_encoder: LabelEncoder, classification_type: str, runtime) -> None:
        """
        Evaluate model performance with accuracy, confusion matrix, ROC curve and compare with other papers' results.
        Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
        :param y_true: Ground truth of the data in one-hot-encoding type.
        :param label_encoder: The label encoder for y value (label).
        :param classification_type: The classification type. Ex: N-B-M: normal, benign and malignant; B-M: benign and
        malignant.
        :param runtime: Runtime in seconds.
        :return: None.
        """
        # Inverse transform y_true and y_pred from one-hot-encoding to original label.
        if label_encoder.classes_.size == 2:
            y_true_inv = y_true
            y_pred_inv = np.round_(self.prediction, 0)
        else:
            y_true_inv = label_encoder.inverse_transform(np.argmax(y_true, axis=1))
            y_pred_inv = label_encoder.inverse_transform(np.argmax(self.prediction, axis=1))

        # Calculate accuracy.
        accuracy = float('{:.4f}'.format(accuracy_score(y_true_inv, y_pred_inv)))
        print("Accuracy = {}\n".format(accuracy))

        prefix = self.get_run_tag()
        
        # Generate CSV report.
        generate_csv_report(y_true_inv, y_pred_inv, label_encoder, accuracy)
        generate_csv_metadata(runtime)

        # Plot confusion matrix and normalised confusion matrix.
        cm = confusion_matrix(y_true_inv, y_pred_inv)  # Calculate CM with original label of classes
        plot_confusion_matrix(cm, 'd', label_encoder, False, prefix=prefix)
        # Calculate normalized confusion matrix with original label of classes.
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized[np.isnan(cm_normalized)] = 0
        plot_confusion_matrix(cm_normalized, '.2f', label_encoder, True, prefix=prefix)

        # Plot ROC curve.
        if label_encoder.classes_.size == 2:  # binary classification
            plot_roc_curve_binary(y_true, self.prediction, prefix=prefix)
        elif label_encoder.classes_.size >= 2:  # multi classification
            plot_roc_curve_multiclass(y_true, self.prediction, label_encoder, prefix=prefix)

        # Compare results with other similar papers' result.
        with open(
                'data_visualisation/other_paper_results.json') as config_file:  # Load other papers' results from JSON.
            data = json.load(config_file)

        dataset_key = config.dataset
        if config.dataset == "mini-MIAS-binary":
            dataset_key = "mini-MIAS"

        df = pd.DataFrame.from_records(data[dataset_key][classification_type],
                                       columns=['paper', 'accuracy'])  # Filter data by dataset and classification type.
        new_row = pd.DataFrame({'paper': 'Dissertation', 'accuracy': accuracy},
                               index=[0])  # Add model result into dataframe to compare.
        df = pd.concat([new_row, df]).reset_index(drop=True)
        df['accuracy'] = pd.to_numeric(df['accuracy'])  # Digitize the accuracy column.
        plot_comparison_chart(df, prefix=prefix)

    def save_model(self):
        """
        Save in a stable folder derived from artifact_stem, so other components
        (temperature saving/loading, test-time eval) can find it deterministically.
        """
        out_dir = f"{self.artifact_stem}_savedmodel"
        os.makedirs(out_dir, exist_ok=True)
        self._savedmodel_dir = out_dir  # remember for debugging if needed

        # SavedModel (robust to custom layers)
        self._model.save(out_dir, save_format="tf")
        # Also save weights (optional)
        self._model.save_weights(os.path.join(out_dir, "weights.h5"))

        print(f"[save_model] Saved to: {out_dir}")

    def save_weights(self) -> None:
        """
        Save the weights and biases of the fully connected layers in numpy format and the entire weights in h5 format.
        :return: None
        """
        if self.model_name == "VGG-common" or self.model_name == "MobileNet":
            print("Saving all weights")
            self._model.save_weights(
                "../saved_models/dataset-{}_mammogramtype-{}_model-{}_lr-{}_b-{}_e1-{}_e2-{}_roi-{}_{}_all_weights.h5".format(
                    config.dataset,
                    config.mammogram_type,
                    config.model,
                    config.learning_rate,
                    config.batch_size,
                    config.max_epoch_frozen,
                    config.max_epoch_unfrozen,
                    config.is_roi,
                    config.name)
            )
            print("Saving {} layer weights".format(self._model.layers[2].name))
            weights_and_biases = self._model.layers[2].get_weights()
            np.save(
                "../saved_models/dataset-{}_mammogramtype-{}_model-{}_lr-{}_b-{}_e1-{}_e2-{}_roi-{}_{}_fc_weights.npy".format(
                    config.dataset,
                    config.mammogram_type,
                    config.model,
                    config.learning_rate,
                    config.batch_size,
                    config.max_epoch_frozen,
                    config.max_epoch_unfrozen,
                    config.is_roi,
                    config.name),
                weights_and_biases
            )

    def load_minimias_weights(self) -> None:
        """
        Load the weights from all the layers pre-trained on the binary mini-MIAS dataset.
        :return: None.
        """
        print("Loading all layers mini-MIAS-binary weights from h5 file.")
        self._model.load_weights(
            "../saved_models/dataset-mini-MIAS-binary_mammogramtype-all_model-MobileNet_lr-0.0001_b-2_e1-150_e2-50_roi-False__all_weights.h5"
        )
        # self._model.load_weights(
        #     "/cs/scratch/agj6/saved_models/dataset-{}_mammogramtype-{}_model-{}_lr-{}_b-{}_e1-{}_e2-{}_roi-{}_{}_all_weights.h5".format(
        #         config.dataset,
        #         config.mammogram_type,
        #         config.model,
        #         config.learning_rate,
        #         config.batch_size,
        #         config.max_epoch_frozen,
        #         config.max_epoch_unfrozen,
        #         config.is_roi,
        #         config.name)
        # )

    def load_minimias_fc_weights(self) -> None:
        """
        Load and set the weights from the fully connected layers pre-trained on the binary mini-MIAS dataset.
        :return: None.
        """
        print("Loading only FC layers mini-MIAS-binary weights from npy file.")
        weights = np.load(
            "../saved_models/dataset-mini-MIAS-binary_mammogramtype-all_model-MobileNet_lr-0.0001_b-2_e1-150_e2-50_roi-False__fc_weights.npy"
        )
        # weights = np.load(
        #     "/cs/scratch/agj6/saved_models/dataset-{}_mammogramtype-{}_model-{}_lr-{}_b-{}_e1-{}_e2-{}_roi-{}_{}_fc_weights.npy".format(
        #         config.dataset,
        #         config.mammogram_type,
        #         config.model,
        #         config.learning_rate,
        #         config.batch_size,
        #         config.max_epoch_frozen,
        #         config.max_epoch_unfrozen,
        #         config.is_roi,
        #     config.name)
        # )
        self._model.layers[2].set_weights(weights)



    def save_temperature_param(self, base_out_path_no_ext=None):
        stem = base_out_path_no_ext or self.artifact_stem
        if getattr(self, "_temperature_", None) is not None:
            save_temperature(stem, self._temperature_,
                                suffix=getattr(config, "calibration_file_suffix", "_temperature.json"))

    def load_temperature_param(self, base_out_path_no_ext=None):
        stem = base_out_path_no_ext or self.artifact_stem
        T = load_temperature(stem, suffix=getattr(config, "calibration_file_suffix", "_temperature.json"))
        self._temperature_ = T

    def get_artifact_stem(self):
        return self.artifact_stem



    @property
    def model(self):
        """
        CNN model getter.
        :return: the model.
        """
        return self._model


    @model.setter
    def model(self, value) -> None:
        """
        CNN model setter.
        :param value:
        :return: None
        """
        pass


def test_model_evaluation(y_true: list, predictions, label_encoder: LabelEncoder, classification_type: str,
                          runtime, threshold = 0.5) -> None:
    """
    Function to evaluate a loaded model not instantiated with the CnnModel class.
    :param y_true: Ground truth of the data in one-hot-encoding type.
    :param predictions: Labels predicted.
    :param label_encoder: The label encoder for y value (label).
    :param classification_type: The classification type. Ex: N-B-M: normal, benign and malignant; B-M: benign and
    malignant.
    :param runtime: Runtime in seconds.
    :return: None.
    """
    prefix = ( (getattr(config, "name", "") or
           f"{config.dataset}_{config.model}_"
           f"{getattr(config,'split_mode','patient')}_"
           f"{getattr(config,'preprocess','none')}_"
           f"{getattr(config,'loss_type','weighted_ce')}"
           f"{'_cal' if getattr(config,'calibrate', False) else ''}") + "_test_" )
    # Inverse transform y_true and y_pred from one-hot-encoding to original label.
    if label_encoder.classes_.size == 2:
        y_true_inv = y_true
        y_pred_inv = (predictions >= threshold).astype(int)
    else:
        y_true_inv = label_encoder.inverse_transform(np.argmax(y_true, axis=1))
        y_pred_inv = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    # Calculate accuracy.
    accuracy = float('{:.4f}'.format(accuracy_score(y_true_inv, y_pred_inv)))
    print("Accuracy = {}\n".format(accuracy))

    # Generate CSV report.
    generate_csv_report(y_true_inv, y_pred_inv, label_encoder, accuracy)
    generate_csv_metadata(runtime)

    # Plot confusion matrix and normalised confusion matrix.
    cm = confusion_matrix(y_true_inv, y_pred_inv)  # Calculate CM with original label of classes
    plot_confusion_matrix(cm, 'd', label_encoder, False, prefix=prefix)
    # Calculate normalized confusion matrix with original label of classes.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized[np.isnan(cm_normalized)] = 0
    plot_confusion_matrix(cm_normalized, '.2f', label_encoder, True, prefix=prefix)

    # Plot ROC curve.
    if label_encoder.classes_.size == 2:  # binary classification
        plot_roc_curve_binary(y_true, predictions, prefix=prefix)
    elif label_encoder.classes_.size >= 2:  # multi classification
        plot_roc_curve_multiclass(y_true, predictions, label_encoder, prefix=prefix)

    # Compare results with other similar papers' result.
    with open(
            'data_visualisation/other_paper_results.json') as config_file:  # Load other papers' results from JSON.
        data = json.load(config_file)

    dataset_key = config.dataset
    if config.dataset == "mini-MIAS-binary":
        dataset_key = "mini-MIAS"

    df = pd.DataFrame.from_records(data[dataset_key][classification_type],
                                   columns=['paper', 'accuracy'])  # Filter data by dataset and classification type.
    new_row = pd.DataFrame({'paper': 'Dissertation', 'accuracy': accuracy},
                           index=[0])  # Add model result into dataframe to compare.
    df = pd.concat([new_row, df]).reset_index(drop=True)
    df['accuracy'] = pd.to_numeric(df['accuracy'])  # Digitize the accuracy column.
    plot_comparison_chart(df, prefix)






    
def make_binary_focal_loss(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        # y_pred are probabilities (after sigmoid)
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1.0 - eps)
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        w  = tf.where(K.equal(y_true, 1), alpha, 1 - alpha)
        return K.mean(-w * K.pow(1.0 - pt, gamma) * K.log(pt))
    return _focal