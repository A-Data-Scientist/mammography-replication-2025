import argparse
from collections import Counter
import time

import numpy as np, collections
import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy
import os
os.environ["MPLBACKEND"] = "Agg" 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cnn_models.cnn_model import CnnModel, test_model_evaluation
import config
from data_operations.dataset_feed import create_dataset
from data_operations.data_preprocessing import calculate_class_weights, dataset_stratified_split, \
    import_cbisddsm_testing_dataset, import_cbisddsm_training_dataset, import_minimias_dataset, \
    import_cbisddsm_patientwise_datasets
from data_operations.data_transformations import generate_image_transforms
from utils import create_label_encoder, load_trained_model, print_cli_arguments, print_error_message, \
    print_num_gpus_available, print_runtime, set_random_seeds
from data_operations.data_preprocessing import encode_labels
from calibration.temperature import (
    fit_temperature_binary, apply_temperature_binary,
    save_temperature, load_temperature
)

def y_from_dataset(ds):
    """
    Collect labels from a batched tf.data.Dataset robustly.
    Works whether each element is a scalar (), shape (1,), or a batch (B,).
    """
    ys = []
    for y in ds.map(lambda x, y: y).as_numpy_iterator():  
        a = np.array(y)
        ys.append(a.ravel())  # force (B,) even if (), (1,), or (B,)
    if not ys:
        return np.array([], dtype=np.int32)
    return np.concatenate(ys, axis=0)

def manual_acc_from_dataset(model, val_dataset, threshold=0.5):
    """
    Compute manual accuracy using the SAME dataset ordering used for predict().
    """
    opt = tf.data.Options()
    opt.experimental_deterministic = True
    val_dataset = val_dataset.with_options(opt)

    y_true = y_from_dataset(val_dataset).astype(int).ravel()
    # Use the underlying Keras model to get probabilities (sigmoid output)
    p = model._model.predict(val_dataset, verbose=0).ravel()
    y_pred = (p >= threshold).astype(int)
    return (y_true == y_pred).mean()

def _collect_labels_from_dataset(ds):
    # Pull labels in the exact iteration order used for predict/evaluate
    return np.concatenate([y.numpy().ravel() for _, y in ds], axis=0)

def keras_binary_acc_stream(keras_model, ds, threshold=0.5):
    # Compute BinaryAccuracy exactly like Keras' metric pipeline
    m = tf.keras.metrics.BinaryAccuracy(threshold=threshold)
    for xb, yb in ds:
        preds = keras_model(xb, training=False)
        m.update_state(yb, preds)
    return float(m.result().numpy())

def run_one_impl(split_mode, preprocess, loss_type, calibrate, l_e, short_epochs=True):
    import os, collections

    # --- keep a copy of epochs for quick sweeps ---
    e1, e2 = config.max_epoch_frozen, config.max_epoch_unfrozen
    if short_epochs:
        config.max_epoch_frozen, config.max_epoch_unfrozen = 3, 1

    # --- knobs for this implementation ---
    config.split_mode = split_mode
    config.preprocess = preprocess
    config.loss_type  = loss_type
    config.calibrate  = calibrate
    config.preprocess = preprocess
    config.use_clahe = (preprocess == "clahe")

    # --- data: classic vs patient-wise ---
    if getattr(config, "split_mode", "patient") == "patient":
        X_train, y_train, X_val, y_val, X_test, y_test = import_cbisddsm_patientwise_datasets(
            l_e,
            train=getattr(config, "train_frac", 0.70),
            val=getattr(config, "val_frac", 0.15),
            test=getattr(config, "test_frac", 0.15),
            seed=getattr(config, "RANDOM_SEED", 42)
        )
    else:
        images, labels = import_cbisddsm_training_dataset(l_e)
        X_train, X_val, y_train, y_val = dataset_stratified_split(
            split=0.25, dataset=images, labels=labels
        )

    # --- tf.data datasets for training ---
    train_dataset = create_dataset(X_train, y_train)
    val_dataset   = create_dataset(X_val, y_val)

    # deterministic iteration
    opt = tf.data.Options()
    opt.experimental_deterministic = True
    val_dataset = val_dataset.with_options(opt)

    # --- class weights for loss (only for weighted_ce) ---
    class_weights = calculate_class_weights(y_train, l_e) if loss_type == "weighted_ce" else None

    # --- model & training ---
    model = CnnModel(config.model, l_e.classes_.size)
    model.train_model(train_dataset, val_dataset, None, None, class_weights)


    # --- collect labels & uncalibrated probs from THE SAME pipeline ---
    y_val_ds = y_from_dataset(val_dataset).astype(int).ravel()
    preds_val_uncal = model._model.predict(val_dataset, verbose=0).ravel()
    # --- accuracy: manual & streaming BinaryAccuracy ---
    manual_acc = ((preds_val_uncal >= 0.5).astype(int) == y_val_ds).mean()
    keras_acc_stream = keras_binary_acc_stream(model._model, val_dataset, threshold=0.5)
    print(f"[sanity] stream_bin_acc={keras_acc_stream:.4f}  manual_acc={manual_acc:.4f}")

    # --- calibration (optional) ---
    preds_val_cal = None
    if calibrate:
        T = fit_temperature_binary(y_val_ds, preds_val_uncal)
        setattr(model, "_temperature_", T)
        preds_val_cal = apply_temperature_binary(preds_val_uncal, T)

        # try to persist T (optional)
        try:
            base_no_ext = model.get_artifact_stem()
            model.save_temperature_param(base_no_ext)
        except Exception as e:
            print(f"[temperature] Warning: could not save T: {e}")

    # --- calibration report ---
    from data_visualisation.calibration_report import calibration_report
    tag    = f"cbis_val_{split_mode}_{preprocess}_{loss_type}" + ("_cal" if calibrate else "")
    prefix = os.path.join("reports", "calibration", tag)
    summary = calibration_report(y_val_ds, preds_val_uncal, p_cal=preds_val_cal, prefix=prefix)

    # choose which block to use
    block_key = "cal" if (calibrate and preds_val_cal is not None) else "uncal"
    ops       = summary[block_key]["ops"]

    # pick a specificity you care about, e.g. 0.80
    thr = ops[0.80]["thr"]

    # --- confusion matrix + ROC for THIS implementation ---
    eval_probs = preds_val_cal if (calibrate and preds_val_cal is not None) else preds_val_uncal

    test_model_evaluation(
        y_true=y_val_ds,
        predictions=eval_probs,
        label_encoder=l_e,
        classification_type='B-M',
        runtime=0.0,  # per-impl runtime not critical for sweep
        threshold=thr,
    )

    # --- summarise accuracies & save debug preds ---
    acc_uncal = ((preds_val_uncal >= 0.5).astype(int) == y_val_ds).mean()
    acc_cal   = None if preds_val_cal is None else ((preds_val_cal >= 0.5).astype(int) == y_val_ds).mean()

    debug_dir = os.path.join("debug_preds")
    os.makedirs(debug_dir, exist_ok=True)

    tag = f"{config.dataset}_{config.model}_{split_mode}_{preprocess}_{loss_type}"
    if calibrate:
        tag += "_cal"

    np.savez(
        os.path.join(debug_dir, f"{tag}_val.npz"),
        y=y_val_ds,
        p_uncal=preds_val_uncal,
        p_cal=(preds_val_cal if preds_val_cal is not None else np.array([]))
    )
    print(f"[debug] Saved val preds to {os.path.join(debug_dir, f'{tag}_val.npz')}")

    # --- restore epochs ---
    config.max_epoch_frozen, config.max_epoch_unfrozen = e1, e2

    return {
        "split_mode": split_mode,
        "preprocess": preprocess,
        "loss_type": loss_type,
        "calibrate": calibrate,
        "acc_val_uncal": float(acc_uncal),
        "acc_val_cal": None if acc_cal is None else float(acc_cal),
        "keras_val_bin_acc_stream": float(keras_acc_stream),
    }


def main() -> None:
    """
    Program entry point. Parses command line arguments to decide which dataset and model to use.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :return: None.
    """
    set_random_seeds()
    parse_command_line_arguments()
    print_num_gpus_available()

    # Create label encoder.
    l_e = create_label_encoder()

    # Run in training mode.
    if config.run_mode == "train":

        print("-- Training model --\n")

        # Start recording time.
        start_time = time.time()


        if config.impl_sweep and config.dataset == "CBIS-DDSM":
            print("-- Running implementation sweep (short epochs) --")
            results = []

            # Baseline (classic split, no calibration)
            results.append(run_one_impl(split_mode="classic", preprocess="none", loss_type="weighted_ce",
                                        calibrate=False, l_e=l_e, short_epochs=False))

            # Patient-wise, no calibration
            results.append(run_one_impl(split_mode="patient", preprocess="none", loss_type="weighted_ce",
                                        calibrate=False, l_e=l_e, short_epochs=False))

            # Patient-wise + calibration
            results.append(run_one_impl(split_mode="patient", preprocess="none", loss_type="weighted_ce",
                                        calibrate=True, l_e=l_e, short_epochs=False))

            # Optional: add a CLAHE + focal row
            results.append(run_one_impl(split_mode="patient", preprocess="clahe", loss_type="focal",
                                        calibrate=True, l_e=l_e, short_epochs=False))

            # Pretty print summary
            print("\n=== Validation Accuracy Summary (CBIS-DDSM) ===")
            for r in results:
                tag = f"{r['split_mode']}, prep={r['preprocess']}, loss={r['loss_type']}, cal={r['calibrate']}"
                if r["acc_val_cal"] is None:
                    print(f"{tag:55s}  Acc (uncal): {r['acc_val_uncal']:.4f}")
                else:
                    print(f"{tag:55s}  Acc (uncal): {r['acc_val_uncal']:.4f}  |  Acc (cal): {r['acc_val_cal']:.4f}")
            return  # skip the regular single-run path


        # Multi-class classification (mini-MIAS dataset)
        if config.dataset == "mini-MIAS":
            # 1) Load full dataset
            images, labels = import_minimias_dataset(
                data_dir="../data/{}/images_processed".format(config.dataset),
                label_encoder=l_e
            )

            # 2) Split 80/20 into train+val vs test
            X_train_all, X_test, y_train_all, y_test = dataset_stratified_split(
                split=0.20, dataset=images, labels=labels
            )

            # 3) Split train into train/val (75/25 of the 80% ⇒ 60/20/20 overall)
            X_train, X_val, y_train, y_val = dataset_stratified_split(
                split=0.25, dataset=X_train_all, labels=y_train_all
            )

            # 4) Class weights from the pre-augmented training labels
            class_weights = calculate_class_weights(y_train, l_e)

            # 5) Optional image augmentation on arrays (returns new arrays)
            #    (Keeps X/y aligned—do NOT shuffle labels separately)
            X_train_aug, y_train_aug = generate_image_transforms(X_train, y_train)

            # 6) Build tf.data datasets for training/validation
            train_dataset = create_dataset(X_train_aug, y_train_aug)
            validation_dataset = create_dataset(X_val, y_val)

            # 7) Create and train model
            model = CnnModel(config.model, l_e.classes_.size)

            if config.verbose_mode:
                print(f"Training set size (aug): {len(y_train_aug)}")
                print(f"Validation set size:    {len(y_val)}")
                print(f"Test set size:          {len(y_test)}")

            if config.loss_type == "weighted_ce":
                # weighted CE: pass class_weights
                model.train_model(train_dataset, validation_dataset, None, None, class_weights)
            else:
                # focal loss: do NOT pass class weights
                model.train_model(train_dataset, validation_dataset, None, None, None)


        # Binary classification (binarised mini-MIAS dataset)
        elif config.dataset == "mini-MIAS-binary":
            # Import entire dataset.
            images, labels = import_minimias_dataset(data_dir="../data/{}/images_processed".format(config.dataset),
                                                     label_encoder=l_e)

            # Split dataset into training/test/validation sets (80/20% split).
            X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.20, dataset=images, labels=labels)

            # Create CNN model and split training/validation set (80/20% split).
            model = CnnModel(config.model, l_e.classes_.size)
            # model.load_minimias_weights()
            # model.load_minimias_fc_weights()

            # Fit model.
            if config.verbose_mode:
                print("Training set size: {}".format(X_train.shape[0]))
                print("Validation set size: {}".format(X_val.shape[0]))
            model.train_model(X_train, X_val, y_train, y_val, None)


        elif config.dataset == "CBIS-DDSM":
            if getattr(config, "split_mode", "patient") == "patient":
                X_train, y_train, X_val, y_val, X_test, y_test = import_cbisddsm_patientwise_datasets(
                    l_e,
                    train=getattr(config, "train_frac", 0.70),
                    val=getattr(config, "val_frac", 0.15),
                    test=getattr(config, "test_frac", 0.15),
                    seed=getattr(config, "RANDOM_SEED", 42)
                )
            else:
                images, labels = import_cbisddsm_training_dataset(l_e)
                X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25, dataset=images, labels=labels)

            train_dataset = create_dataset(X_train, y_train)
            validation_dataset = create_dataset(X_val, y_val)

            class_weights = calculate_class_weights(y_train, l_e)

            model = CnnModel(config.model, l_e.classes_.size)

            if config.loss_type == "weighted_ce":
                model.train_model(train_dataset, validation_dataset, None, None, class_weights)
            else:
                # focal loss: don't pass class weights (alpha in focal handles imbalance)
                model.train_model(train_dataset, validation_dataset, None, None, None)

            #model.train_model(train_dataset, validation_dataset, None, None, class_weights)




        # # Binary classification (CBIS-DDSM dataset).
        # elif config.dataset == "CBIS-DDSM":
        #     images, labels = import_cbisddsm_training_dataset(l_e)

        #     # Split training dataset into training/validation sets (75%/25% split).
        #     X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25, dataset=images, labels=labels)
        #     train_dataset = create_dataset(X_train, y_train)
        #     validation_dataset = create_dataset(X_val, y_val)

        #     # Calculate class weights.
        #     class_weights = calculate_class_weights(y_train, l_e)

        #     # Create and train CNN model.
        #     model = CnnModel(config.model, l_e.classes_.size)
        #     # model.load_minimias_fc_weights()
        #     # model.load_minimias_weights()

        #     # Fit model.
        #     if config.verbose_mode:
        #         print("Training set size: {}".format(X_train.shape[0]))
        #         print("Validation set size: {}".format(X_val.shape[0]))
        #     model.train_model(train_dataset, validation_dataset, None, None, class_weights)

        # Save training runtime.
        runtime = round(time.time() - start_time, 2)

        # Save the model and its weights/biases.
        model.save_model()
        model.save_weights()

        # Evaluate training results.
        print_cli_arguments()
        if config.dataset == "mini-MIAS":
            model.make_prediction(X_val)
            model.evaluate_model(y_val, l_e, 'N-B-M', runtime)
        elif config.dataset == "mini-MIAS-binary":
            model.make_prediction(X_val)
            model.evaluate_model(y_val, l_e, 'B-M', runtime)
        # elif config.dataset == "CBIS-DDSM":
        #     model.make_prediction(validation_dataset)
        #     model.evaluate_model(y_val, l_e, 'B-M', runtime)
        elif config.dataset == "CBIS-DDSM":
            # 1) Uncalibrated validation evaluation (kept for comparison)
            preds_val_uncal = model.predict(x=validation_dataset)
            model_runtime = runtime
            test_model_evaluation(y_val, preds_val_uncal, l_e, 'B-M', model_runtime)

            # 2) Temperature scaling on validation set (binary)
            if getattr(config, "calibrate", True):
                T = fit_temperature_binary(y_val, preds_val_uncal)
                # stash on model (optional)
                setattr(model, "_temperature_", T)

                # 3) Evaluate calibrated validation
                preds_val_cal = apply_temperature_binary(preds_val_uncal, T)
                test_model_evaluation(y_val, preds_val_cal, l_e, 'B-M', model_runtime)

                # 4) Save T with the model artifacts
                # Reuse the same base path you use in save_model/save_weights
                # If your save methods build a path like "<models_dir>/<name>_savedmodel"
                # construct a base path-without-extension to drop the JSON next to them:
                base_no_ext = model.get_artifact_stem()  # or whichever base path you use; ensure it's a "no extension" stem
                model.save_temperature_param(base_no_ext)

        print_runtime("Training", runtime)

    # Run in testing mode.
    elif config.run_mode == "test":

        print("-- Testing model --\n")

        # Start recording time.
        start_time = time.time()

        # Test multi-class classification (mini-MIAS dataset).
        if config.dataset == "mini-MIAS":
            images, labels = import_minimias_dataset(data_dir="../data/{}/images_processed".format(config.dataset),
                                                     label_encoder=l_e)
            _, X_test, _, y_test = dataset_stratified_split(split=0.20, dataset=images, labels=labels)
            model = load_trained_model()
            predictions = model.predict(x=X_test)
            runtime = round(time.time() - start_time, 2)
            test_model_evaluation(y_test, predictions, l_e, 'N-B-M', runtime)

        # Test binary classification (binarised mini-MIAS dataset).
        elif config.dataset == "mini-MIAS-binary":
            pass

        # Test binary classification (CBIS-DDSM dataset).
        elif config.dataset == "CBIS-DDSM":
            images, labels = import_cbisddsm_testing_dataset(l_e)
            test_dataset = create_dataset(images, labels)
            model = load_trained_model()

            preds_test_uncal = model.predict(x=test_dataset)

            # try to load T and apply; if absent, fall back to uncalibrated
            from calibration.temperature import load_temperature, apply_temperature_binary

            # IMPORTANT: base_no_ext must match the stem you used when SAVING T in train mode
            # Option A (if your CnnModel sets this attribute and load_trained_model returns that wrapper):
            # base_no_ext = model._model_path

            # Option B (common fallback): reconstruct the same stem used in save_model/save_temperature
            import os
            base_no_ext = os.path.join(getattr(config, "models_dir", "models"),
                                    f"{config.model}_{config.dataset}")

            T = load_temperature(base_no_ext, suffix=getattr(config, "calibration_file_suffix", "_temperature.json"))
            preds_test = apply_temperature_binary(preds_test_uncal, T) if T is not None else preds_test_uncal

            from data_visualisation.calibration_report import calibration_report
            prefix = os.path.join("reports", "calibration", f"cbis_test_{config.split_mode}_{config.preprocess}_{config.loss_type}")
            calibration_report(labels, preds_test_uncal, p_cal=(preds_test if T is not None else None), prefix=prefix)

            runtime = round(time.time() - start_time, 2)
            test_model_evaluation(labels, preds_test, l_e, 'B-M', runtime)


        # elif config.dataset == "CBIS-DDSM":
        #     # Use the saved patient-wise test manifest
        #     import pandas as pd, os
        #     test_manifest = "../data/CBIS-DDSM/splits/patientwise/test.csv"
        #     if not os.path.exists(test_manifest):
        #         # Fallback to original behavior if manifest not present
        #         images, labels = import_cbisddsm_testing_dataset(l_e)
        #     else:
        #         df = pd.read_csv(test_manifest)
        #         images = df["img_path"].astype(str).values
        #         labels = encode_labels(df["label"].values, l_e)  # import encode_labels if not in scope

        #     test_dataset = create_dataset(images, labels)
        #     model = load_trained_model()
        #     predictions = model.predict(x=test_dataset)
        #     runtime = round(time.time() - start_time, 2)
        #     test_model_evaluation(labels, predictions, l_e, 'B-M', runtime)

        print_runtime("Testing", runtime)


def parse_command_line_arguments() -> None:
    """
    Parse command line arguments and save their value in config.py.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        default="CBIS-DDSM",
                        required=True,
                        help="The dataset to use. Must be either 'mini-MIAS', 'mini-MIAS-binary' or 'CBIS-DDMS'."
                        )
    parser.add_argument("-mt", "--mammogramtype",
                        default="all",
                        help="The mammogram type to use. Can be either 'calc', 'mass' or 'all'. Defaults to 'all'."
                        )
    parser.add_argument("-m", "--model",
                        required=True,
                        help="The model to use. Must be either 'VGG-common', 'VGG', 'ResNet', 'Inception', 'DenseNet', 'MobileNet' or 'CNN'."
                        )
    parser.add_argument("-r", "--runmode",
                        default="train",
                        help="The mode to run the code in. Either train the model from scratch and make predictions, "
                             "otherwise load a previously trained model for predictions. Must be either 'train' or "
                             "'test'. Defaults to 'train'."
                        )
    parser.add_argument("-lr", "--learning-rate",
                        type=float,
                        default=1e-4,
                        help="The learning rate for the non-ImageNet-pre-trained layers. Defaults to 1e-3."
                        )
    parser.add_argument("-b", "--batchsize",
                        type=int,
                        default=16,
                        help="The batch size to use. Defaults to 2."
                        )
    parser.add_argument("-e1", "--max_epoch_frozen",
                        type=int,
                        default=100,
                        help="The maximum number of epochs in the first training phrase (with frozen layers). Defaults "
                             "to 100."
                        )
    parser.add_argument("-e2", "--max_epoch_unfrozen",
                        type=int,
                        default=50,
                        help="The maximum number of epochs in the second training phrase (with unfrozen layers). "
                             "Defaults to 50."
                        )
    # parser.add_argument("-gs", "--gridsearch",
    #                    action="store_true",
    #                    default=False,
    #                    help="Include this flag to run the grid search algorithm to determine the optimal "
    #                         "hyperparameters for the CNN model."
    #                    )
    parser.add_argument("--split_mode", choices=["patient", "classic"], default="patient",
                        help="Use patient-wise split or classic CSV split (CBIS-DDSM only).")
    parser.add_argument("--splits", default="0.70,0.15,0.15",
                        help="Train,Val,Test fractions for patient-wise splitting (e.g., 0.70,0.15,0.15).")
    parser.add_argument("-roi", "--roi",
                        action="store_true",
                        default=False,
                        help="Include this flag to use a cropped version of the images around the ROI. Only use with 'mini-MIAS' dataset."
                        )
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Verbose mode: include this flag additional print statements for debugging purposes."
                        )
    parser.add_argument("-n", "--name",
                        default="",
                        help="The name of the experiment being tested. Defaults to an empty string."
                        )
    parser.add_argument("--preprocess", default="none",
                        choices=["none", "clahe"],
                        help="Image preprocessing: none or clahe")
    parser.add_argument("--loss-type", default="weighted_ce",
                        choices=["weighted_ce", "focal"],
                        help="Loss: weighted_ce or focal")
    parser.add_argument("--focal-alpha", type=float, default=0.8)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--calibrate", action="store_true", default=False,
                        help="Fit temperature scaling on validation and evaluate calibrated outputs.")
    parser.add_argument("--impl-sweep", action="store_true", default=False,
                    help="Run baseline vs patient-wise vs patient-wise+calibration (and optional CLAHE/focal) and print accuracy deltas.")

    args = parser.parse_args()
    config.dataset = args.dataset
    config.mammogram_type = args.mammogramtype
    config.model = args.model
    config.split_mode = args.split_mode
    config.preprocess = args.preprocess
    config.loss_type = args.loss_type
    config.focal_alpha = args.focal_alpha
    config.focal_gamma = args.focal_gamma
    config.calibrate = args.calibrate
    config.impl_sweep = args.impl_sweep
    try:
        t, v, te = [float(x) for x in args.splits.split(",")]
        config.train_frac, config.val_frac, config.test_frac = t, v, te
    except Exception:
        config.train_frac, config.val_frac, config.test_frac = 0.70, 0.15, 0.15
    config.run_mode = args.runmode
    if args.learning_rate <= 0:
        print_error_message()
    config.learning_rate = args.learning_rate
    if args.batchsize <= 0 or args.batchsize >= 25:
        print_error_message()
    config.batch_size = args.batchsize
    if all([args.max_epoch_frozen, args.max_epoch_unfrozen]) <= 0:
        print_error_message()
    config.max_epoch_frozen = args.max_epoch_frozen
    config.max_epoch_unfrozen = args.max_epoch_unfrozen
    # config.is_grid_search = args.gridsearch
    config.is_roi = args.roi
    config.verbose_mode = args.verbose
    config.name = args.name

    if config.verbose_mode:
        print_cli_arguments()


if __name__ == '__main__':
    main()
