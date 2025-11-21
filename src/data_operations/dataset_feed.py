import tensorflow as tf
import os
import numpy as np
import pydicom
import glob
import config

def _to_str(x):
    # handles TF EagerTensor, bytes, str
    try:
        if hasattr(x, "numpy"):
            x = x.numpy()
        if isinstance(x, (bytes, bytearray)):
            return x.decode("utf-8", "ignore")
    except Exception:
        pass
    return str(x)

def _to_path(x):
    """Return a native Python string path from str/bytes/TF tensor."""
    # TensorFlow EagerTensor -> bytes or str
    if hasattr(x, "numpy"):
        x = x.numpy()
    # bytes -> str
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8")
    # already a str or anything else -> stringify
    return str(x)

def create_dataset(x, y):
    """
    Generates a TF dataset for feeding in the data.
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def _cbis_local_path(p):
    """
    Normalize any incoming CBIS path to your local tree under config.CBIS_ROOT.
    Critically: strip everything up to and including the LAST 'CBIS-DDSM/' so
    '.../manifest-xxx/CBIS-DDSM/Calc-Test_P_.../1-1.dcm' becomes
    'Calc-Test_P_.../1-1.dcm'.
    """
    s = _to_str(p).replace("\\", "/")

    # If it's already a real path, return it normalized
    if os.path.exists(s):
        return os.path.normpath(s)

    # Trim leading "./" and "data/"
    while s.startswith("./"):
        s = s[2:]
    if s.lower().startswith("data/"):
        s = s[5:]

    # Keep everything AFTER the LAST 'cbis-ddsm/'
    low = s.lower()
    anchor = "cbis-ddsm/"
    if anchor in low:
        start = low.rfind(anchor) + len(anchor)  # <-- rfind = last occurrence
        rel = s[start:]
    else:
        rel = s.lstrip("/")

    # If another 'cbis-ddsm/' still remains at the front, drop it too
    if rel.lower().startswith("cbis-ddsm/"):
        rel = rel[len("cbis-ddsm/"):]

    # Join under your local root
    full = os.path.normpath(os.path.join(config.CBIS_ROOT, *rel.split("/")))
    return full


def _load_dicom_py(filename):
    """
    Python-side DICOM loader used via tf.py_function.
    - Reads pixels
    - Applies RescaleSlope/RescaleIntercept
    - Inverts MONOCHROME1 if needed
    - Normalizes to [0,1]
    - Returns HxWx1 float32
    """
    path = _cbis_local_path(filename)

    if not os.path.exists(path):
        # Try a best-effort search using the last few path components
        tail = "/".join(path.replace("\\", "/").split("/")[-4:])  # adjust 3-5 if needed
        hits = glob.glob(os.path.join(config.CBIS_ROOT, "**", tail), recursive=True)
        if hits:
            path = hits[0]
        else:
            raise FileNotFoundError(
                f"CBIS file not found after normalization:\n  {path}\n  (CBIS_ROOT={config.CBIS_ROOT})"
            )
    ds = pydicom.dcmread(path)

    arr = ds.pixel_array.astype(np.float32)

    # Apply rescale if present
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # Invert if MONOCHROME1 (so brighter = higher values)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr

    # Normalize to [0,1]
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr[:] = 0.0

    # Add channel axis -> H x W x 1
    arr = np.expand_dims(arr, axis=-1).astype(np.float32)
    return arr


def parse_function(filename, label):
    """
    Mapping function to convert filename to array of pixel values.
    """
    # Load DICOM -> float32 [H, W, 1] in [0,1]
    image = tf.py_function(_load_dicom_py, [filename], tf.float32)
    image.set_shape([None, None, 1])  # unknown H,W at graph build time

    # Pick target size from config (fallback to 512x512)
    if   config.model == "VGG":
        height, width = config.MINI_MIAS_IMG_SIZE["HEIGHT"], config.MINI_MIAS_IMG_SIZE["WIDTH"]
    elif config.model == "VGG-common":
        height, width = config.VGG_IMG_SIZE["HEIGHT"],       config.VGG_IMG_SIZE["WIDTH"]
    elif config.model == "ResNet":
        height, width = config.RESNET_IMG_SIZE["HEIGHT"],    config.RESNET_IMG_SIZE["WIDTH"]
    elif config.model == "Inception":
        height, width = config.INCEPTION_IMG_SIZE["HEIGHT"], config.INCEPTION_IMG_SIZE["WIDTH"]
    elif config.model == "DenseNet":
        height, width = config.DENSE_NET_IMG_SIZE["HEIGHT"], config.DENSE_NET_IMG_SIZE["WIDTH"]
    elif config.model == "MobileNet":
        height, width = config.MOBILE_NET_IMG_SIZE["HEIGHT"], config.MOBILE_NET_IMG_SIZE["WIDTH"]
    else:
        height, width = 512, 512

    # If your backbone expects 3 channels (e.g., Keras MobileNet with imagenet weights),
    # uncomment the next line to replicate grayscale -> RGB:
    # image = tf.image.grayscale_to_rgb(image)

    image = tf.image.resize_with_pad(image, height, width)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label
