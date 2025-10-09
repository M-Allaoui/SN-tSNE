# datasets.py
import os
import numpy as np
import pandas as pd

def _as_float32(X):
    return np.asarray(X, dtype=np.float32)

def _flatten_labels(y):
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.ravel()
    return y

def load_isolet(root="Datasets"):
    """Loads ISOLET (UCI)."""
    train = pd.read_csv(os.path.join(root, "isolet1+2+3+4.data"), header=None)
    test  = pd.read_csv(os.path.join(root, "isolet5.data"), header=None)
    data = pd.concat([train, test], axis=0).reset_index(drop=True)
    X = _as_float32(data.iloc[:, :-1].values)
    y = _flatten_labels(data.iloc[:, -1].values)
    # ISOLET labels are 1..26; make them 0..25 (optional)
    if y.min() == 1:
        y = y - 1
    return X, y

def load_uci_har(root="Datasets/UCI HAR Dataset"):
    """Loads UCI HAR (Human Activity Recognition)."""
    X_train = pd.read_csv(os.path.join(root, "train/X_train.txt"), delim_whitespace=True, header=None)
    y_train = pd.read_csv(os.path.join(root, "train/y_train.txt"), delim_whitespace=True, header=None)
    X_test  = pd.read_csv(os.path.join(root, "test/X_test.txt"),  delim_whitespace=True, header=None)
    y_test  = pd.read_csv(os.path.join(root, "test/y_test.txt"),  delim_whitespace=True, header=None)
    X = _as_float32(np.vstack([X_train.values, X_test.values]))
    y = _flatten_labels(np.vstack([y_train.values, y_test.values]))
    return X, y

def load_fashion_mnist(split="test"):
    """Loads Fashion-MNIST via Keras. Returns flattened [0,1] images."""
    try:
        from tensorflow.keras.datasets import fashion_mnist
    except Exception:
        try:
            from keras.datasets import fashion_mnist  # fallback if tf.keras not available
        except Exception as e:
            raise ImportError("Keras / tf.keras is required to load Fashion-MNIST.") from e

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    if split == "train":
        X, y = x_train, y_train
    elif split == "test":
        X, y = x_test, y_test
    else:  # "all"
        X = np.concatenate([x_train, x_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

    X = X.reshape(X.shape[0], -1) / 255.0
    return _as_float32(X), _flatten_labels(y)

def load_usps_h5(path):
    """Loads USPS from an HDF5 file with groups train/test -> (data, target)."""
    import h5py
    with h5py.File(path, "r") as hf:
        X_tr = hf["train"]["data"][:]
        y_tr = hf["train"]["target"][:]
        X_te = hf["test"]["data"][:]
        y_te = hf["test"]["target"][:]
    X = _as_float32(np.vstack([X_tr, X_te]))
    y = _flatten_labels(np.hstack([y_tr, y_te]))
    return X, y

def load_coil20(root="COIL-20 dataset/coil-20"):
    """Loads COIL-20 from the image folder (PNG). Flattens images."""
    import imageio.v2 as imageio
    images, labels = [], []
    for class_folder in sorted(os.listdir(root)):
        class_path = os.path.join(root, class_folder)
        if not os.path.isdir(class_path):
            continue
        try:
            class_label = int(class_folder.replace("obj", ""))
        except ValueError:
            continue
        for file in sorted(os.listdir(class_path)):
            if file.lower().endswith(".png"):
                img = imageio.imread(os.path.join(class_path, file))
                images.append(img.flatten())
                labels.append(class_label)
    X = _as_float32(np.array(images))
    y = _flatten_labels(np.array(labels))
    # Make labels start from 0 (optional)
    y = y - y.min()
    # Normalize if images are 0..255
    if X.max() > 1.0:
        X = X / 255.0
    return X, y

def load_olivetti():
    """Loads Olivetti Faces from sklearn, flattened and normalized."""
    from sklearn.datasets import fetch_olivetti_faces
    data = fetch_olivetti_faces()
    X = _as_float32(data.data)
    y = _flatten_labels(data.target)
    return X, y

def load_digits():
    """Loads sklearn Digits, normalized to [0,1]."""
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data.astype(np.float32)
    X = X / X.max()
    y = _flatten_labels(digits.target)
    return X, y

def load_dataset(name, **kwargs):
    """
    Generic loader.
    name in {
        'isolet', 'uci_har', 'fashion_mnist', 'usps_h5', 'coil20',
        'olivetti', 'digits'
    }
    Extra args:
      - isolet: root=...
      - uci_har: root=...
      - fashion_mnist: split={'train','test','all'}
      - usps_h5: path=...
      - coil20: root=...
    """
    name = name.lower()
    if name == "isolet":
        return load_isolet(root=kwargs.get("root", "Datasets"))
    if name == "uci_har":
        return load_uci_har(root=kwargs.get("root", "Datasets/UCI HAR Dataset"))
    if name == "fashion_mnist":
        return load_fashion_mnist(split=kwargs.get("split", "test"))
    if name == "usps_h5":
        path = kwargs.get("path")
        if not path:
            raise ValueError("Please provide path=... to the USPS HDF5 file.")
        return load_usps_h5(path=path)
    if name == "coil20":
        return load_coil20(root=kwargs.get("root", "COIL-20 dataset/coil-20"))
    if name == "olivetti":
        return load_olivetti()
    if name == "digits":
        return load_digits()
    raise ValueError(f"Unknown dataset name: {name}")
