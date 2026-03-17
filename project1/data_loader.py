import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_awa2_features(data_dir):
    """
    读取 AwA2 深度特征

    数据格式:
        AwA2-features.txt  (N × D)
        AwA2-labels.txt    (N)

    """

    feature_path = os.path.join(data_dir, "AwA2-features.txt")
    label_path = os.path.join(data_dir, "AwA2-labels.txt")

    features = np.loadtxt(feature_path)
    labels = np.loadtxt(label_path).astype(int)


    return features, labels


def split_train_test(X, y, train_ratio=0.6, random_state=42):

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    classes = np.unique(y)

    for c in classes:

        idx = np.where(y == c)[0]

        X_c = X[idx]
        y_c = y[idx]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_c, y_c,
            train_size=train_ratio,
            random_state=random_state,
            shuffle=True
        )

        X_train.append(X_tr)
        X_test.append(X_te)

        y_train.append(y_tr)
        y_test.append(y_te)

    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)

    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)

    return X_train, X_test, y_train, y_test