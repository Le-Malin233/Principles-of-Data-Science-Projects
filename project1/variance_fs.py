from sklearn.feature_selection import VarianceThreshold


def variance_threshold_fs(X_train, X_test, threshold):

    selector = VarianceThreshold(threshold=threshold)

    Xtr = selector.fit_transform(X_train)
    Xte = selector.transform(X_test)

    dim = Xtr.shape[1]

    return Xtr, Xte, dim