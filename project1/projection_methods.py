import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from joblib import Parallel, delayed

def apply_pca(X_train, X_test, dim):

    pca = PCA(n_components=dim)

    Xtr = pca.fit_transform(X_train)

    Xte = pca.transform(X_test)

    return Xtr, Xte

def apply_incremental_pca(X_train, X_test, dim, batch_size=2048):
    """
    使用增量PCA，适合大数据集
    """
    pca = IncrementalPCA(n_components=dim, batch_size=batch_size)
    
    # 分批拟合
    n_samples = X_train.shape[0]
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        pca.partial_fit(X_train[start:end])
        print(f"Processed batch {start//batch_size + 1}/{(n_samples-1)//batch_size + 1}")
    
    Xtr = pca.transform(X_train)
    Xte = pca.transform(X_test)
    
    return Xtr, Xte

def apply_lda(X_train, X_test, y_train, dim):

    lda = LinearDiscriminantAnalysis(n_components=dim)

    Xtr = lda.fit_transform(
        X_train,
        y_train
    )

    Xte = lda.transform(X_test)

    return Xtr, Xte

def plot_2d_projection(X, y, method="PCA"):

    plt.figure(figsize=(8,6))

    classes = np.unique(y)

    for c in classes:

        idx = y == c

        plt.scatter(
            X[idx,0],
            X[idx,1],
            s=10,
            label=str(c),
            alpha=0.6
        )

    plt.title(method + " 2D Projection")

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.show()