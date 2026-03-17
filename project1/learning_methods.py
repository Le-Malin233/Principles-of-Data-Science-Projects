from sklearn.manifold import TSNE, LocallyLinearEmbedding


def apply_tsne(X_train, X_test, dim):

    tsne = TSNE(n_components=dim, perplexity=30)

    X_train_p = tsne.fit_transform(X_train)

    X_test_p = tsne.fit_transform(X_test)

    return X_train_p, X_test_p


def apply_lle(X_train, X_test, dim, neighbors):

    lle = LocallyLinearEmbedding(
        n_components=dim,
        n_neighbors=neighbors,
        method="standard",      # 改为 standard
        eigen_solver="dense",   # 关键修复
        reg=1e-3,               # 正则化
        n_jobs=-1
    )

    Xtr = lle.fit_transform(X_train)

    Xte = lle.transform(X_test)

    return Xtr, Xte