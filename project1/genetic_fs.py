import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed


def fitness(X, y, chromosome):

    idx = np.where(chromosome == 1)[0]

    if len(idx) == 0:
        return 0

    score = cross_val_score(
        LinearSVC(max_iter=5000),
        X[:, idx],
        y,
        cv=3,
        scoring="accuracy",
        n_jobs=1          # 避免嵌套并行
    ).mean()

    return score


def genetic_feature_selection(
    X_train,
    y_train,
    X_test,
    target_dim,
    pop_size=20,
    generations=15,
    mutation_rate=0.05
):

    n_features = X_train.shape[1]

    # -------------------------
    # 初始化 population
    # 每个染色体约 target_dim 个特征
    # -------------------------
    population = []

    for _ in range(pop_size):

        chromosome = np.zeros(n_features)

        idx = np.random.choice(
            n_features,
            target_dim,
            replace=False
        )

        chromosome[idx] = 1

        population.append(chromosome)

    population = np.array(population)

    # -------------------------
    # GA 迭代
    # -------------------------
    for g in range(generations):

        # 并行计算 fitness
        scores = Parallel(n_jobs=-1)(
            delayed(fitness)(X_train, y_train, chromosome)
            for chromosome in population
        )

        scores = np.array(scores)

        # 保留最优 50%
        best_idx = np.argsort(scores)[-pop_size//2:]

        parents = population[best_idx]

        children = []

        while len(children) < pop_size:

            # 随机选择两个父代
            p1, p2 = parents[
                np.random.randint(len(parents), size=2)
            ]

            # 单点交叉
            point = np.random.randint(n_features)

            child = np.concatenate(
                [p1[:point], p2[point:]]
            )

            # 变异
            mutation_mask = np.random.rand(n_features) < mutation_rate

            child[mutation_mask] = 1 - child[mutation_mask]

            # 控制特征数量接近 target_dim
            idx = np.where(child == 1)[0]

            if len(idx) > target_dim:

                remove = np.random.choice(
                    idx,
                    len(idx) - target_dim,
                    replace=False
                )

                child[remove] = 0

            elif len(idx) < target_dim:

                zero_idx = np.where(child == 0)[0]

                add = np.random.choice(
                    zero_idx,
                    target_dim - len(idx),
                    replace=False
                )

                child[add] = 1

            children.append(child)

        population = np.array(children)

        print(
            f"Generation {g+1}/{generations} | "
            f"Best score: {scores.max():.4f}"
        )

    # -------------------------
    # 最终选择最优个体
    # -------------------------
    final_scores = Parallel(n_jobs=-1)(
        delayed(fitness)(X_train, y_train, chromosome)
        for chromosome in population
    )

    best = population[np.argmax(final_scores)]

    idx = np.where(best == 1)[0]

    Xtr = X_train[:, idx]
    Xte = X_test[:, idx]

    return Xtr, Xte