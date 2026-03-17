import os
import csv
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler

from data_loader import load_awa2_features, split_train_test
from variance_fs import variance_threshold_fs
from genetic_fs import genetic_feature_selection
from svm_utils import train_svm_with_cv, evaluate
from projection_methods import apply_pca, apply_lda, plot_2d_projection, apply_incremental_pca
from learning_methods import apply_lle, apply_tsne


# =====================================================
# 0 temp setting (for Chinese username)
# =====================================================

english_temp = r'D:\data_science\prj1\temp'

if not os.path.exists(english_temp):
    os.makedirs(english_temp)

joblib.parallel_backend('loky', cache_dir=english_temp)

os.environ['JOBLIB_TEMP_FOLDER'] = english_temp
os.environ['PYTHONUTF8'] = '1'


# =====================================================
# train SVM
# =====================================================

def svm_train_test(Xtr, Xte, y_train, y_test):

    model, best_c = train_svm_with_cv(Xtr, y_train)

    acc = evaluate(model, Xte, y_test)

    return acc, best_c


# =====================================================
# 1 load dataset
# =====================================================

print("\n========== Loading Dataset ==========")

data_dir = "ResNet101"

X, y = load_awa2_features(data_dir)

X_train, X_test, y_train, y_test = split_train_test(X, y)

# standardize（for LLE / t-SNE）
print("\nStandardizing data...")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =====================================================
# 2 Variance Threshold Feature Selection
# =====================================================

print("\n========== Variance Threshold Experiment ==========")

thresholds = [0.5, 0.7, 1.0, 1.5, 2.0]

with open("variance_results.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["Method", "Threshold", "Dimension", "Accuracy"])

    for t in thresholds:

        print(f"\nRunning threshold = {t}")

        Xtr, Xte, dim = variance_threshold_fs(
            X_train,
            X_test,
            t
        )

        acc, best_c = svm_train_test(
            Xtr,
            Xte,
            y_train,
            y_test
        )

        print("Reduced dimension:", dim)
        print("Accuracy:", acc)

        writer.writerow([
            "VarianceThreshold",
            t,
            dim,
            acc
        ])


# =====================================================
# 3 Genetic Feature Selection
# =====================================================

print("\n========== Genetic Feature Selection ==========")

genetic_dims = [100, 200, 500]

with open("genetic_results.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["Method", "Dimension", "Accuracy"])

    for dim in genetic_dims:

        print(f"\nRunning Genetic FS dim = {dim}")

        Xtr, Xte = genetic_feature_selection(
            X_train,
            y_train,
            X_test,
            target_dim=dim
        )

        acc, best_c = svm_train_test(
            Xtr,
            Xte,
            y_train,
            y_test
        )

        print("Accuracy:", acc)

        writer.writerow([
            "Genetic",
            dim,
            acc
        ])


# =====================================================
# 4 PCA Experiment
# =====================================================

print("\n========== PCA Experiment ==========")

pca_dims = [2,4,8,16,32,64,128,256,512,1024,2048]

with open("pca_results.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["Method", "Dimension", "Accuracy"])

    for dim in pca_dims:

        print(f"\nRunning PCA dim = {dim}")

        Xtr, Xte = apply_incremental_pca(
            X_train,
            X_test,
            dim
        )

        acc, best_c = svm_train_test(
            Xtr,
            Xte,
            y_train,
            y_test
        )

        print("Accuracy:", acc)

        writer.writerow([
            "PCA",
            dim,
            acc
        ])


# =====================================================
# 5 LDA Experiment
# =====================================================

print("\n========== LDA Experiment ==========")

lda_dims = [4,9,14,19,24,29,34,39,44,49]

with open("lda_results.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["Method", "Dimension", "Accuracy"])

    for dim in lda_dims:

        print(f"\nRunning LDA dim = {dim}")

        Xtr, Xte = apply_lda(
            X_train,
            X_test,
            y_train,
            dim
        )

        acc, best_c = svm_train_test(
            Xtr,
            Xte,
            y_train,
            y_test
        )

        print("Accuracy:", acc)

        writer.writerow([
            "LDA",
            dim,
            acc
        ])


# =====================================================
# 6 LLE Experiment
# =====================================================

print("\n========== LLE Neighbor Experiment ==========")

dim_fixed = 100
neighbors_list = [10, 20, 50, 100, 200]

best_neighbors = None
best_acc = 0

for nb in neighbors_list:

    print(f"\nRunning neighbors = {nb}")

    Xtr, Xte = apply_lle(
        X_train,
        X_test,
        dim_fixed,
        nb
    )

    acc, best_c = svm_train_test(
        Xtr,
        Xte,
        y_train,
        y_test
    )

    print("Accuracy:", acc)

    if acc > best_acc:
        best_acc = acc
        best_neighbors = nb


print("\nBest neighbors:", best_neighbors)


print("\n========== LLE Dimension Experiment ==========")

dims = [500, 1000, 2000]

with open("lle_results.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["Method", "Dimension", "Neighbors", "Accuracy"])

    for dim in dims:

        print(f"\nRunning dim = {dim}")

        Xtr, Xte = apply_lle(
            X_train,
            X_test,
            dim,
            best_neighbors
        )

        acc, best_c = svm_train_test(
            Xtr,
            Xte,
            y_train,
            y_test
        )

        print("Accuracy:", acc)

        writer.writerow([
            "LLE",
            dim,
            best_neighbors,
            acc
        ])


# =====================================================
# 7 t-SNE Experiment
# =====================================================

print("\n========== t-SNE Experiment ==========")

tsne_dims = [2, 3]

with open("tsne_results.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["Method", "Dimension", "Accuracy"])

    for dim in tsne_dims:

        print(f"\nRunning t-SNE dim = {dim}")

        Xtr, Xte = apply_tsne(
            X_train,
            X_test,
            dim
        )

        acc, best_c = svm_train_test(
            Xtr,
            Xte,
            y_train,
            y_test
        )

        print("Accuracy:", acc)

        writer.writerow([
            "t-SNE",
            dim,
            acc
        ])


# =====================================================
# 8 2D Visualization
# =====================================================

print("\n========== Visualization ==========")

# PCA 2D
X_pca_2d, _ = apply_pca(
    X_train,
    X_test,
    2
)

plot_2d_projection(
    X_pca_2d,
    y_train,
    method="PCA"
)

# LDA 2D
X_lda_2d, _ = apply_lda(
    X_train,
    X_test,
    y_train,
    2
)

plot_2d_projection(
    X_lda_2d,
    y_train,
    method="LDA"
)

# t-SNE 2D
X_tsne_2d, _ = apply_tsne(
    X_train,
    X_test,
    2
)

plot_2d_projection(
    X_tsne_2d,
    y_train,
    method="t-SNE"
)


print("\n========== All Experiments Finished ==========")
