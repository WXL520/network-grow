import sys
import numpy as np


# # CCA

def cca_decomp(A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None):
    """Computes CCA vectors, correlations, and transformed matrices
    requires a < n and b < n
    Args:
        A: np.array of size a x n where a is the number of neurons and n is the dataset size
        B: np.array of size b x n where b is the number of neurons and n is the dataset size
    Returns:
        u: left singular vectors for the inner SVD problem
        s: canonical correlation coefficients
        vh: right singular vectors for the inner SVD problem
        transformed_a: canonical vectors for matrix A, a x n array
        transformed_b: canonical vectors for matrix B, b x n array
    """
    assert A.shape[0] <= A.shape[1]
    assert B.shape[0] <= B.shape[1]

    if evals_a is None or evecs_a is None:
        evals_a, evecs_a = np.linalg.eigh(A @ A.T)
    if evals_b is None or evecs_b is None:
        evals_b, evecs_b = np.linalg.eigh(B @ B.T)

    evals_a = (evals_a + np.abs(evals_a)) / 2
    inv_a = np.array([1 / np.sqrt(x) if x > 0 else 0 for x in evals_a])

    evals_b = (evals_b + np.abs(evals_b)) / 2
    inv_b = np.array([1 / np.sqrt(x) if x > 0 else 0 for x in evals_b])

    cov_ab = A @ B.T

    temp = (
            (evecs_a @ np.diag(inv_a) @ evecs_a.T)
            @ cov_ab
            @ (evecs_b @ np.diag(inv_b) @ evecs_b.T)
    )

    try:
        u, s, vh = np.linalg.svd(temp)
    except:
        u, s, vh = np.linalg.svd(temp * 100)
        s = s / 100

    transformed_a = (u.T @ (evecs_a @ np.diag(inv_a) @ evecs_a.T) @ A).T
    transformed_b = (vh @ (evecs_b @ np.diag(inv_b) @ evecs_b.T) @ B).T
    return u, s, vh, transformed_a, transformed_b


def mean_sq_cca_corr(rho):
    """Compute mean squared CCA correlation
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    """
    # len(rho) is min(A.shape[0], B.shape[0])
    return 1 - np.sum(rho * rho) / len(rho)


def mean_cca_corr(rho):
    """Compute mean CCA correlation
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    """
    # len(rho) is min(A.shape[0], B.shape[0])
    return 1 - np.sum(rho) / len(rho)


def pwcca_dist(A, rho, transformed_a):
    """Computes projection weighted CCA distance between A and B given the correlation
    coefficients rho and the transformed matrices after running CCA
    :param A: np.array of size a x n where a is the number of neurons and n is the dataset size
    :param B: np.array of size b x n where b is the number of neurons and n is the dataset size
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    :param transformed_a: canonical vectors for A returned by cca_decomp(A,B)
    :param transformed_b: canonical vectors for B returned by cca_decomp(A,B)
    :return: PWCCA distance
    """
    in_prod = transformed_a.T @ A.T
    weights = np.sum(np.abs(in_prod), axis=1)
    weights = weights / np.sum(weights)
    dim = min(len(weights), len(rho))
    return 1 - np.dot(weights[:dim], rho[:dim])


# # CKA


def lin_cka_dist(A, B):
    """
    Computes Linear CKA distance bewteen representations A and B
    """
    similarity = np.linalg.norm(B @ A.T, ord="fro") ** 2
    normalization = np.linalg.norm(A @ A.T, ord="fro") * np.linalg.norm(
        B @ B.T, ord="fro"
    )
    return 1 - similarity / normalization


def lin_cka_prime_dist(A, B):
    """
    Computes Linear CKA prime distance bewteen representations A and B
    The version here is suited to a, b >> n
    """
    if A.shape[0] > A.shape[1]:
        At_A = A.T @ A  # O(n * n * a)
        Bt_B = B.T @ B  # O(n * n * a)
        numerator = np.sum((At_A - Bt_B) ** 2)
        denominator = np.sum(A ** 2) ** 2 + np.sum(B ** 2) ** 2
        return numerator / denominator
    else:
        similarity = np.linalg.norm(B @ A.T, ord="fro") ** 2
        denominator = np.sum(A ** 2) ** 2 + np.sum(B ** 2) ** 2
        return 1 - 2 * similarity / denominator


# # Procrustes


def procrustes(A, B):
    """
    Computes Procrustes distance bewteen representations A and B
    """
    n = A.shape[1]
    A_sq_frob = np.sum(A ** 2) / n
    B_sq_frob = np.sum(B ** 2) / n
    nuc = np.linalg.norm(A @ B.T, ord="nuc") / n  # O(p * p * n)
    return A_sq_frob + B_sq_frob - 2 * nuc


# # Our predictor distances

def predictor_dist(A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None, lmbda=0):
    """
    Computes distance bewteen best linear predictors on representations A and B
    """
    k, n = A.shape
    l, _ = B.shape
    assert k <= n
    assert l <= n

    if evals_a is None or evecs_a is None:
        evals_a, evecs_a = np.linalg.eigh(A @ A.T)
    if evals_b is None or evecs_b is None:
        evals_b, evecs_b = np.linalg.eigh(B @ B.T)

    evals_a = (evals_a + np.abs(evals_a)) / (2 * n)
    if lmbda > 0:
        inv_a_lmbda = np.array([1 / (x + lmbda) if x > 0 else 1 / lmbda for x in evals_a])
    else:
        inv_a_lmbda = np.array([1 / x if x > 0 else 0 for x in evals_a])

    evals_b = (evals_b + np.abs(evals_b)) / (2 * n)
    if lmbda > 0:
        inv_b_lmbda = np.array([1 / (x + lmbda) if x > 0 else 1 / lmbda for x in evals_b])
    else:
        inv_b_lmbda = np.array([1 / x if x > 0 else 0 for x in evals_b])

    T1 = np.sum(np.square(evals_a * inv_a_lmbda))
    T2 = np.sum(np.square(evals_b * inv_b_lmbda))

    cov_ab = A @ B.T / n
    T3 = np.trace(
        (np.diag(np.sqrt(inv_a_lmbda)) @ evecs_a.T)
        @ cov_ab
        @ (evecs_b @ np.diag(inv_b_lmbda) @ evecs_b.T)
        @ cov_ab.T
        @ (evecs_a @ np.diag(np.sqrt(inv_a_lmbda)))
    )

    return T1 + T2 - 2 * T3


def squared_procrustes(A, B):
    """
    Computes distance bewteen best linear predictors on representations A and B
    """
    k, n = A.shape
    l, _ = B.shape
    assert k < n
    assert l < n

    cov_a = A @ A.T / (n - 1)
    cov_b = B @ B.T / (n - 1)
    cov_ab = A @ B.T / (n - 1)

    T1 = np.trace(cov_a @ cov_a)
    T2 = np.trace(cov_b @ cov_b)
    T3 = np.trace(cov_ab @ cov_ab.T)

    return T1 + T2 - 2 * T3


import pathlib
import pandas as pd
import numpy as np
import pickle as pkl
import numpy as np
# from icecream import ic
import sys
import os
from sklearn.manifold import TSNE, MDS
# import umap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def symmetrize(A):
    n = A.shape[0]
    B = A.copy()
    B[np.tril_indices(n)] = B.T[np.tril_indices(n)]
    return B


if __name__ == '__main__':
    # lmbdas = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    # num_lmbdas = len(lmbdas)
    # METRICS = ["Procrustes", "CKA", "PWCCA", "mean_cca_corr", "mean_sq_cca_corr"] + ["PredDist_" + str(lmbdas[i]) for i
    #                                                                                  in range(num_lmbdas)]
    # print(METRICS)  # 啥意思啊这些metrics？我只考虑CKA就够了吧

    METRICS = ["Procrustes", "CKA", "PWCCA", "mean_cca_corr", "mean_sq_cca_corr"]

    num_layers = 12
    seeds_list = range(1, 11)
    num_seeds = len(seeds_list)
    full_df = pd.read_csv("full_df_self_computed.csv")  # 先跑跑别人下载的bert表示吧

    # mean_sq_cca = np.zeros((num_seeds, num_seeds, num_layers, num_layers))
    # mean_sq_cca[:] = np.nan
    # mean_cca = np.zeros((num_seeds, num_seeds, num_layers, num_layers))
    # mean_cca[:] = np.nan
    # pwcca = np.zeros((num_seeds, num_seeds, num_layers, num_layers))
    # pwcca[:] = np.nan
    lin_cka = np.zeros((num_seeds, num_seeds, num_layers, num_layers))
    lin_cka[:] = np.nan
    # procrustes_dist = np.zeros((num_seeds, num_seeds, num_layers, num_layers))
    # procrustes_dist[:] = np.nan
    # pred_dists = np.zeros((num_seeds, num_seeds, num_layers, num_layers, num_lmbdas))
    # pred_dists[:] = np.nan
    for seed1 in range(num_seeds):
        for seed2 in range(seed1, num_seeds):
            for layer1 in range(num_layers):
                for layer2 in range(num_layers):
                    if seed1 == seed2 and layer1 > layer2:
                        pass
                    else:
                        # print((full_df["seed1"] == seed1 + 1) & (full_df["seed2"] == seed2 + 1) & (
                        #             full_df["layer1"] == layer1) & (full_df["layer2"] == layer2))
                        sub_df = full_df[(full_df["seed1"] == seed1 + 1) & (full_df["seed2"] == seed2 + 1) & (
                                    full_df["layer1"] == layer1) & (full_df["layer2"] == layer2)]


                        # mean_sq_cca[seed1, seed2, layer1, layer2] = sub_df["mean_sq_cca_corr"]
                        # mean_cca[seed1, seed2, layer1, layer2] = sub_df["mean_cca_corr"]
                        # pwcca[seed1, seed2, layer1, layer2] = sub_df["PWCCA"]
                        # print('begin')
                        # print(sub_df["CKA"])
                        lin_cka[seed1, seed2, layer1, layer2] = sub_df["CKA"]
                        # print(lin_cka.shape)
                        #
                        # print('end')
                        # procrustes_dist[seed1, seed2, layer1, layer2] = sub_df["Procrustes"]
                        # pred_dists[seed1, seed2, layer1, layer2, :] = sub_df[METRICS[5:]]


    # rs = np.random.randint(0, 200, num_seeds) / 256  # 这是用于作者的方法的吗？
    # gs = np.random.randint(0, 200, num_seeds) / 256
    # bs = np.random.randint(0, 200, num_seeds) / 256
    # t = 0.6

    cmap = plt.cm.tab10  # 我只要一个图？只画CKA？
    colors = [cmap(i) for i in range(num_seeds)]
    # ls = [1, 3, 5, 7, 9]
    # metric_names = ["GULP λ = 1e-7", "GULP λ = 1e-5", "GULP λ = 1e-3", "GULP λ = 1e-1", "GULP λ = 1e1"]

    fig, axs = plt.subplots(6, 2, figsize=(25, 5))

    for k in range(1):  # 5列，挨个儿画，我先画一列吧
        print(lin_cka)
        D = np.transpose(lin_cka[:, :, :, :], (0, 2, 1, 3))  # pred_dists要改成CKA预测出来的表示
        D = np.reshape(D, (num_seeds * num_layers, num_seeds * num_layers))
        D = symmetrize(D)  # 这啥意思啊？
        D = np.maximum(D, 0)

        ax = axs[0, k]
        im = ax.pcolormesh(D)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_title("CKA")
        ax.set_xticks(np.linspace(num_layers / 2, num_layers * (num_seeds - 1 / 2), num_seeds))
        ax.set_xticklabels(seeds_list)
        ax.set_xlabel("Networks")
        ax.set_yticks(np.linspace(num_layers / 2, num_layers * (num_seeds - 1 / 2), num_seeds))
        ax.set_yticklabels(seeds_list)
        ax.set_ylabel("Networks")

        X_embedded_TSNE = TSNE(n_components=2, perplexity=30.0, init="random", metric="precomputed").fit_transform(D)
        ax = axs[1, k]
        for i in range(num_seeds):
            vals = np.ones((256, 4))
            vals[:, 0] = np.linspace(colors[i][0], t + (1 - t) * colors[i][0], 256)
            vals[:, 1] = np.linspace(colors[i][1], t + (1 - t) * colors[i][1], 256)
            vals[:, 2] = np.linspace(colors[i][2], t + (1 - t) * colors[i][2], 256)
            cmap = ListedColormap(vals)
            ax.plot(X_embedded_TSNE[num_layers * i:num_layers * (i + 1), 0],
                    X_embedded_TSNE[num_layers * i:num_layers * (i + 1), 1], c=colors[i], alpha=0.2)
            ax.scatter(X_embedded_TSNE[num_layers * i:num_layers * (i + 1), 0],
                       X_embedded_TSNE[num_layers * i:num_layers * (i + 1), 1], c=np.arange(num_layers), cmap=cmap)
        ax.set_title("TSNE plot colored by network")

        X_embedded_MDS = MDS(n_components=2, dissimilarity="precomputed").fit_transform(np.sqrt(D))
        ax = axs[2, k]
        for i in range(num_seeds):
            vals = np.ones((256, 4))
            vals[:, 0] = np.linspace(colors[i][0], t + (1 - t) * colors[i][0], 256)
            vals[:, 1] = np.linspace(colors[i][1], t + (1 - t) * colors[i][1], 256)
            vals[:, 2] = np.linspace(colors[i][2], t + (1 - t) * colors[i][2], 256)
            cmap = ListedColormap(vals)
            ax.plot(X_embedded_MDS[num_layers * i:num_layers * (i + 1), 0],
                    X_embedded_MDS[num_layers * i:num_layers * (i + 1), 1], c=colors[i], alpha=0.2)
            ax.scatter(X_embedded_MDS[num_layers * i:num_layers * (i + 1), 0],
                       X_embedded_MDS[num_layers * i:num_layers * (i + 1), 1], c=np.arange(num_layers), cmap=cmap)
        ax.set_title("MDS plot colored by network")

        X_embedded_UMAP = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1).fit_transform(np.sqrt(D))
        ax = axs[3, k]
        for i in range(num_seeds):
            vals = np.ones((256, 4))
            vals[:, 0] = np.linspace(colors[i][0], t + (1 - t) * colors[i][0], 256)
            vals[:, 1] = np.linspace(colors[i][1], t + (1 - t) * colors[i][1], 256)
            vals[:, 2] = np.linspace(colors[i][2], t + (1 - t) * colors[i][2], 256)
            cmap = ListedColormap(vals)
            ax.plot(X_embedded_UMAP[num_layers * i:num_layers * (i + 1), 0],
                    X_embedded_UMAP[num_layers * i:num_layers * (i + 1), 1], c=colors[i], alpha=0.2)
            ax.scatter(X_embedded_UMAP[num_layers * i:num_layers * (i + 1), 0],
                       X_embedded_UMAP[num_layers * i:num_layers * (i + 1), 1], c=np.arange(num_layers), cmap=cmap)
        ax.set_title("UMAP plot colored by network")
    fig.tight_layout()
    plt.savefig("../paper_figures/bert_layer_embedding.pdf")
    plt.show()


