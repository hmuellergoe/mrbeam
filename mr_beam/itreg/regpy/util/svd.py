import random as rd

import numpy as np


def randomized_svd(self, H):
    r = np.linalg.matrix_rank(H)
    N = self.gamma_prior_half.shape[0]
    #    r=N
    X = np.zeros((N, r))
    for i in range(0, N):
        for j in range(0, r):
            X[i, j] = rd.random()

    # second step, compute sample matrix
    Y = np.dot(H, X)

    # third step, QR-decomposition of Y
    Q, R = np.linalg.qr(Y)

    # solve linear system to obtain the matrix B: Q^TY=B(Q^T X)
    B_trans = np.linalg.solve(np.dot(X.transpose(), Q), np.dot(Y.transpose(), Q))
    B = B_trans.transpose()

    # Perform SVD of B
    L, U = np.linalg.eig(B)
    # compute singular vectors
    V = np.dot(Q, U)
    # L are the singular values and V the corresponding singular vectors.
    return L, V


def lanzcos_svd(self, H, number, basis=None, v_1=None):
    #    assert number>1
    #    assert H.transpose()==H
    N = H.shape[0]
    if basis is not None:
        basis = np.eye(N)
    v = np.zeros((N, number))
    w = np.zeros((N, number))
    if v_1 is not None:
        v[:, 0] = np.eye(N)[0, :]
    w_bar = np.zeros((N, number))
    w_bar[:, 0] = np.dot(H, v[:, 0])
    alpha = np.zeros(number)
    alpha[0] = np.dot(w_bar[:, 0], v[:, 0])
    w[:, 0] = w_bar[:, 0] - alpha[0] * v[:, 0]
    beta = np.zeros(number)
    for j in range(1, number):
        beta[j] = np.sqrt(np.vdot(w[:, j], w[:, j]))
        if beta[j] != 0:
            v[:, j] = w[:, j - 1] / beta[j]
        w_bar[:, j] = np.dot(H, v[:, j])
        alpha[j] = np.dot(w_bar[:, j], v[:, j])
        w[:, j] = w_bar[:, j] - alpha[j] * v[:, j] - beta[j] * v[:, j - 1]
    T = np.diag(alpha)
    for i in range(1, number):
        T[i - 1, i] = beta[i]
        T[i, i - 1] = beta[i]
    S, V = np.linalg.eig(T)
    eigenvalues = np.zeros(N)
    eigenvalues[0:number] = S
    return eigenvalues, np.dot(V, v.transpose())
