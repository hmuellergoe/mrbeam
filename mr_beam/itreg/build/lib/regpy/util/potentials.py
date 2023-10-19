import numpy as np


def bell(domain):
    v = domain.zeros()
    r = np.linalg.norm(domain.coords, axis=0)
    v[r < 1] = np.exp(-1 / (1 - r[r < 1]**2))
    return v


def peaks(domain):
    r = np.linalg.norm(domain.coords, axis=0)
    M = mollifier(r)
    X = domain.coords
    f = 3 * (1 - X[0])**2 * np.exp(-X[0]**2 - (X[1] - 1)**2) \
        - 10 * (X[0] / 5 - X[0]**3 - X[1]**5) * np.exp(-r**2) \
        - 1 / 3 * np.exp(-(X[0] + 1)**2 - X[1]**2)
    return f * M


def mollifier(r):
    M = np.zeros(r.shape)
    r_ctf = r < 1
    M[~r_ctf] = 0
    M[r_ctf] = np.exp(1 - 1 / (1 - r[r_ctf]**2))
    return M
