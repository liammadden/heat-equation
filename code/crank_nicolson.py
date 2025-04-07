import numpy as np
import scipy.linalg as la


def solve(u0, n_t, r):
    n_x = len(u0)

    ab = np.zeros((2, n_x), dtype=float)
    ab[0, :] = (1 + 2 * r) * np.ones(n_x)
    ab[1, 0 : n_x - 1] = -r * np.ones(n_x - 1)

    u = np.zeros((n_t, n_x), dtype=float)
    u[0, :] = u0

    for i in range(n_t - 1):

        u_now = u[i, :]
        u_next = (1 - 2 * r) * np.array(u_now)
        u_next[0 : n_x - 1] += r * u_now[1:n_x]
        u_next[1:n_x] += r * u_now[0 : n_x - 1]
        u_next = la.solveh_banded(ab, u_next, lower=True)
        u[i + 1, :] = u_next

    return u


def make_data(n_x, n_t, c_x, c_t, r, u_min, u_max, num_samples):
    m_x = c_x * (n_x + 1) - 1
    m_t = c_t * (n_t - 1) + 1

    u = np.zeros((n_t, num_samples, n_x))
    for i in range(num_samples):
        v0 = np.random.uniform(u_min, u_max, m_x)
        v = solve(v0, m_t, r)
        u[:, i, :] = v[:, c_x - 1 :: c_x][::c_t, :]

    return u
