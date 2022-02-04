import numpy as np


def first_order_moment(i, h, J, beta, gamma):
    """Compute a first-order moment, i.e. <Z_i> for some i.

    Parameters
    ----------
    i: int
        Spin index i.
    h: ndarray
        1D array of linear Hamiltonian couplings (magnetic field in classical Ising).
    J: ndarray
        2D array of quadratic Hamiltonian couplings (spin-spin interaction in classical Ising).
    beta: float
        QAOA beta angle.
    gamma: float
        QAOA gamma angle.

    Returns
    -------
    float
        The first-order moment <Z_i>, where the expectation is for the state prepared by p = 1 QAOA
        with angles beta, gamma.
    """
    return np.sin(2 * beta) * np.sin(2 * gamma * h[i]) * np.product(np.cos(2 * gamma * J[i, :]))


def first_order_moments(h, J, beta, gamma):
    """Compute all first-order moments, i.e. [<Z_0>, <Z_1>, ..., <Z_{n - 1}>] where n is the number
       of qubits.

    Parameters
    ----------
    h: ndarray
        1D array of linear Hamiltonian couplings (magnetic field in classical Ising).
    J: ndarray
        2D array of quadratic Hamiltonian couplings (spin-spin interaction in classical Ising).
    beta: float
        QAOA beta angle.
    gamma: float
        QAOA gamma angle.

    Returns
    -------
    float
        The vector of first-order moments [<Z_0>, <Z_1>, ..., <Z_{n - 1}>], where the expectations are
        for the state prepared by p = 1 QAOA with angles beta, gamma.
    """
    return np.sin(2 * beta) * np.sin(2 * gamma * h) * np.product(np.cos(2 * gamma * J), axis=1)


def second_order_moment(i, j, h, J, beta, gamma):
    """Compute second-order moment <Z_i Z_j> for some (i, j).

    Parameters
    ----------
    i: int
        Spin index i.
    j: int
        Spin index j.
    h: ndarray
        1D array of linear Hamiltonian couplings (magnetic field in classical Ising).
    J: ndarray
        2D array of quadratic Hamiltonian couplings (spin-spin interaction in classical Ising).
    beta: float
        QAOA beta angle.
    gamma: float
        QAOA gamma angle.

    Returns
    -------
    float
        The second-order moment <Z_i Z_j>, where the expectation is for the state prepared by p = 1 QAOA
        with angles beta, gamma.
    """
    n = len(h)
    return 0.5 * np.sin(4 * beta) * np.sin(2 * gamma * J[i, j]) \
        * ( \
           np.cos(2 * gamma * h[i]) * np.product(np.cos(2 * gamma * np.concatenate((J[i, :j], J[i, j + 1:])))) \
           + np.cos(2 * gamma * h[j]) * np.product(np.cos(2 * gamma * np.concatenate((J[j, :i], J[j, i + 1:])))) \
        ) \
        - 0.5 * np.sin(2 * beta) ** 2 * ( \
            np.cos(2 * gamma * (h[i] + h[j])) * np.product(np.cos(2 * gamma * (J[i, :] + J[j, :]))) \
            - np.cos(2 * gamma * (h[i] - h[j])) * np.product(np.cos(2 * gamma * (J[i, :] - J[j, :]))) \
        )

def second_order_moments(h, J, beta, gamma):
    """Compute all second-order moments, i.e. [[1, <Z_0 Z_1>, <Z_0 Z_2>, ..., <Z_0 Z_{n - 1}>], ...,
       [<Z_{n - 1} Z_0>, <Z_{n - 1} Z_1>, ..., <Z_{n - 1} Z_{n - 2}>, 1]] where n is the number
       of qubits.

    Parameters
    ----------
    h: ndarray
        1D array of linear Hamiltonian couplings (magnetic field in classical Ising).
    J: ndarray
        2D array of quadratic Hamiltonian couplings (spin-spin interaction in classical Ising).
    beta: float
        QAOA beta angle.
    gamma: float
        QAOA gamma angle.

    Returns
    -------
    float
        The matrices of second-order moments [[1, <Z_0 Z_1>, <Z_0 Z_2>, ..., <Z_0 Z_{n - 1}>], ...,
        [<Z_{n - 1} Z_0>, <Z_{n - 1} Z_1>, ..., <Z_{n - 1} Z_{n - 2}>, 1]], where the expectations are
        for the state prepared by p = 1 QAOA with angles beta, gamma.
    """
    n = len(h)
    return 0.5 * np.sin(4 * beta) * np.tan(2 * gamma * J) \
        * ( \
           np.cos(2 * gamma * h[:, np.newaxis]) * np.product(np.cos(2 * gamma * J), axis=1)[:, np.newaxis] \
           + np.cos(2 * gamma * h[np.newaxis, :]) * np.product(np.cos(2 * gamma * J), axis=1)[np.newaxis, :] \
        ) \
        - 0.5 * np.sin(2 * beta) ** 2 * ( \
            np.cos(2 * gamma * (h[:, np.newaxis] + h[np.newaxis, :])) * np.product(np.cos(2 * gamma * (J[:, np.newaxis, :] + J[np.newaxis, :, :])), axis=2) \
            - np.cos(2 * gamma * (h[:, np.newaxis] - h[np.newaxis, :])) * np.product(np.cos(2 * gamma * (J[:, np.newaxis, :] - J[np.newaxis, :, :])), axis=2) \
        )


def evaluate(h, J, beta, gamma):
    """Evaluate the expected energy of the state prepared by QAOA, i.e. sum_i h_i <Z_i> + sum_{ij} J_{ij} <Z_i Z_j>,

    Parameters
    ----------
    h: ndarray
        1D array of linear Hamiltonian couplings (magnetic field in classical Ising).
    J: ndarray
        2D array of quadratic Hamiltonian couplings (spin-spin interaction in classical Ising).
    beta: float
        QAOA beta angle.
    gamma: float
        QAOA gamma angle.

    Returns
    -------
    float
        The expected energy of the state prepared by p = 1 QAOA with angles beta, gamma.
    """
    return np.dot(h, first_order_moments(h, J, beta, gamma)) + 0.5 * np.sum(J * second_order_moments(h, J, beta, gamma))
