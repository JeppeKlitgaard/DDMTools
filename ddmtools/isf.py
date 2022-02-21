# from numba import njit, prange, jit
from typing import Tuple

import numpy as np

from lmfit.parameter import Parameters


# @njit(parallel=True, fastmath=True)
def array_intermediate_scattering_function(
    tau_c_0_array: np.ndarray,
    alpha_array: np.ndarray,
    beta_array: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    r"""
    This is a single $f_i(q, t) = α_i * \exp(-\frac{t}{{γ_i ⋅ τ_ci})
    """
    # α * exp(-t/(β * τ_c))
    # x1 = β * τ_c
    i_s = beta_array.shape[0]
    j_s = tau_c_0_array.shape[0]

    # x1 = tau_c_0_array * beta_array[:, np.newaxis]
    x1 = tau_c_0_array * beta_array.reshape(-1, 1)

    # x2 = exp(-t/(β * τ_c))
    # x2 = np.exp(-times / x1[:, :, np.newaxis])
    x2 = np.exp(-times / x1.reshape(i_s, j_s, 1))

    # x3 = A * exp(-t(β * τ_c))
    x3 = np.einsum("ji,ijk->ijk", alpha_array, x2)  # Numba nopython incompatible
    # # See: https://stackoverflow.com/questions/65030112/converting-einsum-notation-to-for-loops
    # i_s, j_s, k_s = x2.shape
    # x3 = np.zeros(x2.shape)
    # for i in prange(i_s):
    #     for j in prange(j_s):
    #         for k in prange(k_s):
    #             x3[i, j, k] += alpha_array[j, i] * x2[i, j, k]

    # sum over i's
    F: np.ndarray = np.sum(x3, 0)

    return F


# @njit(parallel=True, fastmath=True)
def array_image_structure_function(
    a_array: np.ndarray,
    b_array: np.ndarray,
    tau_c_0_array: np.ndarray,
    alpha_array: np.ndarray,
    beta_array: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    F = array_intermediate_scattering_function(tau_c_0_array, alpha_array, beta_array, times)

    # I = A * (1 - F) + B
    # y1 =  1 - F
    y1 = 1 - F

    # y2 = A * (1 - F)
    y2 = np.einsum("j,jk->jk", a_array, y1)  # Numba nopython incompatible
    # See: https://stackoverflow.com/questions/65030112/converting-einsum-notation-to-for-loops
    # j_s, k_s = y1.shape
    # y2 = np.zeros(y1.shape)
    # for j in prange(j_s):
    #     for k in prange(k_s):
    #         y2[j, k] = a_array[j] * y1[j, k]

    # I = A * (1 - F) + B
    I: np.ndarray = y2 + b_array.reshape(-1, 1)

    return I


# @njit(fastmath=True)
def wrap_parameters(
    params_flat_arr: np.ndarray, nqs: int, dispersity_order: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = 0

    size = dispersity_order
    beta_array = params_flat_arr[idx : idx + size]
    idx += size

    size = nqs
    a_array = params_flat_arr[idx : idx + size]
    idx += size

    size = nqs
    b_array = params_flat_arr[idx : idx + size]
    idx += size

    size = nqs
    tau_c_0_array = params_flat_arr[idx : idx + size]
    idx += size

    size = nqs * dispersity_order
    alpha_array = params_flat_arr[idx : idx + size]
    alpha_array = alpha_array.reshape(nqs, dispersity_order)
    idx += size

    return (
        a_array,
        b_array,
        tau_c_0_array,
        alpha_array,
        beta_array,
    )


# @jit(forceobj=True, parallel=True, fastmath=True)
def array_image_structure_function_wrapper(
    params: Parameters, iqtaus: np.ndarray, times: np.ndarray, dispersity_order: int
) -> np.ndarray:
    params_flat_arr = np.array(list(params.valuesdict().values()))

    nqs = iqtaus.shape[-1]

    guess = array_image_structure_function(
        *wrap_parameters(params_flat_arr, nqs, dispersity_order),
        times,
    )

    return guess


# @jit(forceobj=True, parallel=True, fastmath=True)
def array_objective(
    params: Parameters, iqtaus: np.ndarray, times: np.ndarray, dispersity_order: int
) -> np.ndarray:
    guess = array_image_structure_function_wrapper(params, iqtaus, times, dispersity_order)

    log_guess = np.log(guess)
    log_experiment = np.log(iqtaus.T)

    residuals = log_experiment - log_guess
    raveled_residuals: np.ndarray = residuals.ravel()

    return raveled_residuals
