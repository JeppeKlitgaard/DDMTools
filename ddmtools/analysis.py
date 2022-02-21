from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.fft
import scipy.optimize

import colorcet as cc
import lmfit
import matplotlib.colors
import statsmodels.api as sm
import uncertainties.unumpy as unp
from joblib import Parallel, delayed
from lmfit.minimizer import MinimizerResult
from lmfit.parameter import Parameters
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numba import njit, objmode
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.regression.rolling import RollingRegressionResults, RollingWLS
from tqdm.auto import tqdm
from typing_extensions import Literal
from uncertainties import ufloat, umath
from uncertainties.core import AffineScalarFunc as UFloat

from ddmtools.eq_utils import diameter_calculator
from ddmtools.image.frame import Framestack
from ddmtools.isf import (
    array_image_structure_function_wrapper,
    array_intermediate_scattering_function,
    array_objective,
    wrap_parameters,
)
from ddmtools.utils import ProgressParallel, log_spaced, pd_nom, pd_sd

CPU_COUNT = os.cpu_count() or 1

# Some of this code is lifted shamelessly from https://github.com/MathieuLeocmach/DDM


@dataclass
class MinimizingResult:
    minimizer_result: MinimizerResult
    param_df: pd.DataFrame
    dispersity_order: int
    iqtaus: np.ndarray
    taus: np.ndarray
    times: np.ndarray
    framerate: float
    micrometre_per_pixel: float
    temperature: float
    viscosity: float

    def tau_to_time(self, tau: int) -> float:
        return tau / self.framerate

    def plot_image_structure_function_params(self) -> Figure:
        ERROR_ALPHA = 0.5

        fig, ax1 = plt.subplots()
        fig.set_size_inches(8, 8)
        fig.set_dpi(120)

        alpha_elements = []
        markers, caps, bars = ax1.errorbar(
            self.param_df["q"],
            pd_nom(self.param_df["A"]),
            pd_sd(self.param_df["A"]),
            fmt="+",
            label="A",
            color=cc.cm.rainbow(0.0),
            capsize=4,
        )
        alpha_elements += caps + bars

        markers, caps, bars = ax1.errorbar(
            self.param_df["q"],
            pd_nom(self.param_df["B"]),
            pd_sd(self.param_df["B"]),
            fmt="+",
            label="B",
            color=cc.cm.rainbow(1.0),
            capsize=4,
        )
        alpha_elements += caps + bars
        [el.set_alpha(ERROR_ALPHA) for el in alpha_elements]

        ax1.set_yscale("log")
        ax1.set_ylabel(r"$A(q),\, B(q)$")

        ax2 = ax1.twinx()
        ax2_ypoints = []
        ax2_alpha_elements = []
        for i in range(self.dispersity_order):
            markers, caps, bars = ax2.errorbar(
                self.param_df["q"],
                pd_nom(self.param_df[f"alpha_{i}"]),
                pd_sd(self.param_df[f"alpha_{i}"]),
                fmt="+",
                label=rf"$\alpha_{i}$",
                color=plt.cm.autumn(i / self.dispersity_order),
                capsize=4,
            )
            ax2_alpha_elements += caps + bars
            ax2_ypoints.extend(pd_nom(self.param_df[f"alpha_{i}"]).values)

        ax2.set_ylim(min(ax2_ypoints) * 0.9, max(ax2_ypoints) * 1.1)
        [el.set_alpha(0.1) for el in ax2_alpha_elements]

        ax2_ylabel = r",\, ".join(rf"\alpha_{i}(q)" for i in range(self.dispersity_order))
        ax2.set_ylabel(f"${ax2_ylabel}$")

        plt.xscale("log")

        fig.legend()

        return fig

    def fit_diffusion_coefficients(
        self,
        fit_fraction: float = 0.75,
        fit_fraction_minimum: float = 0.05,
        fit_fraction_step: float = 0.02,
        minimal_r_squared: float = 0.98,
        reset_interval: int = 3,
    ) -> FitResult:
        log_qs = np.log(self.param_df["q"])

        dispersity_mode_fits: list[DispersityModeFitResult] = []

        for i in range(self.dispersity_order):
            tau_cs = self.param_df[f"tau_c_{i}"]

            log_tau_cs = unp.log(tau_cs)

            # Loop over decreasing fit fractions until we give up or find a good match
            past_attempted_fits: list[dict[str, Any]] = []
            while True:
                logx = log_qs.copy()
                logy = log_tau_cs.copy()
                weights = 1 / (unp.std_devs(logy) ** 2)

                X = logx
                X = sm.add_constant(X)

                window = np.ceil(len(log_qs) * fit_fraction).astype(int)

                with np.errstate(divide="ignore", invalid="ignore"):
                    rolling_wls = RollingWLS(
                        unp.nominal_values(logy), X, window, weights=weights, missing="drop"
                    )
                    rolling_results = rolling_wls.fit(reset=reset_interval)

                signed_rsquared = rolling_results.rsquared * -np.sign(
                    rolling_results.params.iloc[:, 1]
                )

                best_rsq = np.nanmax(signed_rsquared)
                idx_best_rsq = np.nanargmax(signed_rsquared)

                if best_rsq >= minimal_r_squared:
                    break

                if fit_fraction == fit_fraction_minimum:
                    # Recover best past fit
                    best_attempt = sorted(
                        past_attempted_fits, key=lambda fit: fit["best_rsq"], reverse=True
                    )[0]

                    fit_fraction = best_attempt["fit_fraction"]
                    window = best_attempt["window"]
                    rolling_results = best_attempt["result"]
                    best_rsq = best_attempt["best_rsq"]

                    break

                # Save attempted fit
                past_attempted_fits.append(
                    {
                        "fit_fraction": fit_fraction,
                        "window": window,
                        "result": rolling_results,
                        "best_rsq": best_rsq,
                    }
                )

                fit_fraction = max(fit_fraction - fit_fraction_step, fit_fraction_minimum)

            iqmin = idx_best_rsq - window + 1
            iqmax = idx_best_rsq

            # Refit using regular WLS model to get RegressionResult instance
            # in addition to RollingRegressionResult
            # We use this to easily get predictions later
            single_endog = unp.nominal_values(logy[iqmin:iqmax])
            single_x = logx[iqmin:iqmax]
            single_X = sm.add_constant(single_x)
            single_weights = 1 / (unp.std_devs(logy[iqmin:iqmax]) ** 2)
            single_wls = sm.WLS(
                single_endog,
                single_X,
                weights=single_weights,
                missing="drop",
            )
            single_results = single_wls.fit()

            b, a = single_results.params
            b_std_dev, a_std_dev = single_results.bse

            b_uf = ufloat(b, b_std_dev)
            a_uf = ufloat(a, a_std_dev)

            diff_coeff = umath.exp(-b_uf)

            dispersity_mode_fit = DispersityModeFitResult(
                rolling_model=rolling_wls,
                rolling_results=rolling_results,
                model=single_wls,
                results=single_results,
                b=b_uf,
                a=a_uf,
                fit_fraction=fit_fraction,
                window=window,
                r_squared=best_rsq,
                diffusion_coefficient=diff_coeff,
                idx_best_rsq=idx_best_rsq,
                tau_range=(iqmin, iqmax),
            )

            dispersity_mode_fits.append(dispersity_mode_fit)

        fit_result = FitResult(
            minimizer_result=self.minimizer_result,
            param_df=self.param_df,
            dispersity_order=self.dispersity_order,
            iqtaus=self.iqtaus,
            taus=self.taus,
            times=self.times,
            framerate=self.framerate,
            micrometre_per_pixel=self.micrometre_per_pixel,
            temperature=self.temperature,
            viscosity=self.viscosity,
            dispersity_mode_fits=dispersity_mode_fits,
        )

        return fit_result


@dataclass
class DispersityModeFitResult:
    rolling_model: RollingWLS
    rolling_results: RollingRegressionResults
    model: sm.WLS
    results: RegressionResults
    b: UFloat
    a: UFloat
    fit_fraction: float
    window: int
    r_squared: float
    diffusion_coefficient: UFloat
    idx_best_rsq: int
    tau_range: tuple[int, int]


@dataclass
class FitResult(MinimizingResult):
    dispersity_mode_fits: list[DispersityModeFitResult]

    @staticmethod
    def _calculate_delta_t_qs(tau_range: tuple[int, int], num_delta_t_lines: int) -> np.ndarray:
        delta_t_qs = np.logspace(
            np.log10(tau_range[0]), np.log10(tau_range[1]), num_delta_t_lines, base=10
        ).astype(int)

        return delta_t_qs

    def plot_diffusion_coeff_fit(self) -> Figure:
        fig = plt.figure()
        fig.set_size_inches(14, 7)
        fig.set_dpi(100)

        qs = self.param_df["q"]

        colors = list(cc.cm.rainbow(np.linspace(0.0, 1.0, self.dispersity_order)))
        for i in range(self.dispersity_order):
            mode_fit = self.dispersity_mode_fits[i]
            color = colors[i]

            tau_cs = self.param_df[f"tau_c_{i}"]

            # Plot taus and uncertainties
            markers, caps, bars = plt.errorbar(
                qs,
                pd_nom(tau_cs),
                pd_sd(tau_cs),
                color=color,
                fmt="+",
                label=r"$τ_c^{(%s)}$" % i,
                capsize=4,
            )
            [e.set_alpha(0.5) for e in caps + bars]

            # Plot fit
            fit_qs = np.logspace(np.log10(min(qs)), np.log10(max(qs)), 1000)
            fit_log_qs = np.log(fit_qs)
            fit_X = sm.add_constant(fit_log_qs)
            prediction = mode_fit.results.get_prediction(fit_X)
            prediction_summary = prediction.summary_frame(alpha=1.0 - 0.6827)  # 1 sigma

            mean = np.exp(prediction_summary["mean"].values)
            lower = np.exp(prediction_summary["mean_ci_lower"].values)
            upper = np.exp(prediction_summary["mean_ci_upper"].values)

            plt.plot(fit_qs, mean, "-", color=color)
            plt.plot(fit_qs, lower, "--", alpha=0.5, color=color, label=r"$τ_c^{(%s)}±σ$" % i)
            plt.plot(fit_qs, upper, "--", alpha=0.5, color=color)
            plt.fill_between(fit_qs, lower, upper, alpha=0.1, color=color)

        plt.axvspan(
            qs[self.dispersity_mode_fits[1].tau_range[0]],
            qs[self.dispersity_mode_fits[1].tau_range[1]],
            color="black",
            alpha=0.1,
        )

        plt.xscale("log")
        plt.yscale("log")

        plt.ylabel(r"$\tau_c^{(i)}(q)$")
        plt.xlabel(r"$q\,(\mu m^{-1})$")

        plt.xlim(qs[0] ** 0.9)
        plt.ylim(top=max(pd_nom(tau_cs)) ** 1.1)

        plt.title("Diffusion Coefficient, $D$, fit")

        plt.legend()

        return fig

    def plot_image_structure_functions(
        self,
        dispersity_mode: int = 0,
        *,
        q_interval: int = 5,
        num_delta_t_lines: int = 10,
    ) -> Figure:

        times = self.times
        iqtaus = self.iqtaus
        qs = self.param_df["q"]

        # Calculate fitted iqtaus
        fitted_iqtaus = array_image_structure_function_wrapper(
            self.minimizer_result.params, iqtaus, times, self.dispersity_order
        )

        fig = plt.figure()
        fig.set_dpi(300)
        fig.set_size_inches(14, 6)

        norm1 = matplotlib.colors.LogNorm()
        norm1.autoscale(times)

        # Plot ISF for various $q$ as a function of time deltas
        ax1 = plt.subplot(1, 2, 1)
        for i in range(0, len(times), q_interval):
            plt.plot(
                qs,
                iqtaus[i] / 512**2,
                "o",
                color=plt.cm.autumn_r(norm1(times[i])),
            )
            plt.plot(qs, fitted_iqtaus[:, i] / 512**2, "-k")

        cbar1 = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm1, cmap=plt.cm.autumn_r),
            fraction=0.05,
            pad=0.03,
            aspect=50,
        )

        cbar1.ax.get_yaxis().labelpad = 5
        cbar1.ax.set_ylabel(r"$t \ [s]$", rotation=90)

        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel(r"$\mathcal{I}$")
        plt.xlabel(r"$|\vec{q}| \ [μm^{-1}]$")

        ax2 = plt.subplot(1, 2, 2, sharey=ax1)

        delta_t_qs = self._calculate_delta_t_qs(
            self.dispersity_mode_fits[dispersity_mode].tau_range,
            num_delta_t_lines=num_delta_t_lines,
        )

        norm2 = matplotlib.colors.LogNorm()
        norm2.autoscale(qs)

        for i, iq in enumerate(delta_t_qs):
            plt.plot(
                times,
                iqtaus[:, iq] / 512**2,
                "o",
                color=plt.cm.autumn_r(norm2(qs[i])),
            )
            plt.plot(times, fitted_iqtaus[iq] / 512**2, "-k")

        cbar2 = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm2, cmap=plt.cm.autumn_r),
            fraction=0.05,
            pad=0.03,
            aspect=50,
        )
        cbar2.ax.get_yaxis().labelpad = 5
        cbar2.ax.set_ylabel(r"$|\vec{q}| \ [μm^{-1}]$", rotation=90)

        plt.xscale("log")
        plt.xlabel(r"$t \ [s]$")
        plt.setp(ax2.get_yticklabels(), visible=False)

        fig.suptitle(
            r"Image Structure Functions as a function of the scattered wavevector $|\vec{q}|$"
            + "\n"
            + "and time difference $t$.\n"
            + f"Dispersity mode: {dispersity_mode}"
        )

        return fig

    def plot_intermediate_scattering_function(
        self,
        dispersity_mode: int = 0,
        *,
        num_delta_t_lines: int = 10,
    ) -> Figure:
        # TODO: these need to be patched to be intermediate scattering functions
        # when dispersity_mode != 0

        dispersity_fit = self.dispersity_mode_fits[dispersity_mode]

        as_ = pd_nom(self.param_df["A"])
        bs = pd_nom(self.param_df["B"])
        qs = self.param_df["q"]
        times = self.times
        iqtaus = self.iqtaus

        tau_range = dispersity_fit.tau_range
        qs_range = [qs[tau] for tau in tau_range]

        # Calculate fitted and experimental intermediate scattering functions
        fitted_iqtaus = array_image_structure_function_wrapper(
            self.minimizer_result.params, iqtaus, times, self.dispersity_order
        )

        params_flat_arr = np.array(list(self.minimizer_result.params.valuesdict().values()))
        a_array, b_array, tau_c_0_array, alpha_array, beta_array = wrap_parameters(
            params_flat_arr, len(qs), self.dispersity_order
        )

        fitted_fs = array_intermediate_scattering_function(
            tau_c_0_array=tau_c_0_array, alpha_array=alpha_array, beta_array=beta_array, times=times
        )

        experiment_fs = 1 - (fitted_iqtaus - np.array(bs)[:, None]) / np.array(as_)[:, None]

        delta_t_qs = self._calculate_delta_t_qs(
            tau_range=tau_range, num_delta_t_lines=num_delta_t_lines
        )

        fig = plt.figure()
        fig.set_dpi(300)
        fig.set_size_inches(14, 6)

        norm = matplotlib.colors.Normalize()
        norm.autoscale(qs_range)

        ax1 = plt.subplot(1, 2, 1)
        for i, iq in enumerate(delta_t_qs):
            plt.plot(
                times,
                experiment_fs[iq, :],
                "o",
                color=plt.cm.autumn_r(norm(qs[iq])),
            )
            plt.plot(
                times,
                fitted_fs[iq, :],
                "-k",
            )

        cbar1 = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.autumn_r),
            fraction=0.05,
            pad=0.03,
            aspect=50,
        )

        cbar1.ax.get_yaxis().labelpad = 5
        cbar1.ax.set_ylabel(r"$|\vec{q}| \ [μm ^{-1}]$", rotation=90)

        plt.xscale("log")
        plt.ylabel(r"$f(\vec{q},t)$")
        plt.xlabel(r"$t \ [s]$")

        ax2 = plt.subplot(1, 2, 2, sharey=ax1)
        for i, iq in enumerate(delta_t_qs):
            plt.plot(
                qs[iq] ** 2 * times,
                experiment_fs[iq, :],
                "o",
                color=plt.cm.autumn_r(norm(qs[iq])),
            )

        plt.plot(
            qs[dispersity_fit.idx_best_rsq] ** 2 * times,
            fitted_fs[dispersity_fit.idx_best_rsq, :],
            "-k",
        )

        cbar1 = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.autumn_r),
            fraction=0.05,
            pad=0.03,
            aspect=50,
        )

        cbar1.ax.get_yaxis().labelpad = 5
        cbar1.ax.set_ylabel(r"$|\vec{q}| \ [μm ^{-1}]$", rotation=90)

        plt.xscale("log")
        plt.xlabel(r"$t |\vec{q}|^2 \ [s/μm^2]$")
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.ylim(-0.1, 1.1)

        fig.suptitle(
            "Intermediate Scattering as a function of the time difference $t$ "
            + r"and $t|\vec{q}|^2$."
            + "\n"
            + f"Dispersity mode: {dispersity_mode}"
        )

        return fig

    def get_diffusion_coefficients(self) -> list[float]:
        diff_coeffs: list[UFloat] = []
        for mode in self.dispersity_mode_fits:
            diff_coeffs.append(mode.diffusion_coefficient)

        return diff_coeffs

    def get_particle_diameters(self) -> list[float]:
        particle_diameters: list[float] = []
        for mode in self.dispersity_mode_fits:
            particle_diameters.append(
                diameter_calculator(
                    mode.diffusion_coefficient,
                    self.viscosity,
                    self.temperature,
                    self.micrometre_per_pixel,
                    self.framerate,
                )
            )

        return particle_diameters


def _ddm_do_step(
    stack: Framestack, n_tau: int, max_couples: int, radial_average: RadialAverager
) -> np.ndarray:
    time_averaged = time_average(stack, n_tau, max_couples=max_couples)
    radial_averaged = radial_average(time_averaged)

    return radial_averaged


def ddm(
    stack: Framestack,
    n_taus: Sequence[int],
    max_couples: int = 100,
    progress_bar: bool = True,
    workers: int = -1,
):
    radial_average = RadialAverager(stack.shape)

    # Parallise using joblib
    with ProgressParallel(
        n_jobs=workers,
        prefer="threads",
        use_tqdm=progress_bar,
        total=len(n_taus),
    ) as parallel:
        out = parallel(
            delayed(_ddm_do_step)(stack, n_tau, max_couples, radial_average) for n_tau in n_taus
        )

    return np.array(out)


@njit(parallel=True)
def differential_spectrum(frame1: np.ndarray, frame2: np.ndarray, workers: int = 0) -> np.ndarray:
    """
    This performs an FFT on the difference between the two frames.
    """
    # Todo: CuFFT
    if workers == 0:
        workers = CPU_COUNT

    diff: np.ndarray = frame1 - frame2

    with objmode(transformed="complex128[:, :]"):
        transformed = scipy.fft.fft2(diff, overwrite_x=True, workers=workers)

    absed = np.abs(transformed)
    squared = np.square(absed)

    return squared


def time_average(
    stack: Framestack, n_tau: int, max_couples: int = 300, workers: int = 0
) -> np.ndarray:
    if workers == 0:
        workers = CPU_COUNT

    # How many frames to increment by
    increment = max([(len(stack) - n_tau) // max_couples, 1])

    # Precompute all initial times
    initial_times = np.arange(0, len(stack) - n_tau, increment)

    # Parallise using joblib
    sums = Parallel(n_jobs=workers, prefer="threads")(
        delayed(differential_spectrum)(stack[t], stack[t + n_tau]) for t in initial_times
    )

    avg_fft = np.sum(sums, axis=0)

    return avg_fft / len(initial_times)


class RadialAverager(object):
    """Radial average of a 2D array centred on (0,0), like the result of fft2d."""

    def __init__(self, shape: tuple[int, int]) -> None:
        """
        A RadialAverager instance can process only arrays of a given shape,
        fixed at instanciation.
        """
        self.shape = shape

        if len(shape) != 2:
            raise ValueError("Invalid shape.")

        # Calculate a matrix of distances in frequency space
        self.dists = np.sqrt(
            np.fft.fftfreq(shape[0])[:, None] ** 2 + np.fft.fftfreq(shape[1])[None, :] ** 2
        )

        # Dump the cross
        self.dists[0] = 0
        self.dists[:, 0] = 0

        # Discretize distances into bins
        self.bins = np.arange(max(shape) // 2 + 1) / float(max(shape))

        # Number of pixels at each distance
        self.pixel_density = np.histogram(self.dists, self.bins)[0]

    def __call__(self, spectrum: np.ndarray) -> np.ndarray:
        """Perform and return the radial average of the spectrum"""
        assert spectrum.shape == self.dists.shape

        hw = np.histogram(self.dists, self.bins, weights=spectrum)[0]

        return hw / self.pixel_density


class DDM:
    T_MAX: int = -6

    def __init__(
        self,
        stack: Framestack,
        framerate: float,
        temperature: float,
        viscosity: float,
        micrometre_per_pixel: float,
        workers: Optional[int] = None,
    ) -> None:
        self.stack = stack
        self.framerate = framerate
        self.temperature = temperature
        self.viscosity = viscosity
        self.micrometre_per_pixel = micrometre_per_pixel

        self.taus: Optional[np.ndarray] = None
        self.iqtaus: Optional[np.ndarray] = None
        self.tau_range: Optional[tuple[int, int]] = None

        self.workers: int = workers or os.cpu_count()

        self.radial_averager = RadialAverager(stack.shape)

    def _require_attr(self, attr: str) -> None:
        if getattr(self, attr) is None:
            msg = (
                "DDM object is in illegal state for requested operation. " f"`{attr}`` must be set."
            )

            raise ValueError(msg)

    @classmethod
    def from_folder(
        cls,
        folder: Path,
        framerate: float,
        micrometre_per_pixel: float,
        temperature: float,
        viscosity: float,
        glob_pattern: str = "*.pgm",
    ) -> DDM:

        stack = Framestack.from_folder(folder, glob_pattern=glob_pattern)

        return cls(
            stack=stack,
            framerate=framerate,
            micrometre_per_pixel=micrometre_per_pixel,
            temperature=temperature,
            viscosity=viscosity,
        )

    def get_differential_spectrum(self, idx1: int, idx2: int) -> np.ndarray:
        diff = differential_spectrum(self.stack[idx1], self.stack[idx2], workers=self.workers)

        return diff

    def plot_differential_spectrum(
        self, differential_spectrum: np.ndarray, brightness: float = 1.0
    ) -> Figure:
        fig = plt.figure()
        plt.imshow(
            scipy.fft.fftshift(differential_spectrum),
            "hot",
            vmin=0.0,
            vmax=differential_spectrum.max() / brightness,
        )

        return fig

    def get_time_average(self, n_tau: int, *, max_couples: int = 50) -> np.ndarray:
        average = time_average(
            self.stack, n_tau=n_tau, max_couples=max_couples, workers=self.workers
        )

        return average

    def plot_time_average(self, average: np.ndarray, brightness: float = 1.0) -> Figure:
        fig = plt.figure()
        plt.imshow(
            scipy.fft.fftshift(average),
            "hot",
            vmin=0.0,
            vmax=average.max() / brightness,
        )

        return fig

    def get_radial_average(self, matrix: np.ndarray) -> np.ndarray:
        if self.radial_averager.shape != self.stack.shape:
            self.radial_averager = RadialAverager(self.stack.shape)

        average = self.radial_averager(matrix)

        return average

    def plot_radial_average(self, average_array: np.ndarray) -> Figure:
        fig = plt.Figure()
        fig.set_dpi(150)

        plt.plot(average_array)

        plt.ylabel("Intensity, arb. unit")
        plt.xlabel("Radial Distance, px")

        plt.xscale("log")
        plt.yscale("log")

        plt.title("Radial Average")

        return fig

    def plot_image_structure_function(self, n_q: int) -> Figure:
        self._require_attr("iqtaus")
        self._require_attr("taus")

        fig = plt.figure()
        fig.set_dpi(150)
        fig.set_size_inches(10, 6)

        qs = self.iqtaus_to_qs(self.iqtaus)

        times = self.taus_to_times(self.taus)
        isfs = self.iqtaus[:, n_q]

        plt.plot(times, isfs, "o")

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Time [s]")
        plt.ylabel("Image Structure Function [arb. unit]")

        plt.title(
            f"Time-Dependent Image Structure Function at $|q| = {qs[n_q]:.2f}\\, " + "[μm^{-1}]$"
        )

        return fig

    def get_log_spaced_taus(self, taus_per_decade: int = 25) -> Sequence[int]:
        return log_spaced(len(self.stack), taus_per_decade)

    def tau_to_time(self, tau: int) -> float:
        return tau / self.framerate

    def taus_to_times(self, taus: np.ndarray) -> np.ndarray:
        return taus / self.framerate

    def iqtaus_to_qs(self, iqtaus: Optional[np.ndarray] = None) -> np.ndarray:
        if iqtaus is None:
            iqtaus = self.iqtaus

        nqs = iqtaus.shape[-1]
        qs = 2 * np.pi / (2 * nqs * self.micrometre_per_pixel) * np.arange(1, nqs + 1)

        return qs

    # TODO: Change to run_ddm
    def analyse(
        self,
        taus: Sequence[int],
        *,
        max_couples: int = 50,
        progress_bar: bool = True,
        workers: int = -1,
    ) -> np.ndarray:
        iqtaus = ddm(
            self.stack, taus, max_couples=max_couples, progress_bar=progress_bar, workers=workers
        )

        self.taus = taus
        self.iqtaus = iqtaus

        return iqtaus

    # Post analyse tools

    @staticmethod
    def _loop_intermediate_scattering_function(
        params: Parameters, times: np.ndarray, j: int, i: int
    ):
        r"""
        This is a single $f_i(q, t) = α_i * \exp(-\frac{t}{{γ_i ⋅ τ_ci})
        """
        tau_c_0 = params[f"tau_c_{j}_0"]

        alpha_i = params[f"alpha_{j}_{i}"]
        beta_i = params[f"beta_{i}"]

        f_i = alpha_i * np.exp(-times / (beta_i * tau_c_0))

        return f_i

    @staticmethod
    def _loop_image_structure_function(
        params: Parameters,
        times: np.ndarray,
        j: int,
        dispersity_order: int,
    ):

        A = params[f"A_{j}"]
        B = params[f"B_{j}"]

        F = np.zeros(times.shape)

        for i in range(dispersity_order):
            F += DDM._loop_intermediate_scattering_function(params, times, j, i)

        I = A * (1 - F) + B

        return I

    @staticmethod
    def _loop_objective(
        params: Parameters, iqtaus: np.ndarray, times: np.ndarray, dispersity_order
    ):
        residuals = 0.0 * iqtaus.T

        for j, iqtau in enumerate(iqtaus.T):
            # j is the idx of the q we are using

            experimental_data = iqtau.T
            log_experimental_data = np.log(experimental_data)

            guess = DDM._loop_image_structure_function(params, times, j, dispersity_order)
            log_guess = np.log(guess)

            residuals[j, :] = log_experimental_data - log_guess

        return residuals.T.flatten()

    @staticmethod
    def _minimizer_result_to_df(minimizer_result: MinimizerResult) -> pd.DataFrame:
        pattern = re.compile(r"^([a-zA-Z_]+)_([0-9]+)?(.*)$")

        if not minimizer_result.errorbars:
            raise ValueError("Was unable to estimate errors. Likely not enough data.")

        data_dict: dict[str, ufloat] = {}
        beta_dict: dict[str, ufloat] = {}
        for param in minimizer_result.params:
            if param.startswith("beta_"):
                beta_dict[param] = minimizer_result.params[param]
                continue

            match = pattern.fullmatch(param)
            assert match is not None

            idx = int(match.group(2))
            new_name = match.group(1) + match.group(3)

            if new_name not in data_dict:
                data_dict[new_name] = []

            assert idx == len(data_dict[new_name])

            resolved_param = minimizer_result.params[match.group(0)]
            param_uf = ufloat(resolved_param.value, resolved_param.stderr)
            data_dict[new_name].append(param_uf)

        df = pd.DataFrame.from_dict(data_dict)

        for beta_name, beta in beta_dict.items():
            i = int(beta_name.removeprefix("beta_"))
            if i == 0:
                continue

            beta_uf = ufloat(beta.value, beta.stderr)

            df[f"tau_c_{i}"] = df["tau_c_0"] / beta_uf

        return df

    def fit_image_structure_functions_polydisperse(
        self,
        dispersity_order: int,
        iqtaus: Optional[np.ndarray] = None,
        taus: Optional[np.ndarray] = None,
        method_sequence: Optional[list[str]] = None,
        objective_method: Literal["loop", "array"] = "array",
        max_nfev: Optional[int] = None,
        show_progress: bool = True,
    ) -> MinimizingResult:
        # DOF = (n-1) + q(1+1+1+(n-1))
        # DOF = -1 + n + q(2+n)
        # DOF = -1 + n + +2q + qn
        # DOF = qn + 2q + n -1
        # q = 64, n = 1 => DOF = 192
        # q = 64, n = 2 => DOF = 257
        # q = 64, n = 3 => DOF = 322
        # q = 64, n = 4 => DOF = 387
        # q = 64, n = 5 => DOF = 452
        # q = 64, n = 6 => DOF = 517

        if not method_sequence:
            method_sequence = ["leastsq"]

        if not iqtaus:
            self._require_attr("iqtaus")
            iqtaus = self.iqtaus

        if not taus:
            self._require_attr("taus")
            taus = self.taus

        iqtaus = iqtaus[: self.T_MAX]  # Don't fit last 6
        times = self.taus_to_times(taus[: self.T_MAX])

        fit_params = lmfit.Parameters()

        # NOTE: ORDER MATTERS FOR UNRAVELING!
        # Must come in order:
        # betas
        # As
        # Bs
        # tau_c_0s
        # alphas

        ### betas
        # Make one beta per dispersity order
        # with first beta constrained to one
        for i in range(dispersity_order):
            # WLOG first tau_c_j_0 can be largest tau due to commutative property
            fit_params.add(f"beta_{i}", value=0.5**i, min=0.0, max=1.0)

        fit_params["beta_0"].expr = "1"

        ### As
        for j, iqtau in enumerate(iqtaus.T):
            # A, optical transfer
            fit_params.add(f"A_{j}", value=iqtau.ptp(), min=0.0)

        ### Bs
        for j, iqtau in enumerate(iqtaus.T):
            # B, noise
            fit_params.add(f"B_{j}", value=iqtau.min(), min=0.0)

        ### tau_c_0s
        for j, iqtau in enumerate(iqtaus.T):
            fit_params.add(f"tau_c_{j}_0", value=1.0, min=0.0, max=100.0)

        ### alphas
        for j, iqtau in enumerate(iqtaus.T):
            # j is the idx of the q we are using
            for i in range(dispersity_order):
                # i is the idx of the dispersity
                for i in range(dispersity_order):
                    fit_params.add(
                        f"alpha_{j}_{i}",
                        value=1.0 / dispersity_order,
                        min=0.0,
                        max=1.0,
                    )

                other_alphas = [f"alpha_{j}_{i}" for i in range(dispersity_order) if i != 0]
                fit_params[f"alpha_{j}_0"].expr = f"1-{'-'.join(['0', *other_alphas])}"

        if objective_method == "loop":
            objective = self._loop_objective

        elif objective_method == "array":
            objective = array_objective
        else:
            raise ValueError("Bad argument.")

        # TQDM progress bar
        nfev_total = max_nfev * len(method_sequence) if max_nfev is not None else None
        pbar = tqdm(
            total=nfev_total,
            disable=not show_progress,
            mininterval=0.5,
            miniters=10,
            smoothing=0.05,
        )

        def iter_callback(params, iter, resid, *args, **kws):
            pbar.update()

        for k, method in enumerate(method_sequence):
            print(f"Doing fit {k+1}/{len(method_sequence)} using method `{method}`...")

            fit = lmfit.minimize(
                objective,
                fit_params,
                method=method,
                nan_policy="propagate",
                args=(iqtaus, times, dispersity_order),
                iter_cb=iter_callback,
                max_nfev=max_nfev,
            )

            fit_params = fit.params

        pbar.close()

        df = self._minimizer_result_to_df(fit)
        df["q"] = self.iqtaus_to_qs(iqtaus)

        print("Done.")

        result = MinimizingResult(
            minimizer_result=fit,
            param_df=df,
            dispersity_order=dispersity_order,
            iqtaus=iqtaus,
            taus=taus,
            framerate=self.framerate,
            micrometre_per_pixel=self.micrometre_per_pixel,
            temperature=self.temperature,
            viscosity=self.viscosity,
            times=times,
        )

        return result

    @staticmethod
    def _log_isf_monodisperse(parameters: Sequence[Any], taus: Sequence[float]) -> np.ndarray:
        r"""
        $$
        I(q, τ) = A(q) * \qty[1 - f(q, τ)] + B(q)
        $$

        Where, for Brownian Motion:
        $$
        f(q, τ) = \exp{\frac{-τ}{τ_c(q)}}
        $$

        We fit one Fourier Mode at a time.

        Our parameters are:
        [0]: A
        [1]: B
        [2]: tau_c

        """
        A = parameters[0]
        B = parameters[1]
        tau_c = parameters[2]

        f = np.exp(-taus / tau_c)

        I = A * (1 - f) + B

        return np.log(I)

    @staticmethod
    def _isf_fitter(parameters: Sequence[Any], taus: Sequence[float], log_iqtau: Any) -> Any:
        x = DDM._log_isf_monodisperse(parameters, taus) - log_iqtau

        return x

    # TODO: Refactor
    def fit_image_structure_functions_monodisperse(self) -> tuple[np.ndarray, np.ndarray]:
        self._require_attr("iqtaus")
        self._require_attr("taus")

        nqs = self.iqtaus.shape[-1]
        times = self.taus_to_times(self.taus)

        params = np.zeros([nqs, 3])  # 3 params
        fitting_matrix = np.zeros(self.iqtaus[: self.T_MAX].T.shape)

        for iq, iqtau in enumerate(self.iqtaus[: self.T_MAX].T):
            params[iq] = scipy.optimize.leastsq(
                self._isf_fitter,
                [iqtau.ptp(), iqtau.min(), 1],
                args=(times[: self.T_MAX], np.log(iqtau)),
            )[0]

            fitting_matrix[iq] = np.exp(self._log_isf_monodisperse(params[iq], times[: self.T_MAX]))

        self.isf_fits = fitting_matrix
        self.isf_params = params

        # Return last params for inspection
        return (fitting_matrix, params)
