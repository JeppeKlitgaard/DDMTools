from math import pi

from scipy import constants

from uncertainties import UFloat

from ddmtools.types import IntensiveParameters

k_B: float = constants.Boltzmann


def diffusion_coefficient_calculator(
    diameter: float, viscosity: float, temperature: float
) -> float:
    return k_B * temperature / (3 * pi * viscosity * diameter) * 1.0e12


def diameter_calculator(
    diffusion_coefficient: UFloat, intensive_parameters: IntensiveParameters
) -> UFloat:
    scaling_factor = (
        intensive_parameters.micrometre_per_pixel**2
    ) * intensive_parameters.framerate
    diameter: UFloat = (
        k_B
        * intensive_parameters.temperature
        / (3 * pi * intensive_parameters.viscosity * scaling_factor * diffusion_coefficient)
        * 1.0e12
    )

    return diameter
