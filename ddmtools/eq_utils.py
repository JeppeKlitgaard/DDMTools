from math import pi

from scipy import constants

k_B: float = constants.Boltzmann


def diffusion_coefficient_calculator(
    diameter: float, viscosity: float, temperature: float
) -> float:
    return k_B * temperature / (3 * pi * viscosity * diameter) * 1.0e12


def diameter_calculator(
    diffusion_coefficient: float,
    viscosity: float,
    temperature: float,
    micrometre_per_pixel: float,
    framerate: float,
) -> float:
    scaling_factor = (micrometre_per_pixel**2) * framerate
    diameter = (
        k_B * temperature / (3 * pi * viscosity * scaling_factor * diffusion_coefficient) * 1.0e12
    )

    return diameter
