from dataclasses import dataclass

from uncertainties import UFloat

from ddmtools.utils.uncertainty import FUF, as_ufloat


@dataclass(init=False)
class IntensiveParameters:
    framerate: UFloat
    micrometre_per_pixel: UFloat
    temperature: UFloat
    viscosity: UFloat

    def __init__(self, framerate: FUF, micrometre_per_pixel: FUF, temperature: FUF, viscosity: FUF):
        self.framerate = as_ufloat(framerate)
        self.micrometre_per_pixel = as_ufloat(micrometre_per_pixel)
        self.temperature = as_ufloat(temperature)
        self.viscosity = as_ufloat(viscosity)
