from dataclasses import dataclass

@dataclass()
class ConfigTime:
    time_domain: tuple[int] = (0, 0.02)
    timesteps: int = 40
    substeps: int = 1

    dt: float = None

    def __post_init__(self):
        self.dt = (self.time_domain[1] - self.time_domain[0]) / self.timesteps / self.substeps



