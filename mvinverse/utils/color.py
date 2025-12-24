import numpy as np
import torch

def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """
    sRGB -> Linear RGB (approx. gamma 2.2)
    """
    srgb = np.clip(srgb, 0.0, 1.0)
    return srgb ** 2.2

def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """
    Linear RGB -> sRGB (approx. gamma 2.2)
    """
    linear = np.clip(linear, 0.0, 1.0)
    return linear ** (1 / 2.2)

def srgb_to_linear_tensor(srgb: torch.Tensor) -> torch.Tensor:
    """
    sRGB -> Linear RGB (approx. gamma 2.2)
    """
    srgb = torch.clamp(srgb, 0.0, 1.0)
    return srgb ** 2.2

def linear_to_srgb_tensor(linear: torch.Tensor) -> torch.Tensor:
    """
    Linear RGB -> sRGB (approx. gamma 2.2)
    """
    linear = torch.clamp(linear, 0.0, 1.0)
    return linear ** (1 / 2.2)
