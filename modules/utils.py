import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Set seed for reproducibility across Python, NumPy, and PyTorch (CPU & CUDA).
    Provides controlled randomness without forcing strict determinism
    (avoids CuBLAS workspace errors).
    """
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (CPU & CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Make cuDNN use deterministic convolution algorithms when available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: enforce deterministic hash (helps with Python set/dict randomness)
    os.environ["PYTHONHASHSEED"] = str(seed)

