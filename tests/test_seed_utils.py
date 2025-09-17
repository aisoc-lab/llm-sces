import os
import random
import numpy as np
import torch
import pytest
from modules.utils import set_seed

def test_set_seed_reproducibility():
    set_seed(123)

    # Python's random
    python_rand = [random.randint(0, 100) for _ in range(5)]

    # NumPy
    numpy_rand = np.random.randint(0, 100, size=5).tolist()

    # PyTorch
    torch_rand = torch.randint(0, 100, (5,)).tolist()

    # Reset seed and regenerate → should match
    set_seed(123)
    assert python_rand == [random.randint(0, 100) for _ in range(5)]
    assert numpy_rand == np.random.randint(0, 100, size=5).tolist()
    assert torch_rand == torch.randint(0, 100, (5,)).tolist()

def test_set_seed_changes_results():
    set_seed(111)
    val1 = np.random.randint(0, 100)

    set_seed(222)
    val2 = np.random.randint(0, 100)

    assert val1 != val2  # different seeds give different numbers

def test_pythonhashseed_env(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    set_seed(42)
    assert os.environ["PYTHONHASHSEED"] == "42"
