import pandas as pd
import pytest
from easy_problems.f1micro.f1_solution import f1_micro

import pytest
from easy_problems.f1micro.f1_solution import f1_micro

@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([0, 1, 1], [0, 1, 0], 0.6667),
        ([0,1,2,2], [0,1,2,2], 1.0),
        ([2,2,1,0], [1,2,1,0], 0.75),
    ]
)

def test_f1micro(y_true, y_pred, expected):
    result = f1_micro(y_true, y_pred)
    assert result == pytest.approx(expected, rel=1e-3)
