from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass(frozen=True)
class ValidationResult:
    """
    When doing 2-layer cross validation:
        Losses from inner test folds (n_outer*n_inner, n_model),
        Generalization error of inner folds inside an outer fold (n_outer, n_model),
        Index of best model in each outer split/fold (n_outer),
        Losses of best models on outer test set (n_outer),
        Generalization error for model selection process (float)
    When doing 1-layer cross validation
        Losses of each folds (1*n_inner, n_model),
        Generalization error folds (1, n_model),
        Index of best model (1) (one element float array),
    """
    test_err_inner: np.ndarray
    gen_err_inner: np.ndarray
    idx_best_inner: np.ndarray
    test_err_outer: Optional[np.ndarray] = None
    gen_err_estimate: Optional[float] = None

