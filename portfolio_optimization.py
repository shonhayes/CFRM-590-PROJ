import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Tuple, Optional, Union

def mean_variance_optimization(
    returns_df: pd.DataFrame,
    target_return: float,
    risk_free_rate: float = 0.0,
    allow_short_selling: bool = True,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
    trading_cost_per_dollar: Union[float, np.ndarray] = (1+0.001)**(1/252)-1, # eg 0.
    prev_weights: Optional[np.ndarray] = np.zeros(8),
    use_equality_target: bool = True,   # if False, uses >= target instead of ==
) -> Tuple[np.ndarray, float, float]:
    """
    Mean-variance optimization with *net* target return:
    E[w]^T mu - c * turnover == (or >=) target_return,
    where turnover = sum_i |w_i - w_i_prev| and c is linear trading cost per $ traded.

    trading_cost_per_dollar can be a scalar (same for all assets) or a vector (asset-specific).
    """

    # Inputs
    mu = returns_df.mean().values
    Sigma = returns_df.cov().values
    n = len(mu)

    # Allow per-asset costs
    if np.isscalar(trading_cost_per_dollar):
        c_vec = np.full(n, float(trading_cost_per_dollar))
    else:
        c_vec = np.asarray(trading_cost_per_dollar).reshape(-1)
        assert c_vec.shape == (n,), "trading_cost_per_dollar vector must match number of assets"

    w = cp.Variable(n)
    # Turnover linearization: t_i >= |w_i - w_i_prev|
    t = cp.Variable(n, nonneg=True)

    constraints = [cp.sum(w) == 1]

    if allow_short_selling:
        constraints += [w >= min_weight, w <= max_weight]
    else:
        constraints += [w >= 0, w <= max_weight]

    # Linearize |w - w_prev| with two inequalities: t >=  w - w_prev ; t >= -(w - w_prev)
    constraints += [ t >=  w - prev_weights,
                     t >= -(w - prev_weights) ]

    # Net expected return (after linear trading costs)
    # net_return = mu^T w - c^T t
    net_return = mu @ w - c_vec @ t

    if use_equality_target:
        constraints += [ net_return == target_return ]
    else:
        constraints += [ net_return >= target_return ]  # often more realistic/feasible

    obj = cp.Minimize(cp.quad_form(w, Sigma))

    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status in ("optimal", "optimal_inaccurate"):
        w_star = w.value
        port_ret_gross = float(mu @ w_star)
        port_vol = float(np.sqrt(w_star @ Sigma @ w_star))
        return w_star, port_ret_gross, port_vol
    else:
        raise ValueError(f"Optimization failed: {prob.status}. "
                         f"Target (net) return {target_return:.6f} may be infeasible.")


