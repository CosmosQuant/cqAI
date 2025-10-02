import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class PPTConfig:
    H_SET: Tuple[int, ...] = (1, 2, 3)
    INSAMPLE_END: str = "2005-12-31"
    OOS_END: str = "2015-12-31"
    RETRAIN_STEP: str = "1M"
    lr: float = 0.05
    epochs: int = 300
    lam_p: float = 0.5
    lam_l2: float = 1e-4


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0.0, x)


def map_box(u: float, lo: float, hi: float) -> float:
    return lo + sigmoid(u) * (hi - lo)


def map_symmetric(u: float, bound: float) -> float:
    return bound * tanh(u)


def winsorized_zscore(x: np.ndarray, lower: float = 0.005, upper: float = 0.995) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    z = arr.copy()
    mask = np.isfinite(z)
    if mask.sum() == 0:
        return np.full_like(arr, np.nan)
    lo, hi = np.nanpercentile(z[mask], [lower, upper])
    z = np.clip(z, lo, hi)
    mean = np.nanmean(z[mask])
    std = np.nanstd(z[mask])
    if std < 1e-8:
        z = z - mean
        z[mask] = 0.0
        z[~mask] = np.nan
        return z
    z = (z - mean) / std
    z[~mask] = np.nan
    return z


def rolling_rank_pct(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    if window <= 1:
        return np.where(np.isfinite(values), 1.0, np.nan)

    def rank_last(x: pd.Series) -> float:
        arr = x.to_numpy(dtype=float)
        if np.isnan(arr[-1]) or np.isnan(arr).any():
            return np.nan
        order = np.argsort(arr, kind='mergesort')
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
        return ranks[-1] / len(arr)

    return series.rolling(window, min_periods=window).apply(rank_last, raw=False).to_numpy()


def compute_run_length(mask: np.ndarray) -> np.ndarray:
    run = np.zeros(len(mask), dtype=float)
    count = 0
    for i, flag in enumerate(mask):
        if flag:
            count += 1
        else:
            count = 0
        run[i] = count
    return run


def shifted_return(close: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(close, np.nan)
    if lag <= 0 or lag >= close.size:
        return out
    out[lag:] = close[lag:] / close[:-lag] - 1
    return out




# Parameter structure definition based on mathematical specification
PARAM_STRUCTURE = {
    'gate': ['u_k1', 'u_c1', 'u_wdd', 'u_c2', 'u_t2'],  # Run-length + Capitulation gate
    'driver': ['u_v1', 'u_v2', 'u_v3'],  # ReLU driver with interaction term
    'quality': ['u_alpha', 'u_sq', 'u_kg', 'u_cg', 'u_gamma'],  # Compression + LT trend guard
    'heads': ['u_theta0', 'u_theta1', 'u_phi0', 'u_phi1']  # Parametric heads per horizon
}

# === Data Layer: Feature Creation ===
def make_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Create all features from OHLC data"""
    cols = {col.lower(): col for col in df.columns}
    for required in ("close", "high", "low"):
        if required not in cols:
            raise KeyError(f"Input DataFrame must contain column '{required}'")

    close, high, low = df['close'].values, df['high'].values, df['low'].values

    # Basic returns and run-length
    prev_close = np.concatenate(([np.nan], close[:-1]))
    ret1 = close / np.concatenate([[np.nan], close[:-1]]) - 1  # 1-day return
    ret4 = close / np.concatenate([[np.nan] * 4, close[:-4]]) - 1  # 4-day return
    down_mask = ret1 < 0
    runlen = compute_run_length(np.isfinite(down_mask) & down_mask)

    # True Range and ATR
    hl = high - low
    hc = np.abs(high - prev_close)
    lc = np.abs(low - prev_close)
    tr = np.nanmax(np.stack([hl, hc, lc], axis=0), axis=0)
    atr20 = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()
    atr20_std = pd.Series(atr20).rolling(20, min_periods=20).std().to_numpy()

    # Drawdown measures
    dd_abs = np.maximum(0.0, -ret4)
    dd_atr = np.maximum(0.0, -ret4 / (atr20 + 1e-8))

    # Momentum measures
    mom60 = shifted_return(close, 60)
    mom120 = shifted_return(close, 120)
    mom250 = shifted_return(close, 250)

    # Z-scored drawdown features
    dd_abs_z = winsorized_zscore(dd_abs)
    dd_atr_z = winsorized_zscore(dd_atr)

    # LV features (rank-based)
    atr_rank = rolling_rank_pct(atr20, 60)
    hl_range = hl / np.where(close == 0, np.nan, close)
    hl_rank = rolling_rank_pct(hl_range, 60)
    vov_rank = rolling_rank_pct(atr20_std, 60)
    
    stack_LV = np.stack([atr_rank, hl_rank, vov_rank], axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        LV_raw = np.nanmedian(stack_LV, axis=0)
    LV_raw[np.all(np.isnan(stack_LV), axis=0)] = np.nan
    LV = 1.0 - LV_raw
    LV_z = winsorized_zscore(LV)

    # Long-term momentum (median of multiple horizons)
    stack_mom = np.stack([mom60, mom120, mom250], axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        mom_z = np.nanmedian(stack_mom, axis=0)
    mom_z[np.all(np.isnan(stack_mom), axis=0)] = np.nan
    mom_z = winsorized_zscore(mom_z)

    return {
        'close': close, 'high': high, 'low': low,
        'ret1': ret1, 'ret4': ret4, 'runlen': runlen,
        'atr20': atr20, 'atr20_std': atr20_std,
        'dd_abs': dd_abs, 'dd_atr': dd_atr,
        'mom60': mom60, 'mom120': mom120, 'mom250': mom250,
        'hl': hl, 'tr': tr,
        'dd_abs_z': dd_abs_z, 'dd_atr_z': dd_atr_z,
        'LV_z': LV_z, 'mom_z': mom_z
    }


def create_labels(df: pd.DataFrame, config: PPTConfig) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Create labels: forward returns and directions"""
    cols = {col.lower(): col for col in df.columns}
    close = df[cols["close"]].to_numpy(dtype=float)
    
    mu: Dict[int, np.ndarray] = {}
    p: Dict[int, np.ndarray] = {}
    for h in config.H_SET:
        mu_h = shifted_return(close, h)
        mu[h] = mu_h
        ph = np.where(np.isfinite(mu_h), (mu_h > 0).astype(float), np.nan)
        p[h] = ph
    
    return mu, p


def create_valid_mask(features: Dict[str, np.ndarray], mu: Dict[int, np.ndarray], p: Dict[int, np.ndarray], config: PPTConfig) -> np.ndarray:
    """Create valid data mask"""
    mask = np.ones(len(list(features.values())[0]), dtype=bool)
    
    # Check feature validity
    for feat in features.values():
        mask &= np.isfinite(feat)
    
    # Check label validity
    for h in config.H_SET:
        mask &= np.isfinite(mu[h])
        mask &= np.isfinite(p[h])
    
    return mask


# === Interface Layer: High-level Data Preparation ===
def prepare_data(df: pd.DataFrame, config: PPTConfig) -> Tuple[Dict[str, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray]:
    """High-level interface: complete data preparation pipeline"""
    features = make_features(df)
    mu, p = create_labels(df, config)
    mask = create_valid_mask(features, mu, p, config)
    return features, mu, p, mask


# === Model Layer: Component Computation ===
def compute_gate(features: Dict[str, np.ndarray], gate_params: Dict[str, float]) -> np.ndarray:
    """Compute gate component"""
    # Parameter mapping
    k1 = map_box(gate_params["u_k1"], 0.5, 6.0)
    c1 = map_box(gate_params["u_c1"], 2.5, 5.5)
    w_dd = sigmoid(gate_params["u_wdd"])
    c2 = map_symmetric(gate_params["u_c2"], 3.0)
    t2 = map_box(gate_params["u_t2"], 0.1, 5.0)

    # Run-length gate
    runlen = features["runlen"]
    gate_run = sigmoid(k1 * (runlen - c1)) * (runlen >= 3).astype(float)

    # Capitulation gate
    dd_abs_z = features["dd_abs_z"]
    dd_atr_z = features["dd_atr_z"]
    s_mix = w_dd * dd_abs_z + (1.0 - w_dd) * dd_atr_z
    gate_dd = sigmoid((s_mix - c2) / (t2 + 1e-6))
    
    # Total gate
    gate = np.clip(gate_run * gate_dd, 0.0, 1.0)
    return gate


def compute_driver(features: Dict[str, np.ndarray], driver_params: Dict[str, float]) -> np.ndarray:
    """Compute driver component"""
    # Parameter mapping
    v1 = map_symmetric(driver_params["u_v1"], 5.0)
    v2 = map_symmetric(driver_params["u_v2"], 5.0)
    v3 = map_symmetric(driver_params["u_v3"], 5.0)

    # Driver calculation with interaction term
    dd_abs_z = features["dd_abs_z"]
    dd_atr_z = features["dd_atr_z"]
    runlen = features["runlen"]
    driver_input = v1 * dd_abs_z + v2 * dd_atr_z + v3 * (dd_atr_z * runlen)
    driver = relu(driver_input)
    return driver


def compute_quality(features: Dict[str, np.ndarray], quality_params: Dict[str, float]) -> np.ndarray:
    """Compute quality component"""
    # Parameter mapping
    alpha_q = sigmoid(quality_params["u_alpha"])
    s_q = softplus(quality_params["u_sq"])
    k_g = map_box(quality_params["u_kg"], 0.5, 8.0)
    c_g = map_symmetric(quality_params["u_cg"], 2.0)
    gamma = map_box(quality_params["u_gamma"], 0.05, 0.35)

    # LT trend guard
    ltm = features["mom_z"]
    guard = sigmoid(k_g * (ltm - c_g))
    guard_mean = np.nanmean(guard)
    guard_std = np.nanstd(guard)
    if not np.isfinite(guard_mean):
        guard_mean = 0.0
    if guard_std < 1e-6:
        guard_std = 1.0
    guard_z = (guard - guard_mean) / guard_std

    # Quality calculation
    comp_z = features["LV_z"]
    m_q = alpha_q * comp_z + (1.0 - alpha_q) * guard_z
    quality = 1.0 + s_q * m_q
    q_min = 1.0 - gamma
    q_max = 1.0 + gamma
    quality = np.clip(quality, q_min, q_max)
    return quality


def compute_heads(core: np.ndarray, heads_params: Dict[int, Dict[str, float]], config: PPTConfig) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Compute parametric heads"""
    outputs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for h in config.H_SET:
        head = heads_params[h]
        theta0 = map_symmetric(head["u_theta0"], 3.0)
        theta1 = map_symmetric(head["u_theta1"], 3.0)
        phi0 = map_symmetric(head["u_phi0"], 4.0)
        phi1 = map_symmetric(head["u_phi1"], 4.0)
        mu_hat = theta0 + theta1 * core
        p_hat = sigmoid(phi0 + phi1 * core)
        outputs[h] = (mu_hat, p_hat)
    return outputs


class PPTModel:
    def __init__(self, config: PPTConfig):
        self.config = config
        self.params = self._init_params()

    def _init_params(self) -> Dict:
        """Initialize all model parameters using predefined structure"""
        np.random.seed(42)
        params = {}
        
        # Initialize component parameters
        for component, param_names in PARAM_STRUCTURE.items():
            if component == 'heads':
                # Special case: heads have parameters for each horizon
                params[component] = {}
                for h in self.config.H_SET:
                    params[component][h] = {name: np.random.randn() * 0.1 for name in param_names}
            else:
                # Regular components
                params[component] = {name: np.random.randn() * 0.1 for name in param_names}
        
        return params
    
    def _params_to_array(self) -> np.ndarray:
        """Convert structured parameters to flat array for scipy optimization"""
        param_list = []
        
        # Add component parameters in consistent order
        for component in ['gate', 'driver', 'quality']:
            for param_name in PARAM_STRUCTURE[component]:
                param_list.append(self.params[component][param_name])
        
        # Add head parameters
        for h in self.config.H_SET:
            for param_name in PARAM_STRUCTURE['heads']:
                param_list.append(self.params['heads'][h][param_name])
        
        return np.array(param_list)
    
    def _array_to_params(self, param_array: np.ndarray) -> Dict:
        """Convert flat array back to structured parameters"""
        params = {}
        idx = 0
        
        # Reconstruct component parameters
        for component in ['gate', 'driver', 'quality']:
            params[component] = {}
            for param_name in PARAM_STRUCTURE[component]:
                params[component][param_name] = param_array[idx]
                idx += 1
        
        # Reconstruct head parameters
        params['heads'] = {}
        for h in self.config.H_SET:
            params['heads'][h] = {}
            for param_name in PARAM_STRUCTURE['heads']:
                params['heads'][h][param_name] = param_array[idx]
                idx += 1
        
        return params

    def predict(self, features: Dict[str, np.ndarray], params: Dict = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """High-level interface: model prediction"""
        if params is None:
            p_dict = self.params
        else:
            # If params is array (from optimizer), convert to dict
            if isinstance(params, np.ndarray):
                p_dict = self._array_to_params(params)
            else:
                p_dict = params

        # Compute components
        gate = compute_gate(features, p_dict["gate"])
        driver = compute_driver(features, p_dict["driver"])
        quality = compute_quality(features, p_dict["quality"])
        
        # Core calculation
        core = gate * driver * quality
        
        # Compute heads
        outputs = compute_heads(core, p_dict["heads"], self.config)
        return outputs

    def fit(self, df: pd.DataFrame) -> 'PPTModel':
        """High-level interface: model training"""
        features, mu, p, mask = prepare_data(df, self.config)
        return train_model(self, features, mu, p, mask, self.config)


def huber_loss(pred: np.ndarray, target: np.ndarray, delta: float = 0.01) -> np.ndarray:
    error = pred - target
    abs_error = np.abs(error)
    return np.where(abs_error <= delta, 0.5 * error * error, delta * (abs_error - 0.5 * delta))


def bce_loss(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_clipped = np.clip(pred, 1e-6, 1 - 1e-6)
    return -(target * np.log(pred_clipped) + (1 - target) * np.log(1 - pred_clipped))


def loss_function(params: np.ndarray, model: PPTModel, features: Dict[str, np.ndarray], mu: Dict[int, np.ndarray], p: Dict[int, np.ndarray], config: PPTConfig) -> float:
    outputs = model.predict(features, params)
    total_loss = 0.0
    for h in config.H_SET:
        mu_hat, p_hat = outputs[h]
        valid = np.isfinite(mu[h]) & np.isfinite(p[h])
        if not np.any(valid):
            continue
        mu_h = mu[h][valid]
        p_h = p[h][valid]
        mu_pred = mu_hat[valid]
        p_pred = p_hat[valid]
        reg_loss = np.mean(huber_loss(mu_pred, mu_h))
        cls_loss = np.mean(bce_loss(p_pred, p_h))
        total_loss += reg_loss + config.lam_p * cls_loss
    total_loss += config.lam_l2 * np.sum(params * params)
    return total_loss


def train_model(model: PPTModel, features: Dict[str, np.ndarray], mu: Dict[int, np.ndarray], p: Dict[int, np.ndarray], mask: np.ndarray, config: PPTConfig) -> PPTModel:
    feat_clean = {k: v[mask] for k, v in features.items()}
    mu_clean = {h: mu[h][mask] for h in config.H_SET}
    p_clean = {h: p[h][mask] for h in config.H_SET}

    print(f"Training on {mask.sum()} samples...")

    def objective(param_array: np.ndarray) -> float:
        return loss_function(param_array, model, feat_clean, mu_clean, p_clean, config)

    # Get initial parameters as array
    initial_params = model._params_to_array()
    initial_loss = objective(initial_params)
    print(f"Initial loss: {initial_loss:.6f}")

    result = minimize(objective, initial_params, method="L-BFGS-B", options={"maxiter": config.epochs, "disp": True})

    # Update model parameters
    model.params = model._array_to_params(result.x)
    final_loss = objective(result.x)
    print(f"Final loss: {final_loss:.6f}")

    return model


if __name__ == "__main__":
    print("Testing PPT implementation...")
    np.random.seed(42)
    n_days = 1000
    returns = np.random.normal(0, 0.02, n_days)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    dates = pd.date_range("2000-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({"date": dates, "close": close, "high": high, "low": low})

    config = PPTConfig()
    model = PPTModel(config)
    print(f"Model parameters: {len(model.params)}")

    model = model.fit(df)
    features, mu, p, mask = prepare_data(df, config)
    test_features = {k: v[mask][:100] for k, v in features.items()}
    outputs = model.predict(test_features)
    for h in config.H_SET:
        mu_hat, p_hat = outputs[h]
        print(f"H={h}: mu_hat[{np.nanmin(mu_hat):.4f}, {np.nanmax(mu_hat):.4f}], p_hat[{np.nanmin(p_hat):.4f}, {np.nanmax(p_hat):.4f}]")
    print("PPT test completed successfully.")
