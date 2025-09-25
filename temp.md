def run_std(series: pd.Series, window: int, type_exact: bool=False, sample: bool=True) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be positive")

    isnan = series.isna()
    if isnan.all():
        return pd.Series(np.nan, index=series.index)
    first_valid_pos = isnan.idxmin()
    first_valid_idx = series.index.get_loc(first_valid_pos)
    if isnan.iloc[first_valid_idx:].any():
        raise ValueError("NaNs are not allowed after the first valid value.")

    # Special case: window == 1
    if window == 1:
        if sample:
            # ddof=1 with 1-point window -> NaN (matches pandas)
            return pd.Series(np.nan, index=series.index)
        else:
            # ddof=0 with 1-point window -> 0 from first_valid onward
            out = np.full(len(series), np.nan, dtype=np.float64)
            out[first_valid_idx:] = 0.0
            return pd.Series(out, index=series.index)

    sma_x = run_SMA(series, window, exact=type_exact)
    sma_x2 = run_SMA(series * series, window, exact=type_exact)

    var = sma_x2 - (sma_x ** 2)

    var_values = var.values.astype(np.float64)
    np.maximum(var_values, 0.0, out=var_values)

    if not sample:
        return pd.Series(np.sqrt(var_values), index=series.index)

    if type_exact:
        correction = np.sqrt(window / (window - 1.0))
        return pd.Series(np.sqrt(var_values) * correction, index=series.index)
    else:
        n_valid = min(window, len(series) - first_valid_idx)
        n = np.empty(len(series), dtype=np.float64)
        n[:] = np.nan
        if n_valid > 0:
            seq = np.arange(1, n_valid + 1, dtype=np.float64)
            n[first_valid_idx:first_valid_idx + n_valid] = seq
            if first_valid_idx + n_valid < len(series):
                n[first_valid_idx + n_valid:] = window

        correction = np.full(len(series), np.nan, dtype=np.float64)
        mask = n > 1
        correction[mask] = np.sqrt(n[mask] / (n[mask] - 1.0))

        std = np.sqrt(var_values)
        out = std * correction
        return pd.Series(out, index=series.index)
