
## Project - reformatting one classical rule-based trading signal into a modern AI stype

- keep the targeting model structure as State/Gate x Driver x Quality
- interfaces for monotonic splines/NAM extension, group sparsity, calibration, rolling/expanding training, etc

Here is a complete pseudocode skeleton for the JAX core implementation, from feature engineering → parametrized Gate/Driver/Quality → heads → loss → JAX training → rolling OOS test → decision → backtest → metrics.
Extendable with monotone splines, group lasso, calibration, etc.

0) Conventions & Data Interface
# Data conventions:
# df: DataFrame[Date, Close, High, Low], daily, range 1980-01-01 ~ 2015-12-31
# Execution benchmark: signal computed at close t, enter at same close T=t, returns from [t+1, ..., t+H] close
# Costs: 0 (PoC)
# Horizons H ∈ {1,2,3}

H_SET = (1, 2, 3)
INSAMPLE_END = "2005-12-31"    # training cutoff
OOS_END       = "2015-12-31"   # test end
RETRAIN_STEP  = "1M"           # monthly retrain (can also be quarterly)

1) Feature Engineering
def make_features(df):
    Close, High, Low = df.Close, df.High, df.Low

    # Basic
    ret_1 = Close.pct_change()
    ret_4 = ret_1.rolling(4).sum()

    tr = max(High-Low, (High-Close.shift(1)).abs(), (Low-Close.shift(1)).abs())
    ATR20 = tr.rolling(20).mean()
    VoV   = ATR20.rolling(20).std()

    # Run-length (down streaks)
    runlen_down = consecutive_count(ret_1 < 0)

    # CLV and compression proxies
    CLV = (Close - (High+Low)/2) / ((High-Low)/2)
    hl_over_c = (High-Low) / Close

    # rolling ranks (0–1)
    atr_rank = rolling_rank(ATR20, L=60)
    hl_rank  = rolling_rank(hl_over_c, L=60)
    vov_rank = rolling_rank(VoV, L=60)
    comp = 1 - median(atr_rank, hl_rank, vov_rank)   # higher = tighter

    # LT trend
    mom = lambda L: Close/Close.shift(L) - 1
    LTm = median(mom(60), mom(120), mom(250))

    # Normalization
    dd_abs = clip(-ret_4, 0, None)
    dd_atr = clip(-ret_4 / ATR20, 0, None)

    zscore = lambda x: (winsorize(x, 0.005, 0.995) - mean(x)) / std(x)
    feats = {
        "dd_abs_z": zscore(dd_abs),
        "dd_atr_z": zscore(dd_atr),
        "runlen":   runlen_down,
        "comp_z":   zscore(comp),
        "LTm":      zscore(LTm),
    }

    # Labels
    fwd = {H: Close.shift(-H)/Close.shift(-1) - 1 for H in H_SET}
    mu  = {H: fwd[H] for H in H_SET}
    p   = {H: (mu[H] > 0).astype(float) for H in H_SET}

    mask = finite_all([feats[k] for k in feats]) & finite_all([mu[h] for h in H_SET])

    return feats, mu, p, mask

2) Parametrization and Extendable Components
# --- Mapping helpers ---
def map_box(u, lo, hi):        # unbounded → [lo, hi]
    return lo + sigmoid(u) * (hi - lo)

def map_symmetric(u, bound):   # unbounded → [-bound, bound]
    return bound * tanh(u)

# --- Monotone spline placeholders ---
class MonoSpline1D:
    def __init__(self, knots):
        self.knots = knots
        self.w     = Param(len(knots))
    def __call__(self, x):
        phi = basis_monotone(x, self.knots)  # [T, K]
        w   = softplus(self.w)
        return phi @ w

class IsoSpline1D(MonoSpline1D):
    def __call__(self, x):
        return -(super().__call__(x))

# --- Gate / Driver / Quality ---
class Gate:
    def __init__(self, cfg):
        self.u_k1  = Param()
        self.u_c1  = Param()
        self.u_c2  = Param()
        self.u_t2  = Param()
        self.S_abs = MonoSpline1D(cfg.knots_abs)
        self.S_atr = MonoSpline1D(cfg.knots_atr)

    def __call__(self, F):
        k1  = map_box(self.u_k1, 0.5, 5.0)
        c1  = map_box(self.u_c1, 3.5, 4.5)
        c2  = map_symmetric(self.u_c2, 3.0)
        t2  = map_box(self.u_t2, 0.3, 3.0)

        run  = F["runlen"]
        gate_run = sigmoid(k1*(run - c1)) * (run >= 3)

        s = self.S_abs(F["dd_abs_z"]) + self.S_atr(F["dd_atr_z"])
        gate_dd = sigmoid((s - c2) / (t2 + 1e-6))

        return clip(gate_run * gate_dd, 0.0, 1.0)

class Driver:
    def __init__(self, cfg):
        self.D_abs = MonoSpline1D(cfg.knots_abs)
        self.D_atr = MonoSpline1D(cfg.knots_atr)
        self.u_mix = Param()

    def __call__(self, F):
        base  = self.D_abs(F["dd_abs_z"]) + self.D_atr(F["dd_atr_z"])
        inter = softplus(self.u_mix) * F["dd_atr_z"] * F["runlen"]
        return softplus(base + inter)

class Quality:
    def __init__(self, cfg):
        self.u_qmin = Param()
        self.u_qmax = Param()
        self.Qc     = MonoSpline1D(cfg.knots_comp)
        self.Qt     = MonoSpline1D(cfg.knots_trend)
        self.u_wc   = Param()
        self.u_wt   = Param()

    def __call__(self, F):
        qmin = map_box(self.u_qmin, 0.60, 0.95)
        qmax = map_box(self.u_qmax, 1.05, 1.40)
        raw  = (1
                + softplus(self.u_wc) * self.Qc(F["comp_z"])
                + softplus(self.u_wt) * self.Qt(F["LTm"]))
        return clip(raw, qmin, qmax)

3) Model Heads & Forward
class Head:
    def __init__(self):
        self.u_t0 = Param(); self.u_t1 = Param()
        self.u_p0 = Param(); self.u_p1 = Param()
    def __call__(self, core):
        mu_hat = map_symmetric(self.u_t0, 3.0) + map_symmetric(self.u_t1, 3.0) * core
        p_hat  = sigmoid(map_symmetric(self.u_p0, 3.0) + map_symmetric(self.u_p1, 3.0) * core)
        return mu_hat, p_hat

class PPTModel:
    def __init__(self, cfg):
        self.gate = Gate(cfg); self.driver = Driver(cfg); self.quality = Quality(cfg)
        self.head = {H: Head() for H in H_SET}

    def forward(self, F):
        core = self.gate(F) * self.quality(F) * self.driver(F)
        outs = {}
        for H in H_SET:
            outs[H] = self.head[H](core)
        return outs

4) Loss, Regularization, Calibration
def huber(yhat, y, delta=0.01):
    e = yhat - y
    a = abs(e)
    return where(a<=delta, 0.5*e*e, delta*(a-0.5*delta))

def bce(phat, y01):
    ph = clip(phat, 1e-6, 1-1e-6)
    return -(y01*log(ph) + (1-y01)*log(1-ph))

def group_lasso(params, groups):
    reg = 0.0
    for g in groups:
        v = concat([p.ravel() for p in g])
        reg += norm(v, 2)
    return reg

def loss_fn(model, batch, lam_p=0.5, lam_l2=1e-4, lam_grp=0.0, groups=None):
    F, mu, p, mask = batch
    outs = model.forward(F)

    loss = 0.0
    for H in H_SET:
        mu_hat, p_hat = outs[H]
        m = mask & isfinite(mu[H])
        loss += mean(huber(mu_hat[m], mu[H][m])) \
                + lam_p * mean(bce(p_hat[m], p[H][m]))

    all_params = model.parameters()
    l2 = sum([sum(x*x) for x in all_params])
    loss += lam_l2 * l2

    if lam_grp > 0 and groups is not None:
        loss += lam_grp * group_lasso(all_params, groups)

    return loss

def calibrate_probs(phat_train, y_train, method="isotonic"):
    ...

5) Optimizer & Training (JAX/Optax)
def train_window_jax(model, batch, lr=0.05, epochs=300, lam_p=0.5, lam_l2=1e-4):
    opt = optax.adam(lr)
    opt_state = opt.init(model.parameters())

    @jax.jit
    def step(params, opt_state, batch):
        model = bind_params(params)
        l, grads = jax.value_and_grad(loss_fn)(model, batch, lam_p, lam_l2)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l

    params = model.parameters()
    for _ in range(epochs):
        params, opt_state, l = step(params, opt_state, batch)
    model = bind_params(params)
    return model

6) Rolling / Expanding Training & Inference
def date_grid(df, step="1M"):
    ...

def prepare_batch(feats, mu, p, mask, idx):
    F = {k: asarray(v[idx]) for k, v in feats.items()}
    MU = {H: asarray(mu[H][idx]) for H in H_SET}
    P  = {H: asarray(p[H][idx]) for H in H_SET}
    M  = asarray(mask[idx])
    return (F, MU, P, M)

def run_training_rolling(df, cfg):
    feats, mu, p, mask = make_features(df)

    insample_end = to_date(INSAMPLE_END)
    oos_end      = to_date(OOS_END)
    grid = date_grid(df[(df.Date>=to_date("1980-01-01")) & (df.Date<=insample_end)], step=RETRAIN_STEP)

    model_ckpts = []
    calib_objs  = {H: [] for H in H_SET}
    scores_out  = {H: np.full(len(df), np.nan) for H in H_SET}
    phat_out    = {H: np.full(len(df), np.nan) for H in H_SET}
    muhat_out   = {H: np.full(len(df), np.nan) for H in H_SET}

    for T0, T1 in zip(grid[:-1], grid[1:]):
        tr_idx = (df.Date <= T0)
        te_idx = (df.Date >  T0) & (df.Date <= min(T1, oos_end))

        model = PPTModel(cfg)
        batch_tr = prepare_batch(feats, mu, p, mask, tr_idx)
        model    = train_window_jax(model, batch_tr, lr=0.05, epochs=300)

        outs_tr  = model.forward(batch_tr[0])
        calib = {}
        for H in H_SET:
            _, p_hat_tr = outs_tr[H]
            calib[H] = calibrate_probs(to_numpy(p_hat_tr[batch_tr[3]]),
                                       to_numpy(batch_tr[2][H][batch_tr[3]]),
                                       method="isotonic")
            calib_objs[H].append(calib[H])

        batch_te = prepare_batch(feats, mu, p, mask, te_idx)
        outs_te  = model.forward(batch_te[0])

        for H in H_SET:
            mu_hat, p_hat = outs_te[H]
            p_hat_adj     = apply_calibrator(calib[H], to_numpy(p_hat))
            score         = p_hat_adj * to_numpy(mu_hat)

            scores_out[H][te_idx] = score
            phat_out[H][te_idx]   = p_hat_adj
            muhat_out[H][te_idx]  = to_numpy(mu_hat)

        model_ckpts.append(serialize_params(model))

    return dict(scores=scores_out, phat=phat_out, muhat=muhat_out, models=model_ckpts)

7) Decision & Backtest
def decide(score, phat, p_th=0.55):
    return (score > 0) & (phat > p_th)

def backtest_simple(mu, decisions, H):
    pnl = np.where(decisions, mu[H], 0.0)
    return pnl

def evaluate_oos(df, run_out, mu, p):
    metrics = {}
    pnl_sum = {}
    for H in H_SET:
        score = run_out["scores"][H]
        phat  = run_out["phat"][H]
        dec   = decide(score, phat, p_th=0.55)
        pnl   = backtest_simple(mu, dec, H)

        metrics[H] = {
            "IC":     corr(run_out["muhat"][H], mu[H]),
            "RankIC": spearman(run_out["muhat"][H], mu[H]),
            "Hit":    mean( (run_out["muhat"][H] > 0) == (mu[H] > 0) ),
            "PerTradeMu": mean(mu[H][dec==1]),
            "Turnover":   mean(dec.astype(int).diff().abs()==1),
            "NTrades":    sum(dec==1),
            "CumPnL":     np.nansum(pnl),
        }
        pnl_sum[H] = pnl
    return metrics, pnl_sum

8) Main Flow
def main(df):
    feats, mu, p, mask = make_features(df)

    cfg = SimpleNamespace(
        knots_abs=[-3,-1,0,1,3],
        knots_atr=[-3,-1,0,1,3],
        knots_comp=[-3,-1,0,1,3],
        knots_trend=[-3,-1,0,1,3],
    )

    run_out = run_training_rolling(df, cfg)
    metrics, pnl = evaluate_oos(df, run_out, mu, p)

    report(metrics, pnl)


