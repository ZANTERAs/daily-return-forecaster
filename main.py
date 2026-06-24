import sys
import json
import time
import re
from datetime import date, datetime
import yfinance as yf

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for box/arrow chars)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass
import pandas as pd
import numpy as np
import ta
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ────────────────────────────────────────────────────────────────────
# ↓ The ONLY line you need to change when switching companies ↓
TICKER        = "MSFT"
# ─────────────────────────────────────────────────────────────────────────────
PERIOD        = "15y"
WINDOW        = 60           # look-back window (trading days)
HORIZON       = 20           # forward return horizon (~1 month)
TRAIN_RATIO   = 0.70         # used for WF burn-in; final split uses VAL_DAYS
VAL_RATIO     = 0.15         # test set fraction
VAL_DAYS      = 252          # rolling val: always the last ~1 year before test

# QUANTILES are auto-calibrated in main() based on historical return volatility.
# Low-vol stocks (σ < threshold) → P10/P50/P90  (80% band)
# High-vol stocks (σ ≥ threshold) → P05/P50/P95  (90% band, avoids undercoverage)
QUANTILE_WIDEN_VOL = 12.0   # pp std of 20-day returns that triggers wider bands
EARN_LOOKAHEAD     = 7      # calendar days ahead to flag as "earnings window"
QUANTILES     = [0.10, 0.50, 0.90]   # overwritten at runtime if σ ≥ threshold
N_Q           = len(QUANTILES)

EPOCHS        = 100
BATCH_SIZE    = 64
HIDDEN_DIM    = 64
NUM_LAYERS    = 2
DROPOUT       = 0.35
BIDIRECTIONAL = True
SPEARMAN_W    = 0.3
LR            = 3e-4
LR_PATIENCE   = 7
ES_PATIENCE   = 25
N_ENSEMBLE    = 5            # final model: average of N_ENSEMBLE independently seeded models
WF_SPLITS     = 5
WF_BURN_IN    = 0.60

# ── Auto sector ETF via yfinance ─────────────────────────────────────────────
# yfinance returns one of the 11 standard GICS sector names for any US equity.
# We map each sector to its SPDR Select Sector ETF.  Unknown sectors (e.g. ETFs,
# foreign stocks, indices) fall back to SPY (broad market).
SECTOR_TO_ETF = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",
    "Energy":                 "XLE",
    "Utilities":              "XLU",
    "Healthcare":             "XLV",
    "Consumer Defensive":     "XLP",
    "Consumer Cyclical":      "XLY",
    "Industrials":            "XLI",
    "Real Estate":            "XLRE",
    "Basic Materials":        "XLB",
    "Communication Services": "XLC",
}


def _get_sector_etf(ticker: str) -> tuple:
    """
    Look up `ticker`'s GICS sector via yfinance and return (sector_name, etf).
    Falls back to ('Unknown', 'SPY') if the lookup fails or the sector is not
    in SECTOR_TO_ETF (e.g. the ticker is itself an ETF or a foreign stock).
    """
    try:
        sector = yf.Ticker(ticker).info.get("sector", "")
        etf    = SECTOR_TO_ETF.get(sector, "")
        if etf:
            return sector, etf
    except Exception:
        pass
    return "Unknown", "SPY"


_SECTOR_NAME, SECTOR_ETF = _get_sector_etf(TICKER)


# ── Period helper ─────────────────────────────────────────────────────────────

def _period_to_kwargs(period: str) -> dict:
    """
    yfinance only accepts: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
    Anything like '15y' '20y' is invalid and silently returns empty data.
    This converts e.g. '15y' → {'start': '2010-05-20'} so the download
    always uses an explicit date range instead of a period string.
    """
    m = re.match(r"^(\d+)y$", period)
    if m:
        years     = int(m.group(1))
        today     = date.today()
        try:
            start = today.replace(year=today.year - years)
        except ValueError:          # Feb 29 on non-leap year
            start = today.replace(year=today.year - years, day=28)
        return {"start": start.strftime("%Y-%m-%d")}
    return {"period": period}       # already a valid yfinance period string


# ── Compile helper ────────────────────────────────────────────────────────────

def _maybe_compile(model: nn.Module, n_feat: int) -> nn.Module:
    """Use torch.compile (reduce-overhead) when the Inductor C++ backend is
    available (requires MSVC on Windows).  Falls back silently to eager mode."""
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        with torch.no_grad():
            compiled(torch.zeros(1, WINDOW, n_feat))
        return compiled
    except Exception:
        return model


# ── Data ──────────────────────────────────────────────────────────────────────

def _fetch_close(ticker: str, period: str) -> pd.Series:
    raw = yf.download(ticker, **_period_to_kwargs(period),
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    return raw["Close"].squeeze()


def _get_earnings_dates(ticker: str) -> pd.DatetimeIndex:
    """
    Fetch historical + upcoming earnings dates via yfinance.
    Returns a DatetimeIndex (timezone-naive).  Empty index on failure.
    Note: yfinance typically provides ~40 quarters (~10 years) of history.
    """
    try:
        t    = yf.Ticker(ticker)
        earn = t.get_earnings_dates(limit=60)   # ~15 years of quarterly dates
        if earn is None or earn.empty:
            earn = t.earnings_dates             # fallback attribute
        if earn is not None and not earn.empty:
            return earn.index.tz_localize(None) if earn.index.tz else earn.index
    except Exception:
        pass
    return pd.DatetimeIndex([])


def load_and_prepare_data(ticker: str, period: str) -> pd.DataFrame:
    print(f"\n[1/3] Downloading {ticker} price data ({period})...")
    raw = pd.DataFrame()
    for attempt in range(1, 4):
        raw = yf.download(ticker, **_period_to_kwargs(period), auto_adjust=True)
        if not raw.empty:
            break
        print(f"      attempt {attempt}/3 returned empty data — retrying in 3s...")
        time.sleep(3)
    if raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for '{ticker}' after 3 attempts. "
            "Check the ticker symbol and your internet connection."
        )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    df = raw[["High", "Low", "Close", "Volume"]].dropna()
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
    print(f"      {len(df)} rows  [{df.index[0].date()} – {df.index[-1].date()}]")

    # ── Technical indicators + Realized Volatility ───────────────────────────
    print("[2/4] Computing technical indicators + realized volatility...", end="", flush=True)
    df["EMA_20"]    = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df["RSI_14"]    = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["MACD"]      = ta.trend.MACD(close=close).macd_diff()
    df["OBV"]       = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    sma50           = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    sma200          = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    df["GC_spread"] = sma50 - sma200
    bb              = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["BB_width"]  = bb.bollinger_wband()
    df["ATR_14"]    = ta.volatility.AverageTrueRange(
                          high=high, low=low, close=close, window=14).average_true_range()
    df["STOCH_K"]   = ta.momentum.StochasticOscillator(
                          high=high, low=low, close=close, window=14, smooth_window=3).stoch()
    # Realized volatility: rolling std of daily returns at 3 horizons (in % pp)
    # Gives the model live awareness of how volatile the stock currently is,
    # which helps the quantile bands adapt and improves regime detection.
    daily_ret      = close.pct_change() * 100
    df["RV_5"]     = daily_ret.rolling(5).std()
    df["RV_20"]    = daily_ret.rolling(20).std()
    df["RV_60"]    = daily_ret.rolling(60).std()
    print(" done  (EMA, RSI, MACD, OBV, GC, BB, ATR, STOCH, RV×3)")

    # ── Market context (regime awareness) ────────────────────────────────────
    print(f"[3/4] Fetching market context: SPY / {SECTOR_ETF} / ^TNX...", end="", flush=True)
    spy            = _fetch_close("SPY",       period)
    sect           = _fetch_close(SECTOR_ETF,  period)
    tnx            = _fetch_close("^TNX",      period)
    print(" done")

    df["SPY_ret"]  = spy.pct_change(20).mul(100).reindex(df.index)
    df["SECT_ret"] = sect.pct_change(20).mul(100).reindex(df.index)
    df["RATE_ch"]  = tnx.diff(20).reindex(df.index)

    # ── Earnings calendar ─────────────────────────────────────────────────────
    # Flags the EARN_LOOKAHEAD calendar days before each earnings date as 1.
    # Valid: earnings dates are publicly scheduled, so this is not look-ahead bias.
    print(f"[4/4] Fetching earnings calendar...", end="", flush=True)
    earn_dates = _get_earnings_dates(ticker)
    df["EARN_flag"] = 0.0
    if len(earn_dates) > 0:
        idx_norm = df.index.normalize()
        for ed in earn_dates:
            ed_n = ed.normalize() if hasattr(ed, "normalize") else ed
            mask = (idx_norm < ed_n) & (idx_norm >= ed_n - pd.Timedelta(days=EARN_LOOKAHEAD))
            df.loc[mask, "EARN_flag"] = 1.0
        flagged = int(df["EARN_flag"].sum())
        print(f" {len(earn_dates)} earnings dates  →  {flagged} flagged rows")
    else:
        print(" not available — feature set to 0")

    return df.dropna()


FEAT_COLS = [
    "Close", "Volume", "EMA_20", "RSI_14", "MACD", "OBV",
    "GC_spread", "BB_width", "ATR_14", "STOCH_K",          # 10 technicals
    "SPY_ret", "SECT_ret", "RATE_ch",                       #  3 market context
    "RV_5", "RV_20", "RV_60",                               #  3 realized volatility
    "EARN_flag",                                             #  1 earnings calendar
]


def make_sequences(scaled: np.ndarray, close_raw: np.ndarray,
                   window: int, horizon: int):
    """
    X[i]  = scaled[i : i+window]            (features, shape window×n_feat)
    y[i]  = 100 × (close[i+window-1+horizon] − close[i+window-1]) / close[i+window-1]
            (forward return in % over the next `horizon` trading days)
    """
    max_i = len(scaled) - window - horizon
    X     = np.stack([scaled[i : i + window] for i in range(max_i)]).astype(np.float32)
    base  = close_raw[window - 1          : window - 1          + max_i]
    fwd   = close_raw[window - 1 + horizon : window - 1 + horizon + max_i]
    y     = (100.0 * (fwd - base) / base).astype(np.float32)
    return X, y


def to_tensors(X: np.ndarray, y: np.ndarray):
    return torch.tensor(X), torch.tensor(y).unsqueeze(1)


# ── Model ─────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(lstm_out), dim=1)
        return (w * lstm_out).sum(dim=1)


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM + soft attention → Q quantile predictions.

    Outputs [P10, P50, P90] of the HORIZON-day forward return distribution.
    - P50 is the point forecast (used for DirAcc and IC).
    - P10/P90 form an ~80% confidence interval, replacing MC-Dropout.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, bidirectional: bool = True, n_quantiles: int = N_Q):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        out_dim   = hidden_dim * (2 if bidirectional else 1)
        self.attn = Attention(out_dim)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(out_dim, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.drop(self.attn(out)))   # (N, n_quantiles)


# ── Losses ────────────────────────────────────────────────────────────────────

def pinball_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Pinball (quantile) loss averaged across all Q quantiles.
      pred : (N, Q)  predicted quantiles
      true : (N, 1)  actual returns
    For quantile q: loss = q*(y-yhat) if y>=yhat else (q-1)*(y-yhat)
    This forces pred[:,0] ≤ pred[:,1] ≤ pred[:,2] to minimise loss.
    """
    total = 0.0
    for i, q in enumerate(QUANTILES):
        err    = true - pred[:, i : i + 1]
        total += torch.where(err >= 0, q * err, (q - 1) * err).mean()
    return total / N_Q


def soft_spearman_loss(pred_p50: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Differentiable (1 − Spearman ρ) applied to the P50 predictions.
    Pushes the model to rank samples in the same order as actual returns →
    directly improves IC (correlation between predicted and actual return).

    Soft rank: rank[i] ≈ Σ_j sigmoid((x[i]−x[j]) / τ)
    τ is set adaptively to the spread of predictions to keep sigmoid gradients alive.
    """
    p   = pred_p50.squeeze(1)
    t   = true.squeeze(1)
    tau = (p.detach().std() + 1e-4).item()

    r_p = torch.sigmoid((p.unsqueeze(0) - p.unsqueeze(1)) / tau).sum(dim=1)
    r_t = torch.sigmoid((t.unsqueeze(0) - t.unsqueeze(1)) / tau).sum(dim=1)

    rp_c = r_p - r_p.mean()
    rt_c = r_t - r_t.mean()
    rho  = (rp_c * rt_c).sum() / (
        torch.sqrt((rp_c ** 2).sum() * (rt_c ** 2).sum()) + 1e-8
    )
    return 1.0 - rho


def combined_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return ((1 - SPEARMAN_W) * pinball_loss(pred, true)
            + SPEARMAN_W    * soft_spearman_loss(pred[:, 1:2], true))


# ── Training ──────────────────────────────────────────────────────────────────

def train(model: nn.Module, train_loader: DataLoader,
          val_loader: DataLoader, verbose: bool = True,
          label: str = "") -> tuple:
    """
    Train with early stopping.

    label  — when non-empty, prints a compact one-line `\\r` progress bar that
             overwrites itself each epoch (used for walk-forward folds so the
             terminal never looks frozen without flooding it with epoch lines).
             verbose=True prints the traditional multi-line epoch log.

    Returns (stopped_epoch, best_val_loss).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=LR_PATIENCE, factor=0.5)
    best_val, best_w, no_imp = float("inf"), None, 0
    stopped_epoch = EPOCHS

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = combined_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(xb)
        t_loss /= len(train_loader.dataset)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                v_loss += combined_loss(model(xb), yb).item() * len(xb)
        v_loss /= len(val_loader.dataset)
        scheduler.step(v_loss)

        # ── Progress display ─────────────────────────────────────────────────
        if label:
            # Compact inline bar — overwrites same line every epoch
            bar_done  = int(20 * epoch / EPOCHS)
            bar       = "#" * bar_done + "-" * (20 - bar_done)
            print(f"\r    {label}  [{bar}] ep {epoch:3d}/{EPOCHS}"
                  f"  train={t_loss:.4f}  val={v_loss:.4f}"
                  f"  best={min(best_val, v_loss):.4f}",
                  end="", flush=True)
        elif verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"    Epoch {epoch:3d}/{EPOCHS}  train={t_loss:.4f}  val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            best_w   = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp   = 0
        else:
            no_imp += 1
            if no_imp >= ES_PATIENCE:
                stopped_epoch = epoch
                if label:
                    print()   # end the \r line before caller prints results
                elif verbose:
                    print(f"    Early stop at epoch {epoch}  (best val={best_val:.4f})")
                break
    else:
        # All EPOCHS completed without early stop
        if label:
            print()   # end the \r line

    model.load_state_dict(best_w)
    return stopped_epoch, best_val


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_metrics(pred_q: np.ndarray, true: np.ndarray) -> dict:
    """
    pred_q : (N, Q)  quantile predictions — columns = [P10, P50, P90]
    true   : (N,)    actual returns

    DirAcc   sign(P50) == sign(actual)               random baseline = 50%
    IC       Pearson r(P50, actual)                  random baseline = 0.00
    Coverage fraction of actuals inside [P10, P90]   ideal ≈ 80%
    """
    p10, p50, p90 = pred_q[:, 0], pred_q[:, 1], pred_q[:, 2]
    return dict(
        rmse     = float(np.sqrt(np.mean((p50 - true) ** 2))),
        mae      = float(np.mean(np.abs(p50 - true))),
        dir_acc  = float(np.mean(np.sign(p50) == np.sign(true))),
        ic       = float(np.corrcoef(p50, true)[0, 1]) if len(p50) > 1 else 0.0,
        coverage = float(np.mean((p10 <= true) & (true <= p90))),
    )


def compute_backtest(pred_p50: np.ndarray, true_returns: np.ndarray,
                     threshold: float = 0.0) -> dict:
    """
    Simple long/short backtest driven by the P50 signal.

    Uses every HORIZON-th sample so periods are non-overlapping (clean Sharpe).
    threshold  — minimum |P50| to enter a trade; 0 = always trade.

    Returns Sharpe (annualised), max drawdown (pp), total return (pp),
    win rate, and trade count.
    """
    idx    = np.arange(0, len(pred_p50), HORIZON)   # non-overlapping slices
    p50    = pred_p50[idx]
    ret    = true_returns[idx]

    signal = np.where(p50 >  threshold,  1.0,
             np.where(p50 < -threshold, -1.0, 0.0))
    pnl    = signal * ret                            # pp gain/loss per trade

    trades_per_year = 252 / HORIZON
    sharpe  = float((pnl.mean() / (pnl.std() + 1e-8)) * np.sqrt(trades_per_year))

    cum    = np.cumsum(pnl)
    peak   = np.maximum.accumulate(np.concatenate([[0.0], cum]))
    max_dd = float((peak[:-1] - cum).max()) if len(cum) > 0 else 0.0

    active   = signal != 0
    win_rate = float(np.mean(pnl[active] > 0)) if active.sum() > 0 else 0.0

    return dict(
        sharpe    = sharpe,
        max_dd    = max_dd,
        total_ret = float(cum[-1]) if len(cum) > 0 else 0.0,
        win_rate  = win_rate,
        n_trades  = int(active.sum()),
    )


def print_backtest(bt: dict, prefix: str = "") -> None:
    sr  = bt["sharpe"]
    tag = "good" if sr > 0.5 else ("marginal" if sr > 0 else "negative")
    print(f"{prefix}Backtest  (long/short on P50, non-overlapping {HORIZON}-day periods)")
    print(f"{prefix}  Sharpe={sr:+.2f} ({tag})  MaxDD={bt['max_dd']:.1f}pp  "
          f"TotalRet={bt['total_ret']:+.1f}pp  "
          f"WinRate={bt['win_rate']*100:.0f}%  N={bt['n_trades']} trades")


def print_metrics(m: dict, prefix: str = "") -> None:
    ideal_cov = int((QUANTILES[-1] - QUANTILES[0]) * 100)   # 80% or 90%
    q_lo_lbl  = f"P{int(QUANTILES[0]*100):02d}"
    q_hi_lbl  = f"P{int(QUANTILES[-1]*100):02d}"
    print(f"{prefix}RMSE={m['rmse']:.2f}pp  MAE={m['mae']:.2f}pp  "
          f"DirAcc={m['dir_acc']*100:.1f}%  IC={m['ic']:.3f}  "
          f"Coverage={m['coverage']*100:.0f}%"
          f"  (random: 50%/0.00, ideal coverage ~{ideal_cov}% [{q_lo_lbl}–{q_hi_lbl}])")


# ── Ensemble helpers ──────────────────────────────────────────────────────────

def build_model(n_feat: int) -> nn.Module:
    return _maybe_compile(
        LSTMModel(n_feat, HIDDEN_DIM, NUM_LAYERS, DROPOUT, BIDIRECTIONAL, N_Q).to(device),
        n_feat,
    )


def predict_ensemble(models: list, X_t: torch.Tensor) -> np.ndarray:
    """Average quantile predictions across all ensemble members. Returns (N, Q)."""
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(X_t.to(device)).cpu().numpy())
    return np.mean(preds, axis=0)


# ── Walk-forward Validation ───────────────────────────────────────────────────

def walk_forward_evaluate(values: np.ndarray, close_raw: np.ndarray,
                          n_feat: int) -> list:
    """
    Walk-forward validation with rolling val window.
    Each fold trains a single model (not an ensemble) for speed;
    the final model uses N_ENSEMBLE seeds for deployment quality.
    """
    n         = len(values)
    burn_in   = int(n * WF_BURN_IN)
    fold_size = (n - burn_in) // WF_SPLITS
    results   = []

    print(f"\nWalk-forward validation  "
          f"({WF_SPLITS} folds, burn-in={WF_BURN_IN:.0%}, "
          f"rolling val={VAL_DAYS}d, single model per fold)")
    print("-" * 78)

    wf_t0 = time.time()

    for k in range(WF_SPLITS):
        train_end   = burn_in + k * fold_size
        test_end    = min(train_end + fold_size, n)
        if test_end - train_end < WINDOW + HORIZON + 5:
            break

        # Rolling val: last VAL_DAYS rows within the training block
        val_start_k = max(train_end - VAL_DAYS, WINDOW + 1)

        scaler_k = MinMaxScaler()
        scaler_k.fit(values[:val_start_k])
        scaled_k = scaler_k.transform(values)

        X, y     = make_sequences(scaled_k, close_raw, WINDOW, HORIZON)
        tr_s     = val_start_k - WINDOW          # seq idx: end of train
        va_s     = train_end   - WINDOW          # seq idx: end of val / start of test
        te_end_k = min(test_end - WINDOW + 1, len(X))

        if tr_s <= 0 or va_s <= tr_s or te_end_k <= va_s + 1:
            continue

        tr_ld = DataLoader(TensorDataset(*to_tensors(X[:tr_s],        y[:tr_s])),
                           BATCH_SIZE, shuffle=True)
        va_ld = DataLoader(TensorDataset(*to_tensors(X[tr_s:va_s],    y[tr_s:va_s])),
                           BATCH_SIZE)
        Xt, yt = to_tensors(X[va_s:te_end_k], y[va_s:te_end_k])

        if len(Xt) < 2:
            continue

        print(f"\n  Fold {k+1}/{WF_SPLITS}  "
              f"[{len(X[:tr_s]):,} train / {len(X[tr_s:va_s]):,} val / {len(Xt):,} test seqs]")

        fold_t0 = time.process_time()
        m_k = build_model(n_feat)
        ep, bv = train(m_k, tr_ld, va_ld, verbose=False,
                       label=f"Fold {k+1}/{WF_SPLITS}")
        fold_sec = time.process_time() - fold_t0

        m_k.eval()
        with torch.no_grad():
            pred_q = m_k(Xt.to(device)).cpu().numpy()
        true_k = yt.squeeze(1).numpy()

        met = compute_metrics(pred_q, true_k)
        results.append({**met, "fold": k + 1})

        ideal_cov = int((QUANTILES[-1] - QUANTILES[0]) * 100)
        print(f"  >>  ep={ep:3d}  best_val={bv:.4f}  {fold_sec:.0f}s  |  "
              f"RMSE={met['rmse']:.2f}pp  DirAcc={met['dir_acc']*100:.1f}%  "
              f"IC={met['ic']:.3f}  Coverage={met['coverage']*100:.0f}%"
              f"  (ideal ~{ideal_cov}%)")

    if results:
        avg = {m: np.mean([r[m] for r in results])
               for m in ("rmse", "mae", "dir_acc", "ic", "coverage")}
        total_sec = time.time() - wf_t0
        print(f"\n{'─' * 78}")
        print(f"  Average  ({len(results)} folds, {total_sec:.0f}s total)  |  "
              f"RMSE={avg['rmse']:.2f}pp  DirAcc={avg['dir_acc']*100:.1f}%  "
              f"IC={avg['ic']:.3f}  Coverage={avg['coverage']*100:.0f}%")

    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(test_dates, true_returns: np.ndarray, pred_q: np.ndarray,
         current_close: float, fc_p10: float, fc_p50: float, fc_p90: float) -> None:
    """
    Panel 1: Test set — actual return vs P50 prediction with P10/P90 confidence band.
    Panel 2: Ensemble forecast — P10 / P50 / P90 for the next HORIZON days.
    """
    p10 = pred_q[:, 0]
    p50 = pred_q[:, 1]
    p90 = pred_q[:, 2]
    fd  = list(test_dates)

    met        = compute_metrics(pred_q, true_returns)
    target_p50 = current_close * (1 + fc_p50 / 100)
    signal     = "BUY" if fc_p50 > 0 else "SELL"
    ideal_cov  = int((QUANTILES[-1] - QUANTILES[0]) * 100)
    q_lo_lbl   = f"P{int(QUANTILES[0]*100):02d}"
    q_hi_lbl   = f"P{int(QUANTILES[-1]*100):02d}"

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            (f"Test set — P50 predicted vs actual {HORIZON}-day return (%)  |  "
             f"DirAcc={met['dir_acc']*100:.1f}%  IC={met['ic']:.3f}  "
             f"Coverage={met['coverage']*100:.0f}% (ideal ~{ideal_cov}%)"),
            f"Ensemble forecast  ({N_ENSEMBLE} models × {q_lo_lbl}/P50/{q_hi_lbl})",
        ],
        vertical_spacing=0.14,
        row_heights=[0.65, 0.35],
    )

    # Shaded confidence band (P_lo – P_hi)
    fig.add_trace(go.Scatter(
        x=fd + list(reversed(fd)),
        y=list(p90) + list(reversed(p10)),
        fill="toself", fillcolor="rgba(255,183,77,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{q_lo_lbl}–{q_hi_lbl} band", hoverinfo="skip",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=test_dates, y=true_returns,
                             name="Actual return",
                             line=dict(color="#4fc3f7", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_dates, y=p50,
                             name="P50 predicted",
                             line=dict(color="#ffb74d", dash="dash", width=1.5)), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)

    # Forecast panel — horizontal error bar showing P10/P50/P90
    s = "+" if fc_p50 >= 0 else ""
    fig.add_trace(go.Bar(
        x=[fc_p50], y=["Next 20d"],
        orientation="h",
        marker_color="#7e57c2",
        name=f"P50: {s}{fc_p50:.1f}%",
        text=f"P50  {s}{fc_p50:.1f}%",
        textposition="inside",
        width=0.4,
    ), row=2, col=1)
    # outer quantile markers
    fig.add_trace(go.Scatter(
        x=[fc_p10, fc_p90], y=["Next 20d", "Next 20d"],
        mode="markers+text",
        marker=dict(symbol="line-ns-open", size=16, color="white", line_width=2),
        text=[f"{q_lo_lbl} {fc_p10:+.1f}%", f"{q_hi_lbl} {fc_p90:+.1f}%"],
        textposition=["bottom center", "bottom center"],
        name=f"{q_lo_lbl} / {q_hi_lbl}",
    ), row=2, col=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#ef5350", row=2, col=1)

    fig.update_layout(
        title=(f"{TICKER}  |  {HORIZON}-day Return Model (ensemble x{N_ENSEMBLE})  |  "
               f"Current ${current_close:.2f}  →  P50 {s}{fc_p50:.1f}%  |  "
               f"Range ${current_close*(1+fc_p10/100):.2f}–${current_close*(1+fc_p90/100):.2f}"
               f"  |  {signal}  →  ~${target_p50:.2f}"),
        template="plotly_dark",
        height=740,
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Signal date (last day of input window)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_xaxes(title_text="Predicted 20-day return (%)", row=2, col=1)
    fig.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global QUANTILES, N_Q

    df        = load_and_prepare_data(TICKER, PERIOD)
    values    = df[FEAT_COLS].values
    close_raw = df["Close"].values
    n, n_feat = values.shape

    # ── Auto-calibrate quantile bands from training-set return volatility ─────
    # Use the first TRAIN_RATIO fraction as a proxy for training data.
    # Wide bands (P5/P50/P95) give better coverage for high-vol stocks like NVDA;
    # the default P10/P50/P90 is fine for low-vol stocks like EXC.
    approx_end   = max(int(n * TRAIN_RATIO) - HORIZON, HORIZON + 1)
    base_c       = close_raw[:approx_end]
    fwd_c        = close_raw[HORIZON : approx_end + HORIZON]
    min_len      = min(len(base_c), len(fwd_c))
    train_rets   = 100.0 * (fwd_c[:min_len] - base_c[:min_len]) / base_c[:min_len]
    ret_std      = float(np.std(train_rets))

    if ret_std >= QUANTILE_WIDEN_VOL:
        QUANTILES = [0.05, 0.50, 0.95]   # 90% band for high-vol stocks
    else:
        QUANTILES = [0.10, 0.50, 0.90]   # 80% band (default)
    N_Q = len(QUANTILES)

    q_lo  = int(QUANTILES[0]  * 100)
    q_hi  = int(QUANTILES[-1] * 100)
    print("=" * 78)
    print(f"  Ticker      : {TICKER}")
    print(f"  Sector ETF  : {SECTOR_ETF}  "
          f"({'SPY fallback — sector not detected' if _SECTOR_NAME == 'Unknown' else _SECTOR_NAME})")
    print(f"  Return vol  : {ret_std:.1f}pp (20-day, training window)")
    print(f"  Quantiles   : P{q_lo}/P50/P{q_hi}  "
          f"({'widened — high vol' if q_lo == 5 else 'default'})")
    print(f"  Data        : {n} rows  |  {n_feat} features  "
          f"|  {df.index[0].date()} – {df.index[-1].date()}")
    print("=" * 78)

    print("\n" + "=" * 78)
    print("STEP 1 / 2  —  Walk-forward validation")
    print("=" * 78)
    walk_forward_evaluate(values, close_raw, n_feat)

    # ── Final ensemble model ───────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("STEP 2 / 2  —  Final ensemble model")
    print("=" * 78)
    # Test boundary: last VAL_RATIO of data
    test_start = int(n * (TRAIN_RATIO + VAL_RATIO))

    # Rolling val: 1 year immediately before the test boundary
    val_start  = max(test_start - VAL_DAYS, WINDOW + 1)

    # Scaler fitted only on training rows (no leakage into val/test)
    scaler = MinMaxScaler()
    scaler.fit(values[:val_start])
    scaled = scaler.transform(values)

    X, y  = make_sequences(scaled, close_raw, WINDOW, HORIZON)
    tr_s  = val_start  - WINDOW     # exclusive upper index for train seqs
    va_s  = test_start - WINDOW     # exclusive upper index for val seqs

    X_tr, y_tr = X[:tr_s],     y[:tr_s]
    X_va, y_va = X[tr_s:va_s], y[tr_s:va_s]
    X_te, y_te = X[va_s:],     y[va_s:]

    tr_ld  = DataLoader(TensorDataset(*to_tensors(X_tr, y_tr)), BATCH_SIZE, shuffle=True)
    va_ld  = DataLoader(TensorDataset(*to_tensors(X_va, y_va)), BATCH_SIZE)
    X_te_t, y_te_t = to_tensors(X_te, y_te)

    print(f"\nFinal ensemble — {device}  |  {n_feat} features  |  {N_ENSEMBLE} seeds")
    print(f"Train: {len(X_tr):,} seqs  |  Val (rolling {VAL_DAYS}d): {len(X_va):,} seqs  "
          f"|  Test: {len(X_te):,} seqs")
    print(f"Loss: {1-SPEARMAN_W:.0%} pinball + {SPEARMAN_W:.0%} Spearman(P50)")
    print("=" * 78)

    models      = []
    ens_t0      = time.time()
    for seed in range(N_ENSEMBLE):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n  [Model {seed+1}/{N_ENSEMBLE}  seed={seed}]")
        model_t0 = time.time()
        m = build_model(n_feat)
        ep, bv = train(m, tr_ld, va_ld, verbose=True)
        model_sec = time.time() - model_t0
        print(f"  => Done  ep={ep}  best_val={bv:.4f}  {model_sec:.0f}s")
        models.append(m)
        # Save underlying module (unwrap compile wrapper if present)
        base = getattr(m, "_orig_mod", m)
        torch.save(base.state_dict(), f"{TICKER}_lstm_e{seed}.pt")

    ens_sec = time.time() - ens_t0
    print(f"\n{'=' * 78}")
    print(f"Ensemble complete — {N_ENSEMBLE} models  {ens_sec:.0f}s total")

    pred_q = predict_ensemble(models, X_te_t)   # (N, Q)
    true_r = y_te_t.squeeze(1).numpy()
    test_met = compute_metrics(pred_q, true_r)
    backtest = compute_backtest(pred_q[:, 1], true_r)

    print(f"\nTest set evaluation (ensemble average):")
    print_metrics(test_met, prefix="  ")
    print()
    print_backtest(backtest, prefix="  ")

    # Signal dates: last day of each test input window
    test_signal_dates = df.index[
        va_s + WINDOW - 1 : va_s + WINDOW - 1 + len(y_te)
    ]

    # Forecast from the final available window
    current_close = float(df["Close"].iloc[-1])
    last_t        = torch.tensor(scaled[-WINDOW:], dtype=torch.float32).unsqueeze(0)
    fc_q          = predict_ensemble(models, last_t).squeeze(0)   # (Q,)
    fc_p10, fc_p50, fc_p90 = float(fc_q[0]), float(fc_q[1]), float(fc_q[2])

    q_lo_lbl = f"P{int(QUANTILES[0]*100):02d}"
    q_hi_lbl = f"P{int(QUANTILES[-1]*100):02d}"
    signal_str = "BUY" if fc_p50 > 0 else "SELL"
    print(f"\nEnsemble forecast  ({HORIZON}-day horizon from {df.index[-1].date()}):")
    print(f"  {q_lo_lbl} / P50 / {q_hi_lbl} : {fc_p10:+.2f}% / {fc_p50:+.2f}% / {fc_p90:+.2f}%")
    print(f"  Current close   : ${current_close:.2f}")
    print(f"  Implied range   : "
          f"${current_close*(1+fc_p10/100):.2f} – ${current_close*(1+fc_p90/100):.2f}")
    print(f"  Signal          : {signal_str}")

    # ── Save metadata JSON ────────────────────────────────────────────────────
    metadata = {
        "ticker":       TICKER,
        "sector":       _SECTOR_NAME,
        "sector_etf":   SECTOR_ETF,
        "trained_at":   datetime.now().isoformat(timespec="seconds"),
        "data_start":   str(df.index[0].date()),
        "data_end":     str(df.index[-1].date()),
        "n_rows":       n,
        "n_features":   n_feat,
        "features":     FEAT_COLS,
        "quantiles":    QUANTILES,
        "config": {
            "period":        PERIOD,
            "window":        WINDOW,
            "horizon":       HORIZON,
            "hidden_dim":    HIDDEN_DIM,
            "num_layers":    NUM_LAYERS,
            "dropout":       DROPOUT,
            "bidirectional": BIDIRECTIONAL,
            "spearman_w":    SPEARMAN_W,
            "n_ensemble":    N_ENSEMBLE,
            "batch_size":    BATCH_SIZE,
            "lr":            LR,
        },
        "test_metrics": test_met,
        "backtest":     backtest,
        "forecast": {
            "from_date":     str(df.index[-1].date()),
            "horizon_days":  HORIZON,
            "current_close": current_close,
            "p10_pct":       round(fc_p10, 4),
            "p50_pct":       round(fc_p50, 4),
            "p90_pct":       round(fc_p90, 4),
            "price_low":     round(current_close * (1 + fc_p10 / 100), 2),
            "price_mid":     round(current_close * (1 + fc_p50 / 100), 2),
            "price_high":    round(current_close * (1 + fc_p90 / 100), 2),
            "signal":        signal_str,
        },
    }
    meta_path = f"{TICKER}_model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved → {meta_path}")

    plot(test_signal_dates, true_r, pred_q,
         current_close, fc_p10, fc_p50, fc_p90)


if __name__ == "__main__":
    main()
