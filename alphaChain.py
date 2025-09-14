import pandas as pd
import numpy as np

# -----------------------
# Load and basic features
# -----------------------
df = pd.read_csv("prices.xlsx", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
for col in ["Open","High","Low","Close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["TR"] = np.maximum(df["High"]-df["Low"], np.maximum(abs(df["High"]-df["Close"].shift(1)), abs(df["Low"]-df["Close"].shift(1))))
df["ATR"] = df["TR"].rolling(14, min_periods=1).mean()
df["SMA200"] = df["Close"].rolling(200, min_periods=1).mean()

# Prior extremes
df["H_prev"] = df["High"].shift(1)
df["L_prev"] = df["Low"].shift(1)
df["H3_prev"] = df["High"].shift(1).rolling(3).max()
df["L3_prev"] = df["Low"].shift(1).rolling(3).min()

# Binary and normalized features
eps = 1e-6
df["U"] = (df["High"] > df["H_prev"]).astype(int)
df["D"] = (df["Low"]  < df["L_prev"]).astype(int)
df["CL"] = (df["Close"] - df["Low"]) / (np.maximum(df["High"]-df["Low"], eps))
df["gap"] = (df["Open"] - df["Close"].shift(1)) / df["ATR"]
df["x_hi"] = (df["High"] - df["H3_prev"]) / df["ATR"]
df["x_lo"] = (df["L3_prev"] - df["Low"])  / df["ATR"]
df["regime"] = np.where(df["Close"] >= df["SMA200"], "up", "down")

# -----------------------------------
# Rules-seeded Taylor state labelling
# -----------------------------------
def label_state(row):
    # Buy-Day
    if ((row["D"] == 1 and row["CL"] >= 0.6) or
        (row["x_lo"] > 0 and row["CL"] >= 0.5)):
        return "B"
    # Sell-Day
    if ((row["U"] == 1 and row["CL"] <= 0.4) or
        (row["x_hi"] > 0 and row["CL"] <= 0.5)):
        return "S"
    # Sell-Short Day (fallback or down-bias)
    return "SS"

df["state_raw"] = df.apply(label_state, axis=1)

# Optional regime-aware tie-breaks to nudge labels
def adjust_by_regime(row):
    s = row["state_raw"]
    if row["regime"] == "up" and s == "SS" and row["CL"] >= 0.55:
        return "B"
    if row["regime"] == "down" and s == "B" and row["CL"] <= 0.45:
        return "SS"
    return s

df["state"] = df.apply(adjust_by_regime, axis=1)

# -------------------------------------
# Estimate transition matrices per regime
# -------------------------------------
def estimate_T(sub_df, alpha=0.8):
    states = ["B","S","SS"]
    counts = pd.DataFrame(0.0, index=states, columns=states)
    prev = sub_df["state"].shift(1)
    cur  = sub_df["state"]
    for i,j in zip(prev, cur):
        if pd.isna(i) or pd.isna(j): 
            continue
        counts.loc[i,j] += 1
    T = (counts + alpha) / (counts.sum(axis=1).values.reshape(-1,1) + alpha*len(states))
    return T

T_up   = estimate_T(df[df["regime"]=="up"])
T_down = estimate_T(df[df["regime"]=="down"])

# ---------------------------------------------
# State-conditional event probabilities per regime
# ---------------------------------------------
def event_stats(sub_df):
    # next-day high/low break vs today
    sub_df = sub_df.copy()
    sub_df["H_next"] = sub_df["High"].shift(-1)
    sub_df["L_next"] = sub_df["Low"].shift(-1)
    sub_df["state_next"] = sub_df["state"].shift(-1)

    out = []
    for s in ["B","S","SS"]:
        rows = sub_df[sub_df["state"]==s]
        n = len(rows) - 1  # last row has no next
        if n <= 5:
            out.append((s, np.nan, np.nan))
            continue
        p_hi = np.mean((rows["H_next"] > rows["High"]).iloc[:-1])
        p_lo = np.mean((rows["L_next"] < rows["Low"]).iloc[:-1])
        out.append((s, p_hi, p_lo))
    return pd.DataFrame(out, columns=["state","p_hi","p_lo"]).set_index("state")

ev_up   = event_stats(df[df["regime"]=="up"])
ev_down = event_stats(df[df["regime"]=="down"])

# ---------------------------------------------
# Forecaster for the most recent day
# ---------------------------------------------
def forecast_next(df):
    last = df.iloc[-1]
    regime = last["regime"]
    s_t = last["state"]
    if regime == "up":
        T = T_up
        ev = ev_up
    else:
        T = T_down
        ev = ev_down

    # One-step state probabilities
    pi = pd.Series(0.0, index=["B","S","SS"])
    pi[s_t] = 1.0
    pi_next = (pi.values @ T.values)
    pi_next = pd.Series(pi_next, index=T.columns)

    # Event probabilities as mixture of next states
    p_hi = float((pi_next * ev["p_hi"]).sum(skipna=True))
    p_lo = float((pi_next * ev["p_lo"]).sum(skipna=True))

    return {
        "today_state": s_t,
        "regime": regime,
        "P_next": pi_next.to_dict(),
        "P_take_out_prior_high": p_hi,
        "P_take_out_prior_low":  p_lo,
        "today_levels": {
            "prior_high": float(last["High"]),
            "prior_low":  float(last["Low"]),
            "ATR":        float(last["ATR"])
        }
    }

fx = forecast_next(df)
print(fx)
