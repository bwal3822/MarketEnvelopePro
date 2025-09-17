"""
alphaChain.py  —  Taylor 3-day Markov scaffold with XLSX output

Usage:
  python -X dev -u alphaChain.py --input prices.xlsx --sheet price_sheet --out_sheet price_output
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Helpers ----------
def pct(x, n=2):
    if x is None or pd.isna(x):
        return np.nan
    return round(float(x) * 100.0, n)

def load_prices(path="prices.xlsx", sheet="price_sheet"):
    p = Path(path).expanduser().resolve()
    print("Loading:", p)
    ext = p.suffix.lower()

    if ext == ".csv":
        # Try common encodings on Windows if UTF-8 fails
        for enc in (None, "utf-8-sig", "cp1252", "latin1"):
            try:
                kw = {"parse_dates": ["Date"]}
                if enc is not None:
                    kw["encoding"] = enc
                df = pd.read_csv(p, **kw)
                break
            except UnicodeDecodeError:
                print(f"Retry CSV with encoding={enc}")
        else:
            raise RuntimeError("Could not decode CSV with tried encodings.")
        return df
    elif ext in (".xlsx", ".xls"):
        try:
            return pd.read_excel(p, sheet_name=sheet, engine="openpyxl")
        except ValueError as e:
            # If sheet not found, fall back to the first sheet
            print(f"Warning: {e}. Falling back to the first sheet.")
            xls = pd.ExcelFile(p, engine="openpyxl")
            return pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def compute_features(df):
    # Ensure types
    for col in ["Open","High","Low","Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # True range and ATR
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"]  - df["Close"].shift(1)).abs()
    df["TR"] = np.maximum(tr1, np.maximum(tr2, tr3))
    df["ATR"] = df["TR"].rolling(14, min_periods=1).mean()

    # 200-day SMA as regime filter
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
    return df

def label_state(row):
    # Buy-Day
    if ((row["D"] == 1 and row["CL"] >= 0.6) or
        (row["x_lo"] > 0 and row["CL"] >= 0.5)):
        return "B"
    # Sell-Day
    if ((row["U"] == 1 and row["CL"] <= 0.4) or
        (row["x_hi"] > 0 and row["CL"] <= 0.5)):
        return "S"
    # Sell-Short Day (fallback)
    return "SS"

def adjust_by_regime(row):
    s = row["state_raw"]
    if row["regime"] == "up" and s == "SS" and row["CL"] >= 0.55:
        return "B"
    if row["regime"] == "down" and s == "B" and row["CL"] <= 0.45:
        return "SS"
    return s

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

def event_stats(sub_df):
    sub_df = sub_df.copy()
    sub_df["H_next"] = sub_df["High"].shift(-1)
    sub_df["L_next"] = sub_df["Low"].shift(-1)

    out = []
    for s in ["B","S","SS"]:
        rows = sub_df[sub_df["state"]==s]
        n = len(rows) - 1
        if n <= 5:
            out.append((s, np.nan, np.nan))
            continue
        p_hi = np.mean((rows["H_next"] > rows["High"]).iloc[:-1])
        p_lo = np.mean((rows["L_next"] < rows["Low"]).iloc[:-1])
        out.append((s, p_hi, p_lo))
    return pd.DataFrame(out, columns=["state","p_hi","p_lo"]).set_index("state")

def forecast_next(df, T_up, T_down, ev_up, ev_down):
    last = df.iloc[-1]
    regime = last["regime"]
    s_t = last["state"]

    T  = T_up if regime == "up" else T_down
    ev = ev_up if regime == "up" else ev_down

    states = ["B","S","SS"]
    pi = pd.Series(0.0, index=states)
    pi[s_t] = 1.0
    pi_next = pd.Series(pi.values @ T.values, index=T.columns)

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
            "ATR":        float(last["ATR"]),
        },
        "date_used": pd.to_datetime(last["Date"]).date()
    }

def build_output_table(df, T_up, T_down, ev_up, ev_down):
    rows = []
    states = ["B","S","SS"]
    for _, row in df.iterrows():
        s = row.get("state")
        reg = row.get("regime")
        if pd.isna(s) or pd.isna(reg):
            continue
        T  = T_up if reg == "up" else T_down
        ev = ev_up if reg == "up" else ev_down

        pi = np.array([1.0 if k == s else 0.0 for k in states])
        pi_next = pd.Series(pi @ T.values, index=states)

        p_hi = float((pi_next * ev["p_hi"]).sum(skipna=True))
        p_lo = float((pi_next * ev["p_lo"]).sum(skipna=True))

        rows.append({
            "Date": pd.to_datetime(row["Date"]).date(),
            "state": s,
            "regime": reg,
            "P_B_%": pct(pi_next["B"]),
            "P_S_%": pct(pi_next["S"]),
            "P_SS_%": pct(pi_next["SS"]),
            "P_break_prior_high_%": pct(p_hi),
            "P_break_prior_low_%": pct(p_lo),
            "prior_high": round(float(row["High"]), 2),
            "prior_low":  round(float(row["Low"]), 2),
            "ATR":        round(float(row["ATR"]), 2),
        })
    return pd.DataFrame(rows).sort_values("Date")

def write_output_sheet(input_path, output_df, sheet_name="price_output"):
    p = Path(input_path).resolve()
    if p.suffix.lower() in (".xlsx", ".xls"):
        # Append-replace sheet inside the same workbook
        with pd.ExcelWriter(p, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
            output_df.to_excel(xw, index=False, sheet_name=sheet_name)
        print(f"Wrote {len(output_df)} rows to sheet '{sheet_name}' in {p.name}")
    else:
        out_xlsx = p.with_name(p.stem + "_out.xlsx")
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
            output_df.to_excel(xw, index=False, sheet_name=sheet_name)
        print(f"Input is CSV. Wrote {len(output_df)} rows to '{out_xlsx.name}' sheet '{sheet_name}'")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="prices.xlsx")
    ap.add_argument("--sheet", default="price_sheet", help="Input sheet name for XLSX")
    ap.add_argument("--out_sheet", default="price_output", help="Output sheet name")
    args = ap.parse_args()

    df = load_prices(args.input, sheet=args.sheet).sort_values("Date").reset_index(drop=True)
    print("Rows:", len(df))

    df = compute_features(df)

    # Label states
    df["state_raw"] = df.apply(label_state, axis=1)
    df["state"] = df.apply(adjust_by_regime, axis=1)

    # Fit transition matrices by regime
    T_up   = estimate_T(df[df["regime"]=="up"])
    T_down = estimate_T(df[df["regime"]=="down"])

    # Event stats
    ev_up   = event_stats(df[df["regime"]=="up"])
    ev_down = event_stats(df[df["regime"]=="down"])

    # Forecast for most recent day
    fx = forecast_next(df, T_up, T_down, ev_up, ev_down)
    print(
        f"Date used: {fx['date_used']} | state={fx['today_state']} | regime={fx['regime']} | "
        f"P_next -> B:{pct(fx['P_next']['B'])}%  S:{pct(fx['P_next']['S'])}%  SS:{pct(fx['P_next']['SS'])}% | "
        f"Break probs -> high:{pct(fx['P_take_out_prior_high'])}%  low:{pct(fx['P_take_out_prior_low'])}% | "
        f"Levels -> high:{fx['today_levels']['prior_high']:.2f} low:{fx['today_levels']['prior_low']:.2f} ATR:{fx['today_levels']['ATR']:.2f}"
    )

    # Build and write full dated table
    out_df = build_output_table(df, T_up, T_down, ev_up, ev_down)
    write_output_sheet(args.input, out_df, sheet_name=args.out_sheet)

if __name__ == "__main__":
    main()
