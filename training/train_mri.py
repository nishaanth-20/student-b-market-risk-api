"""Train the Student B Market Risk Index model (LightGBM + Optuna).

Usage (from the repo root):
    python -m training.train_mri --data-dir path/to/provided_dataset --trials 50

Writes:
    models/mri_lgbm.joblib        the final model (loaded by the API)
    models/model_metadata.json    importance, best params, metrics (served by /model-metadata)
    models/training_report.json   full results for the report (search space, splits, baselines)

Steps
1. Load the provided CoinMarketCap CSVs (one per token), skip empty files,
   drop duplicate token+date rows. Each token is put on a daily calendar but
   gaps are NOT filled (e.g. XLM's missing 2023), so no rolling window or
   7-day target ever crosses a gap.
2. Build features with the agreed formulas and the MRI target at t+7.
3. Split by time with a 7-day gap (group agreement, point 8):
   train <= 2022-12-31, validation 2023-01-08 to 2023-12-31, test from 2024-01-08.
4. Optuna tunes LightGBM on the validation set (RMSE).
5. Refit on train + validation with the best parameters, evaluate on test,
   and compare with two baselines.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import FORECAST_HORIZON, MODEL_FEATURES  # noqa: E402
from app.features import (  # noqa: E402
    add_drawdown_risk,
    add_peer_features,
    add_target,
    to_daily_calendar,
)

SEED = 42
H = FORECAST_HORIZON

# Filename keywords -> token symbol.
TOKEN_ALIASES = {
    "BTC": ["btc", "bitcoin"],
    "ETH": ["eth", "ethereum"],
    "XRP": ["xrp", "ripple"],
    "SOL": ["sol", "solana"],
    "DOGE": ["doge", "dogecoin"],
    "XMR": ["xmr", "monero"],
    "XLM": ["xlm", "stellar"],
    "TRX": ["trx", "tron"],
}

# Time split agreed by the group: 7 days dropped AFTER each boundary, so the
# t+7 targets of the last rows in one split never share dates with the next.
# train <= 2022-12-31, validation 2023-01-08 .. 2023-12-31, test >= 2024-01-08.
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_START, VAL_END = pd.Timestamp("2023-01-01") + pd.Timedelta(days=H), pd.Timestamp("2023-12-31")
TEST_START, TEST_END = pd.Timestamp("2024-01-01") + pd.Timedelta(days=H), pd.Timestamp("2099-12-31")

SEARCH_SPACE = {
    "learning_rate": "log-uniform [0.01, 0.2]",
    "num_leaves": "int [8, 128]",
    "max_depth": "int [3, 12]",
    "min_child_samples": "int [10, 200]",
    "subsample": "uniform [0.5, 1.0] (bagging_freq = 1)",
    "colsample_bytree": "uniform [0.5, 1.0]",
    "reg_alpha": "log-uniform [1e-8, 10]",
    "reg_lambda": "log-uniform [1e-8, 10]",
    "n_estimators": "up to 2000, early stopping 100 rounds on validation RMSE",
}


# ------------------------------------------------------------------- data
def token_from_path(path: Path, data_dir: Path) -> str | None:
    """Find the coin in the file name or its folder, e.g. 'Bitcoin (BTC)/x.csv'."""
    rel = path.relative_to(data_dir).with_suffix("")
    words = set(re.split(r"[^a-z]+", str(rel).lower()))
    for token, aliases in TOKEN_ALIASES.items():
        if words & set(aliases):
            return token
    return None


def load_dataset(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.rglob("*.csv")):
        name = str(path.relative_to(data_dir))
        if path.stat().st_size <= 3:  # empty, or only a UTF-8 BOM
            print(f"skip empty file: {name}")
            continue
        token = token_from_path(path, data_dir)
        if token is None:
            print(f"skip (no coin name in folder or file name): {name}")
            continue
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")  # ',' or ';', strips BOM
        except (pd.errors.EmptyDataError, csv.Error):
            df = pd.DataFrame()
        if df.empty:
            print(f"skip empty file: {name}")
            continue
        df.columns = [c.strip() for c in df.columns]
        date_col = "timeOpen" if "timeOpen" in df.columns else "date"
        df["date"] = (
            pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None).dt.normalize()
        )
        df["token"] = token
        frames.append(df[["token", "date", "open", "high", "low", "close", "volume"]])
        print(f"loaded {name}: {token}, {len(df)} rows")
    if not frames:
        raise SystemExit(f"No usable CSV files in {data_dir}")

    data = pd.concat(frames, ignore_index=True)
    before = len(data)
    data = data.drop_duplicates(["token", "date"], keep="last")
    print(f"dropped {before - len(data)} duplicate token+date rows")
    return data


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    out = []
    for token, df in data.groupby("token"):
        df = to_daily_calendar(df.drop(columns="token"))
        df = add_peer_features(df)
        df = add_drawdown_risk(df)
        df = add_target(df, horizon=H)
        df["token"] = token
        out.append(df)
    full = pd.concat(out, ignore_index=True)
    # Keep rows with a target and all features (matches the group's row counts).
    # Missing features are added back on purpose later by add_missing_copies().
    return full.dropna(subset=["MRI_target", *MODEL_FEATURES]).reset_index(drop=True)


def split(df: pd.DataFrame):
    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)]
    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)]
    return train, val, test


def add_missing_copies(df: pd.DataFrame, frac: float = 0.15) -> pd.DataFrame:
    """Copy some training rows with one peer API's features blanked out.

    In production Student A or C may be down; these copies teach LightGBM
    where to send missing values instead of guessing.
    """
    rng = np.random.default_rng(SEED)
    a_cols = ["Return20"]
    c_cols = ["Volatility20", "MaxPrice20", "GapRisk", "DrawdownRisk"]
    copies = []
    for cols in (a_cols, c_cols):
        sample = df.sample(frac=frac, random_state=int(rng.integers(1e9))).copy()
        sample[cols] = np.nan
        copies.append(sample)
    return pd.concat([df, *copies], ignore_index=True)


# ------------------------------------------------------------------ model
def metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def objective(trial, X_tr, y_tr, X_va, y_va):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    model = lgb.LGBMRegressor(n_estimators=2000, random_state=SEED, verbose=-1, **params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    trial.set_user_attr("best_iteration", int(model.best_iteration_ or 2000))
    return metrics(y_va, model.predict(X_va))["rmse"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    df = build_features(load_dataset(args.data_dir))
    train, val, test = split(df)
    print(f"rows: train {len(train)}, validation {len(val)}, test {len(test)}")
    for name, part in (("train", train), ("validation", val), ("test", test)):
        if part.empty:
            raise SystemExit(f"The {name} split is empty; check the dataset dates.")

    train_aug = add_missing_copies(train)
    X_tr, y_tr = train_aug[MODEL_FEATURES], train_aug["MRI_target"]
    X_va, y_va = val[MODEL_FEATURES], val["MRI_target"]

    # 1. Optuna search on the validation year.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: objective(t, X_tr, y_tr, X_va, y_va), n_trials=args.trials,
                   show_progress_bar=False)
    best = study.best_trial
    best_params = {**best.params, "subsample_freq": 1}
    n_estimators = best.user_attrs["best_iteration"]
    print(f"best validation RMSE {best.value:.3f} with {n_estimators} trees: {best_params}")

    # 2. Default LightGBM on the same data, to show what tuning bought.
    default = lgb.LGBMRegressor(random_state=SEED, verbose=-1).fit(X_tr, y_tr)
    default_val = metrics(y_va, default.predict(X_va))

    # 3. Refit on train + validation, evaluate once on the untouched test set.
    final_train = add_missing_copies(pd.concat([train, val]))
    final = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=SEED, verbose=-1, **best_params)
    final.fit(final_train[MODEL_FEATURES], final_train["MRI_target"])

    y_te = test["MRI_target"]
    test_pred = np.clip(final.predict(test[MODEL_FEATURES]), 0, 100)
    test_metrics = metrics(y_te, test_pred)

    # Baselines: "risk stays the same as today" and "always the training mean".
    has_now = test["MRI"].notna()
    persistence = metrics(y_te[has_now], test["MRI"][has_now])
    mean_baseline = metrics(y_te, np.full(len(y_te), final_train["MRI_target"].mean()))

    # Robustness: test error when one peer API is down.
    robustness = {}
    for label, cols in (("student_a_down", ["Return20"]),
                        ("student_c_down", ["Volatility20", "MaxPrice20", "GapRisk", "DrawdownRisk"])):
        X = test[MODEL_FEATURES].copy()
        X[cols] = np.nan
        robustness[label] = metrics(y_te, np.clip(final.predict(X), 0, 100))

    per_token = {
        tok: metrics(g["MRI_target"], np.clip(final.predict(g[MODEL_FEATURES]), 0, 100))
        for tok, g in test.groupby("token") if len(g) > 10
    }

    gain = final.booster_.feature_importance(importance_type="gain")
    importance = {f: round(float(g / gain.sum()), 4) for f, g in zip(MODEL_FEATURES, gain)}
    importance = dict(sorted(importance.items(), key=lambda kv: -kv[1]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, args.out_dir / "mri_lgbm.joblib")

    meta = {
        "feature_importance": importance,
        "feature_importance_type": "gain, normalised to sum to 1",
        "best_params": {**best_params, "n_estimators": n_estimators},
        "metrics": {
            "test": test_metrics,
            "baseline_persistence_test": persistence,
            "baseline_mean_test": mean_baseline,
        },
        "training_period": f"{df['date'].min():%Y-%m-%d} to {VAL_END:%Y-%m-%d} (train + validation)",
        "test_period": f"{TEST_START:%Y-%m-%d} to {min(TEST_END, test['date'].max()):%Y-%m-%d}",
        "tokens": sorted(df["token"].unique().tolist()),
    }
    (args.out_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2))

    report = {
        **meta,
        "search_space": SEARCH_SPACE,
        "optuna_trials": args.trials,
        "best_validation_rmse": best.value,
        "default_lightgbm_validation": default_val,
        "robustness_test": robustness,
        "per_token_test": per_token,
        "rows": {"train": len(train), "validation": len(val), "test": len(test),
                 "train_with_missing_copies": len(train_aug)},
        "splits": {
            "train": f"<= {TRAIN_END:%Y-%m-%d}",
            "validation": f"{VAL_START:%Y-%m-%d} to {VAL_END:%Y-%m-%d}",
            "test": f"{TEST_START:%Y-%m-%d} onwards",
        },
        "optuna_history": [
            {"trial": t.number, "rmse": t.value, **t.params} for t in study.trials
        ],
    }
    (args.out_dir / "training_report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\nTest set ({TEST_START:%Y-%m-%d} onwards)")
    print(f"  LightGBM      RMSE {test_metrics['rmse']:.2f}  MAE {test_metrics['mae']:.2f}  R2 {test_metrics['r2']:.3f}")
    print(f"  Persistence   RMSE {persistence['rmse']:.2f}  MAE {persistence['mae']:.2f}  R2 {persistence['r2']:.3f}")
    print(f"  Mean          RMSE {mean_baseline['rmse']:.2f}")
    print(f"  Feature importance: {importance}")
    print(f"\nSaved model and metadata to {args.out_dir}/")


if __name__ == "__main__":
    main()
