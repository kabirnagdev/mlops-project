import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
later = pd.read_csv("data/later.csv")
holdout = pd.read_csv("data/holdout_features.csv")

numeric = train.select_dtypes(include="number").columns.tolist()

print("\n========== 1. LEAKING COLUMN CANDIDATES ==========")
print("Train columns:", train.columns.tolist())
print("Later columns:", later.columns.tolist())
print("Holdout columns:", holdout.columns.tolist())
print("Train columns absent from holdout:",
      sorted(set(train.columns) - set(holdout.columns)))

print("\n========== 2. UNIT MIXUP CANDIDATES ==========")
for col in numeric:
    if col not in ["plates_sold"]:
        print(f"\n{col}")
        print(train.groupby("block")[col].agg(
            ["count", "min", "mean", "median", "max"]
        ).round(2))

print("\n========== 3. SENTINEL VALUES ==========")
for col in numeric:
    counts = train[col].value_counts()
    suspicious = counts[counts.index.isin([999, 9999, -999, -9999])]
    if not suspicious.empty:
        print(col, suspicious.to_dict())

print("\n========== 4. DUPLICATE ROWS ==========")
print("Exact duplicates:", train.duplicated().sum())
print("Duplicate record IDs:", train["record_id"].duplicated().sum())

print("\n========== 5. STUCK SENSOR CANDIDATES ==========")
daily = train.copy()
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date")

for col in numeric:
    if col == "plates_sold":
        continue
    daily_values = daily.groupby("date")[col].median()
    runs = daily_values.groupby(
        daily_values.ne(daily_values.shift()).cumsum()
    )
    for _, run in runs:
        if len(run) >= 3:
            print(col, "value:", run.iloc[0],
                  "start:", run.index[0].date(),
                  "end:", run.index[-1].date(),
                  "days:", len(run))

print("\n========== 6. TARGET OUTLIERS ==========")
median_target = train["plates_sold"].median()
print("Median:", median_target)
print("Rows above 10x median:")
print(train.loc[
    train["plates_sold"] > 10 * median_target,
    ["record_id", "date", "block", "plates_sold"]
].to_string(index=False))

print("\n========== 7. SPELLING VARIANTS ==========")
for col in ["day_type", "block"]:
    print(f"\n{col} — training values:")
    print(train[col].value_counts(dropna=False).to_string())
    print(f"\n{col} — later values:")
    print(later[col].value_counts(dropna=False).to_string())

print("\n========== INSPECTION COMPLETE ==========")
