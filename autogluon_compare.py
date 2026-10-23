"""AI4I 분할 v2로 AutoGluon 모델별 · 앙상블 조합별 비교.

실행: pip install autogluon.tabular  →  python autogluon_compare.py
- 학습 = train, 고르기 = val(score_val), 마지막 확인 = test(score_test, 고를 때 보지 않음)
- demo 몫은 쓰지 않음
"""
from itertools import combinations
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

DATA = Path(__file__).resolve().parent.parent / "csv"
TARGET = "failure_type"      # "failure_type"(NONE · TWF · HDF · PWF · OSF) 또는 "Machine failure"(0/1)
ADD_FEATURES = True          # 온도 차 · 출력 칸을 더함(Pack도 같은 값을 계산함)
TIME_LIMIT = 600             # 초
TOP_K = 4                    # 조합을 만들 상위 모델 수(val 기준)

DROP = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


def load(name):
    df = pd.read_csv(DATA / f"ai4i-{name}.csv")
    if TARGET == "failure_type":
        # 한 행에 고장 유형이 둘 이상이면 TWF > HDF > PWF > OSF 순으로 하나(train 14행). RNF(무작위 고장)만 있는 행은 NONE
        df[TARGET] = "NONE"
        for t in ["OSF", "PWF", "HDF", "TWF"]:
            df.loc[df[t] == 1, TARGET] = t
    else:
        df[TARGET] = df["Machine failure"]
    if ADD_FEATURES:
        df["temp_diff_k"] = df["Process temperature [K]"] - df["Air temperature [K]"]
        df["power_w"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * 3.141592653589793 / 60
    return df.drop(columns=[c for c in DROP if c != TARGET])


train, val, test = load("train"), load("val"), load("test")
print(train[TARGET].value_counts(), "\n")

metric = "f1_macro" if TARGET == "failure_type" else "f1"
predictor = TabularPredictor(
    label=TARGET,
    eval_metric=metric,
    sample_weight="balance_weight",   # 고장 행이 적어서 클래스 비율로 무게를 맞춤
    path="ag-ai4i",
).fit(
    train_data=train,
    tuning_data=val,
    presets="medium_quality",
    time_limit=TIME_LIMIT,
)

# 1) 모델별: AutoGluon이 만든 단일 모델 + 기본 앙상블(WeightedEnsemble_L2)
lb = predictor.leaderboard(test, extra_metrics=["balanced_accuracy", "accuracy"], silent=True)

# 2) 앙상블 조합별: val 상위 TOP_K 단일 모델로 2 · 3개 조합을 직접 만듦
singles = lb[~lb["model"].str.startswith("WeightedEnsemble")].sort_values("score_val", ascending=False)
top = singles["model"].head(TOP_K).tolist()
for r in (2, 3):
    for combo in combinations(top, r):
        suffix = "_" + "+".join(m.split("_")[0] for m in combo)
        predictor.fit_weighted_ensemble(base_models=list(combo), name_suffix=suffix)

lb = predictor.leaderboard(test, extra_metrics=["balanced_accuracy", "accuracy"], silent=True)
lb = lb.sort_values("score_val", ascending=False)   # 고르는 기준은 val
cols = ["model", "score_val", "score_test", "balanced_accuracy", "accuracy", "pred_time_test", "fit_time"]
pd.set_option("display.width", 200)
print(lb[cols].to_string(index=False))
lb[cols].to_csv("ag-ai4i-leaderboard.csv", index=False)

best = lb.iloc[0]["model"]
print(f"\nval 1등: {best} — test {lb.iloc[0]['score_test']:.4f}")
print("앙상블이 쓴 모델:", predictor.info()["model_info"][best].get("children_info", {}).keys() if best.startswith("WeightedEnsemble") else "단일 모델")
