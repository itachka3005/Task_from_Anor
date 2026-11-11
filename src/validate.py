from pathlib import Path

import pandas as pd

from features import make_features
from model import make_lgbm

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

HORIZON = 30   #дней прогноза, как в задаче
LOOKBACK = 56  #дней истории для бейзлайнов (8 недель)

# название окна -> последний день обучения
FOLDS = {
    "A: 11.04-10.05.2026": "2026-04-10",  # самое свежее окно
    "B: 12.03-10.04.2026": "2026-03-11",  # внутри Рамазан хайит и Навруз
    "C: 11.05-09.06.2025": "2025-05-10",  # те же даты год назад, внутри Курбан хайит
}


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    return (y_true - y_pred).abs().sum() / y_true.sum()


def load_history() -> pd.DataFrame:
    """История с признаками. Признаки смотрят только на данные старше 30 дней,
    поэтому их можно посчитать один раз на всей таблице - будущее в тест не попадёт."""
    panel = pd.read_parquet(DATA_DIR / "panel.parquet")
    df = make_features(panel)
    return df[~df["is_forecast"]].copy()


def split(df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train - всё до cutoff включительно, test - следующие HORIZON дней."""
    cutoff = pd.Timestamp(cutoff)
    train = df[df["date"] <= cutoff]
    test = df[(df["date"] > cutoff) & (df["date"] <= cutoff + pd.Timedelta(days=HORIZON))]
    return train, test


def recent(train: pd.DataFrame) -> pd.DataFrame:
    """Последние LOOKBACK дней обучения."""
    return train[train["date"] > train["date"].max() - pd.Timedelta(days=LOOKBACK)]


def baseline_weekday(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Среднее по филиалу и дню недели за последние 8 недель."""
    last = recent(train)
    last = last.assign(branch=last["branch_id"].astype(str), weekday=last["date"].dt.day_of_week)
    by_weekday = last.groupby(["branch", "weekday"])["y"].mean().rename("pred")
    keys = pd.DataFrame(
        {"branch": test["branch_id"].astype(str), "weekday": test["date"].dt.day_of_week}
    )
    pred = keys.join(by_weekday, on=["branch", "weekday"])["pred"]
    #если у филиала нет такого дня недели в истории - берём его общее среднее
    fallback = keys["branch"].map(last.groupby("branch")["y"].mean())
    return pred.fillna(fallback).astype(float)


def baseline_plan_raw(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """План загрузки банка как есть."""
    return test["planned_cash_load_uzs"]


def baseline_plan(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """План загрузки, делённый на типичный запас филиала (медиана план / факт за 8 недель)."""
    last = recent(train)
    last = last[last["y"] > 0]
    ratio = (last["planned_cash_load_uzs"] / last["y"]).groupby(last["branch_id"].astype(str)).median()
    k = test["branch_id"].astype(str).map(ratio).astype(float).fillna(ratio.median())
    return test["planned_cash_load_uzs"] / k


BASELINES = {
    "среднее по дню недели": baseline_weekday,
    "план банка как есть": baseline_plan_raw,
    "план банка / запас": baseline_plan,
}


def evaluate(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """WAPE каждой модели на каждом окне. Строки - модели, столбцы - окна."""
    scores = {}
    for fold_name, cutoff in FOLDS.items():
        train, test = split(df, cutoff)
        scores[fold_name] = {name: wape(test["y"], fn(train, test)) for name, fn in models.items()}
    table = pd.DataFrame(scores)
    table["среднее"] = table.mean(axis=1)
    return table


MODELS = {
    **BASELINES,
    "LightGBM (l1)": make_lgbm("l1"),
    "LightGBM (l2)": make_lgbm("l2"),
}


def day_type(df: pd.DataFrame) -> pd.Series:
    """Тип дня - чтобы видеть, где именно модель ошибается."""
    kind = pd.Series("обычный", index=df.index)
    kind[df["is_payday"] == 1] = "зарплата"
    kind[df["is_pre_holiday"] == 1] = "разгон перед праздником"
    kind[df["is_holiday"] == 1] = "праздник"
    return kind


def errors_by_day_type(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """WAPE по типам дней, все окна вместе."""
    parts = []
    for cutoff in FOLDS.values():
        train, test = split(df, cutoff)
        part = pd.DataFrame({"y": test["y"], "kind": day_type(test)})
        for name, fn in models.items():
            part[name] = (test["y"] - fn(train, test)).abs()
        parts.append(part)
    allp = pd.concat(parts)
    grouped = allp.groupby("kind")
    table = grouped[list(models)].sum().div(grouped["y"].sum(), axis=0)
    table["доля в сумме снятий"] = grouped["y"].sum() / allp["y"].sum()
    return table


def main() -> None:
    df = load_history()
    for fold_name, cutoff in FOLDS.items():
        train, test = split(df, cutoff)
        print(f"{fold_name}: обучение до {cutoff}, тест {len(test)} строк, "
              f"{test['branch_id'].nunique()} филиалов")

    table = evaluate(df, MODELS)
    print("\nWAPE, %")
    print((table * 100).round(2).to_string())

    print("\nWAPE по типам дней, все окна вместе, %")
    print((errors_by_day_type(df, MODELS) * 100).round(1).T.to_string())


if __name__ == "__main__":
    main()