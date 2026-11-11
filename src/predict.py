"""Обучает модель на всей истории и заполняет submission.csv.

Запуск: python src/predict.py
Перед этим должны быть выполнены build_daily.py и prepare.py.
"""
from pathlib import Path

import pandas as pd

from features import KIND, make_features
from model import fit, predict
from validate import baseline_plan

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_PATH = ROOT / "submission.csv"

#Вариант модели выбирается по результатам validate.py
OBJECTIVE = "l2"
LOG_TARGET = True
USE_PLAN = False

#Дни, про которые данные не дают однозначного ответа: {дата: вес сценария "праздник"}.
#11.05.2026 - официальный выходной (перенос с субботы 9 мая). В истории такой перенос
#был один раз (10.03.2025) и снятия были обычными, но воскресенье 10.05.2026 уже
#аномально низкое. Поэтому берём середину между двумя сценариями: так ошибка в этот
#день ограничена половиной разницы, какой бы сценарий ни сбылся.
UNCERTAIN_HOLIDAYS = {"2026-05-11": 0.5}


def predict_with_scenarios(model, test: pd.DataFrame, fallback: pd.Series) -> pd.Series:
    """Прогноз модели; для спорных дней - взвешенное среднее сценариев "обычный день" и "праздник"."""
    forecast = predict(model, test, fallback)
    for day, weight in UNCERTAIN_HOLIDAYS.items():
        mask = test["date"] == pd.Timestamp(day)
        as_holiday = test[mask].copy()
        as_holiday["hol_kind"], as_holiday["hol_day"] = KIND["minor"], 1
        holiday_forecast = predict(model, as_holiday, fallback[mask])
        forecast[mask] = (1 - weight) * forecast[mask] + weight * holiday_forecast
    return forecast


def main() -> None:
    panel = pd.read_parquet(DATA_DIR / "panel.parquet")
    df = make_features(panel)
    train, test = df[~df["is_forecast"]], df[df["is_forecast"]]

    model = fit(train, OBJECTIVE, LOG_TARGET, USE_PLAN)
    forecast = predict_with_scenarios(model, test, fallback=baseline_plan(train, test))
    result = pd.DataFrame({
        "branch_id": test["branch_id"].astype(str),
        "date": test["date"],
        "forecast_uzs": forecast.round().astype("int64"),
    })

    #порядок строк и набор (филиал, дата) - строго как в шаблоне
    template = pd.read_csv(DATA_DIR / "submission_template.csv", parse_dates=["date"])
    submission = template[["branch_id", "date"]].merge(result, on=["branch_id", "date"], how="left")

    assert len(submission) == len(template), "число строк не совпало с шаблоном"
    assert submission["forecast_uzs"].notna().all(), "есть строки без прогноза"
    assert (submission["forecast_uzs"] >= 0).all(), "есть отрицательный прогноз"

    submission["date"] = submission["date"].dt.strftime("%Y-%m-%d")
    submission.to_csv(OUT_PATH, index=False)

    last30 = train[train["date"] > train["date"].max() - pd.Timedelta(days=30)]
    print(f"сохранено: {OUT_PATH}, строк: {len(submission)}")
    print(f"прогноз, всего: {forecast.sum() / 1e9:,.1f} млрд сум")
    print(f"факт за последние 30 дней истории: {last30['y'].sum() / 1e9:,.1f} млрд сум")
    print(f"план банка на период прогноза: {test['planned_cash_load_uzs'].sum() / 1e9:,.1f} млрд сум")

    by_day = pd.DataFrame({
        "date": test["date"], "прогноз": forecast / 1e9, "план": test["planned_cash_load_uzs"] / 1e9,
        "праздник": test["hol_kind"], "разгон": test["pre_days"], "зарплата": test["payday_kind"],
    }).groupby("date").agg({"прогноз": "sum", "план": "sum", "праздник": "max", "разгон": "max", "зарплата": "max"})
    by_day.insert(0, "день", by_day.index.day_name().str[:3])
    by_day["спорный"] = by_day.index.strftime("%Y-%m-%d").isin(UNCERTAIN_HOLIDAYS).astype(int)
    print("\nПо дням, млрд сум:")
    print(by_day.round(1).to_string())

    importance = pd.Series(model.feature_importances_, index=model.feature_name_).sort_values(ascending=False)
    print("\nВажность признаков (число разбиений):")
    print(importance.to_string())


if __name__ == "__main__":
    main()