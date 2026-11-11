"""Диагностика: где календарь расходится с данными и в какие дни ошибается модель.

Сохраняет две таблицы в reports/:
- calendar_check.csv - все необычные дни истории: насколько снятия отличаются от нормы
  для этого дня недели и что про этот день говорит календарь;
- fold_errors.csv - факт, прогноз и ошибка по каждому дню трёх окон валидации.

Запуск: python src/diagnose.py
"""
from pathlib import Path

import holidays
import pandas as pd

from model import make_lgbm
from validate import FOLDS, load_history, split

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
OBJECTIVE = "l2"


def calendar_check(df: pd.DataFrame) -> pd.DataFrame:
    """Отношение суммарных снятий к медиане того же дня недели за +-4 недели
    и то, что про этот день говорит наш календарь (features.build_calendar)."""
    flags = ["hol_kind", "hol_day", "pre_kind", "pre_days", "payday_kind", "is_payday_nominal"]
    days = df.groupby("date").agg(total=("y", "sum"), **{f: (f, "first") for f in flags})
    days["weekday"] = days.index.day_name().str[:3]
    days["dom"] = days.index.day
    norm = days.groupby(days.index.day_of_week)["total"].transform(
        lambda s: s.rolling(9, center=True, min_periods=5).median()
    )
    days["ratio_to_norm"] = (days["total"] / norm).round(3)
    years = range(days.index.year.min(), days.index.year.max() + 1)
    uz = holidays.country_holidays("UZ", years=years, language="en_US")
    days["official_name"] = [uz.get(d, "") for d in days.index]

    unusual = (days["ratio_to_norm"] < 0.8) | (days["ratio_to_norm"] > 1.2)
    flagged = (days[["hol_kind", "pre_days", "payday_kind"]] > 0).any(axis=1)
    days["status"] = "ok"
    days.loc[unusual & ~flagged, "status"] = "необычный день без флага"
    days.loc[flagged & days["ratio_to_norm"].between(0.85, 1.15), "status"] = "флаг есть, день обычный"
    cols = ["weekday", "dom", "ratio_to_norm"] + flags + ["official_name", "status"]
    return days.loc[unusual | flagged | (days["official_name"] != ""), cols]


def fold_errors(df: pd.DataFrame) -> pd.DataFrame:
    """По каждому дню окон валидации: факт, прогноз, сумма модулей ошибок по филиалам."""
    fit_predict = make_lgbm("l2", log_target=True, use_plan=False)
    parts = []
    for fold_name, cutoff in FOLDS.items():
        train, test = split(df, cutoff)
        pred = fit_predict(train, test)
        part = pd.DataFrame({
            "fold": fold_name[0], "date": test["date"], "actual": test["y"], "pred": pred,
            "abs_err": (test["y"] - pred).abs(),
            "is_holiday": test["is_holiday"], "is_pre_holiday": test["is_pre_holiday"],
            "is_payday": test["is_payday"],
        })
        parts.append(part)
    rows = pd.concat(parts)
    by_day = rows.groupby(["fold", "date"]).agg(
        actual=("actual", "sum"), pred=("pred", "sum"), abs_err=("abs_err", "sum"),
        is_holiday=("is_holiday", "max"), is_pre_holiday=("is_pre_holiday", "max"),
        is_payday=("is_payday", "max"),
    ).reset_index()
    by_day.insert(2, "weekday", by_day["date"].dt.day_name().str[:3])
    by_day["pred_to_actual"] = (by_day["pred"] / by_day["actual"]).round(3)
    by_day["wape_day"] = (by_day["abs_err"] / by_day["actual"]).round(3)
    #доля дня в общей ошибке окна: где теряем больше всего
    by_day["share_of_fold_error"] = (
        by_day["abs_err"] / by_day.groupby("fold")["abs_err"].transform("sum")
    ).round(3)
    for col in ("actual", "pred", "abs_err"):
        by_day[col] = (by_day[col] / 1e9).round(2)  #млрд сум
    return by_day


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    df = load_history()

    cal = calendar_check(df)
    cal.to_csv(REPORTS_DIR / "calendar_check.csv")
    print(f"calendar_check.csv: {len(cal)} дней")
    print("Расхождения календаря с данными:")
    print(cal[cal["status"] != "ok"].drop(columns=["official_name"]).to_string())

    errors = fold_errors(df)
    errors.to_csv(REPORTS_DIR / "fold_errors.csv", index=False)
    print(f"fold_errors.csv: {len(errors)} дней")

    print("\nДесять дней с наибольшим вкладом в ошибку, по окнам:")
    top = errors.sort_values("share_of_fold_error", ascending=False).groupby("fold").head(10)
    print(top.sort_values(["fold", "share_of_fold_error"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()