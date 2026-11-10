"""
Что делает:
1. берёт дневную сумму снятий (ATM + касса) - это целевая переменная y;
2. строит полную сетку дат, чтобы пропущенные дни были видны явно;
3. присоединяет cash_ops и справочник филиалов;
4. восстанавливает y по балансу наличных там, где записи в журнале потеряны.

Запуск: python src/prepare.py
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = DATA_DIR / "panel.parquet"

# Периоды, когда наличные из филиалов уходили, а операций в журнале нет
# (найдено в notebooks/eda.ipynb, раздел 6). None = все филиалы.
LOSS_WINDOWS = [
    ("2024-07-01", "2024-08-15", None),  # нет записей банкоматов
    ("2025-11-10", "2025-11-14", ["B051", "B052", "B053", "B054"]),  # Xorazm, нет записей вообще
]


def load_target() -> pd.DataFrame:
    """Дневная сумма снятий по филиалу."""
    daily = pd.read_parquet(DATA_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    return (
        daily[daily["txn_type"] == "withdrawal"]
        .groupby(["branch_id", "date"], as_index=False)["amount_uzs"].sum()
        .rename(columns={"amount_uzs": "y_raw"})
    )


def build_grid(target: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """Каждый филиал из шаблона x каждый день от его первой операции до конца прогноза."""
    first_day = target.groupby("branch_id")["date"].min()
    last_day = template["date"].max()
    parts = [
        pd.DataFrame({"branch_id": b, "date": pd.date_range(first_day[b], last_day)})
        for b in sorted(template["branch_id"].unique())
    ]
    return pd.concat(parts, ignore_index=True)


def restore_lost_days(panel: pd.DataFrame) -> pd.DataFrame:
    """В периоды потери записей берём снятия из баланса: план + довоз - инкассация."""
    balance = (
        panel["planned_cash_load_uzs"]
        + panel["extra_cash_load_uzs"]
        - panel["cash_collected_eod_uzs"]
    )
    lost = pd.Series(False, index=panel.index)
    for start, end, branch_ids in LOSS_WINDOWS:
        in_window = panel["date"].between(start, end)
        if branch_ids is not None:
            in_window &= panel["branch_id"].isin(branch_ids)
        lost |= in_window

    panel["y"] = panel["y_raw"].where(~lost, balance)
    panel["y_restored"] = lost
    return panel


def main() -> None:
    target = load_target()
    branches = pd.read_csv(DATA_DIR / "branches.csv", parse_dates=["open_date"])
    cash = pd.read_csv(DATA_DIR / "cash_ops.csv", parse_dates=["date"])
    template = pd.read_csv(DATA_DIR / "submission_template.csv", parse_dates=["date"])

    hist_end = target["date"].max()

    panel = (
        build_grid(target, template)
        .merge(target, on=["branch_id", "date"], how="left")
        .merge(cash, on=["branch_id", "date"], how="left")
        .merge(branches, on="branch_id", how="left")
    )
    panel["is_forecast"] = panel["date"] > hist_end
    panel = restore_lost_days(panel)
    panel = panel.sort_values(["branch_id", "date"]).reset_index(drop=True)
    panel.to_parquet(OUT_PATH, index=False)

    hist = panel[~panel["is_forecast"]]
    fcst = panel[panel["is_forecast"]]
    print(f"сохранено: {OUT_PATH}")
    print(f"филиалов: {panel['branch_id'].nunique()}, строк: {len(panel)}")
    print(f"история: {hist['date'].min().date()} - {hist['date'].max().date()}, строк: {len(hist)}")
    print(f"прогноз: {fcst['date'].min().date()} - {fcst['date'].max().date()}, строк: {len(fcst)}")
    print(f"восстановлено по балансу: {int(hist['y_restored'].sum())} дней")
    print(f"история без y после восстановления: {int(hist['y'].isna().sum())} дней")
    print(f"прогноз без плана загрузки: {int(fcst['planned_cash_load_uzs'].isna().sum())} дней")

    # Контроль: в обычные дни баланс должен совпадать с журналом
    ok = hist[~hist["y_restored"] & (hist["y_raw"] > 0)]
    balance = ok["planned_cash_load_uzs"] + ok["extra_cash_load_uzs"] - ok["cash_collected_eod_uzs"]
    err = (balance / ok["y_raw"] - 1).abs()
    print(f"расхождение баланса и журнала в обычные дни: медиана {err.median():.2%}, 95% дней до {err.quantile(0.95):.2%}")

    # Контроль: сумма снятий до и после восстановления в затронутые дни
    fixed = hist[hist["y_restored"]]
    print(f"снятия в восстановленные дни: было {fixed['y_raw'].sum() / 1e9:,.1f} млрд, стало {fixed['y'].sum() / 1e9:,.1f} млрд")


if __name__ == "__main__":
    main()