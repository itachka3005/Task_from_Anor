"""Признаки для модели. Все считаются одинаково для истории и для периода прогноза.

Главное правило: признаки, которые зависят от прошлых снятий, смотрят только
на данные старше HORIZON дней. Прогноз строится на 30 дней вперёд, значит,
для последнего дня прогноза самые свежие известные снятия - 30 дней назад.
Так в признаки не может попасть будущее ни на валидации, ни в бою.

Источник календаря праздников: пакет `holidays` (Узбекистан, включая переносы
и дополнительные выходные по указам Президента).
"""
import holidays
import numpy as np
import pandas as pd

HORIZON = 30        #горизонт прогноза, дней
LEVEL_WINDOW = 28   #окно для оценки уровня филиала, дней
PAYDAYS = (5, 20)   #числа месяца с пиком снятий (видно в EDA)

CATEGORICAL = ["branch_id", "region", "branch_type"]
FEATURES = CATEGORICAL + [
    "dow", "dom", "month", "days_to_month_end",
    "is_off", "off_kind", "is_pre_off", "is_post_off", "days_to_off", "days_since_off",
    "is_payday", "is_payday_nominal", "days_to_payday", "days_since_payday",
    "plan_ratio", "wd_ratio", "log_level", "age_days",
]


def build_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Таблица по датам: выходные, праздники, зарплатные дни."""
    cal = pd.DataFrame({"date": pd.date_range(start, end)})
    cal["dow"] = cal["date"].dt.day_of_week
    cal["dom"] = cal["date"].dt.day
    cal["month"] = cal["date"].dt.month
    cal["days_to_month_end"] = cal["date"].dt.days_in_month - cal["dom"]

    #праздники и дополнительные выходные
    uz = holidays.country_holidays("UZ", years=range(start.year, end.year + 1))
    names = cal["date"].map(lambda d: uz.get(d, ""))
    moved = names.str.contains("koʻchirilgan|qoʻshimcha", regex=True)
    cal["off_kind"] = np.select([names == "", moved], [0, 2], default=1)  #1 праздник,2 перенос/доп.выходной
    cal["is_off"] = (cal["off_kind"] > 0).astype(int)
    cal["is_pre_off"] = ((cal["is_off"].shift(-1, fill_value=0) == 1) & (cal["is_off"] == 0)).astype(int)
    cal["is_post_off"] = ((cal["is_off"].shift(1, fill_value=0) == 1) & (cal["is_off"] == 0)).astype(int)
    cal["days_to_off"] = _days_to_next(cal["is_off"])
    cal["days_since_off"] = _days_to_next(cal["is_off"][::-1])[::-1]

    #зарплатные дни: 5-е и 20-е, с выходного сдвигаются на рабочий день раньше
    cal["is_payday_nominal"] = cal["dom"].isin(PAYDAYS).astype(int)
    non_working = (cal["dow"] >= 5) | (cal["is_off"] == 1)
    payday = pd.Series(0, index=cal.index)
    for i in cal.index[cal["is_payday_nominal"] == 1]:
        j = i
        while j > 0 and non_working[j]:
            j -= 1
        payday[j] = 1
    cal["is_payday"] = payday
    cal["days_to_payday"] = _days_to_next(cal["is_payday"])
    cal["days_since_payday"] = _days_to_next(cal["is_payday"][::-1])[::-1]
    return cal


def _days_to_next(flag: pd.Series, cap: int = 10) -> pd.Series:
    """Сколько дней до ближайшего дня с flag == 1 (0 в сам день), не больше cap."""
    out, dist = [], cap
    for value in flag.iloc[::-1]:
        dist = 0 if value == 1 else min(dist + 1, cap)
        out.append(dist)
    return pd.Series(out[::-1], index=flag.index)


def add_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Уровень филиала и его недельный профиль - только по данным старше HORIZON дней."""
    df = df.sort_values(["branch_id", "date"]).copy()
    #обычный рабочий день: пн-пт, не праздник. По ним считаем уровень.
    regular = (df["dow"] < 5) & (df["is_off"] == 0)
    df["_y_regular"] = df["y"].where(regular)
    df["_y_normal"] = df["y"].where((df["is_off"] == 0) & (df["is_pre_off"] == 0))

    by_branch = df.groupby("branch_id", sort=False)
    #уровень: медиана обычных рабочих дней за 4 недели, заканчивающиеся 30 дней назад
    df["level"] = by_branch["_y_regular"].transform(
        lambda s: s.shift(HORIZON).rolling(LEVEL_WINDOW, min_periods=10).median()
    )
    #недельный профиль: среднее того же дня недели за 8 недель (сдвиг 5 недель = 35 дней)
    wd_mean = df.groupby(["branch_id", "dow"], sort=False)["_y_normal"].transform(
        lambda s: s.shift(5).rolling(8, min_periods=3).mean()
    )
    df["wd_ratio"] = wd_mean / df["level"]
    df["plan_ratio"] = df["planned_cash_load_uzs"] / df["level"]
    df["log_level"] = np.log(df["level"])
    df["age_days"] = (df["date"] - by_branch["date"].transform("min")).dt.days.clip(upper=365)
    return df.drop(columns=["_y_regular", "_y_normal"])


def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Панель (филиал x день) -> та же таблица с признаками и целевой долей ratio = y / level."""
    cal = build_calendar(panel["date"].min(), panel["date"].max() + pd.Timedelta(days=15))
    df = panel.merge(cal, on="date", how="left")
    df = add_level_features(df)
    df["ratio"] = df["y"] / df["level"]
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df.reset_index(drop=True)