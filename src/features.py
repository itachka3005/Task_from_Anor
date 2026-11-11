"""Признаки для модели. Все считаются одинаково для истории и для периода прогноза.

Главное правило: признаки, которые зависят от прошлых снятий, смотрят только
на данные старше HORIZON дней. Прогноз строится на 30 дней вперёд, значит,
для последнего дня прогноза самые свежие известные снятия - 30 дней назад.
Так в признаки не может попасть будущее ни на валидации, ни в бою.

Календарь праздников. Даты берём из внешних источников, а то, сколько дней
длится эффект, - из самих данных (reports/calendar_check.csv):
- даты Рамазан хайита и Курбан хайита: пакет `holidays` (Узбекистан);
- остальные праздники имеют фиксированные даты;
- официальные переносы и дополнительные выходные из пакета НЕ используем:
  в данных часть таких дней обычная (например, 30.12.2024 - пик, а не выходной).
"""
import holidays
import numpy as np
import pandas as pd

HORIZON = 30        #горизонт прогноза, дней
LEVEL_WINDOW = 28   # кно для оценки уровня филиала, дней

#Малые праздники: один день, снятия падают примерно до 0,4 от нормы.
#День независимости в данных длится два дня (1 и 2 сентября).
MINOR_HOLIDAYS = [(3, 8), (5, 9), (9, 1), (9, 2), (10, 1), (12, 8)]
#Большие праздники: блок из 3 дней (снятия ~0,2 от нормы) и разгон за 3 дня до него.
BLOCK_DAYS = 3
RAMP_DAYS = 3
KIND = {"minor": 1, "navruz": 2, "fitr": 3, "adha": 4, "newyear": 5}

PAYDAYS = {5: 1, 20: 2}  #число месяца -> вид выплаты (пик 5-го сильнее, чем 20-го)

CATEGORICAL = ["branch_id", "region", "branch_type"]
FEATURES = CATEGORICAL + [
    "dow", "dom", "month", "days_to_month_end",
    "hol_kind", "hol_day", "pre_kind", "pre_days",
    "payday_kind", "is_payday_nominal",
    "plan_ratio", "wd_ratio", "log_level", "age_days",
]


def eid_dates(years) -> dict:
    """Первые дни Рамазан хайита (fitr) и Курбан хайита (adha) из пакета holidays."""
    uz = holidays.country_holidays("UZ", years=years, language="en_US")
    found = {"fitr": [], "adha": []}
    for day, name in uz.items():
        for part in name.split("; "):
            base = part.replace(" (estimated)", "")
            if base == "Eid al-Fitr":
                found["fitr"].append(pd.Timestamp(day))
            elif base == "Eid al-Adha":
                found["adha"].append(pd.Timestamp(day))
    return found


def holiday_blocks(years) -> list:
    """Список (вид праздника, [дни блока]) для больших праздников."""
    blocks = []
    for year in years:
        #Навруз: 21-23 марта
        start = pd.Timestamp(year, 3, 21)
        blocks.append(("navruz", list(pd.date_range(start, periods=BLOCK_DAYS))))
        #Новый год: 31 декабря - 2 января плюс выходные, которые примыкают к блоку
        days = list(pd.date_range(pd.Timestamp(year - 1, 12, 31), periods=BLOCK_DAYS))
        while (days[0] - pd.Timedelta(days=1)).day_of_week >= 5:
            days.insert(0, days[0] - pd.Timedelta(days=1))
        while (days[-1] + pd.Timedelta(days=1)).day_of_week >= 5:
            days.append(days[-1] + pd.Timedelta(days=1))
        blocks.append(("newyear", days))
    #хайиты: 3 дня начиная с первого дня праздника
    for kind, starts in eid_dates(years).items():
        for start in starts:
            blocks.append((kind, list(pd.date_range(start, periods=BLOCK_DAYS))))
    #хайиты записываем последними: при наложении на Навруз (март 2026) они важнее
    return sorted(blocks, key=lambda b: b[0] in ("fitr", "adha"))


def build_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Таблица по датам: день недели, праздники, разгон перед праздником, зарплатные дни."""
    cal = pd.DataFrame({"date": pd.date_range(start, end)}).set_index("date", drop=False)
    cal["dow"] = cal["date"].dt.day_of_week
    cal["dom"] = cal["date"].dt.day
    cal["month"] = cal["date"].dt.month
    cal["days_to_month_end"] = cal["date"].dt.days_in_month - cal["dom"]
    years = range(start.year, end.year + 2)

    hol_kind = pd.Series(0, index=cal.index)   #вид праздника в этот день
    hol_day = pd.Series(0, index=cal.index)    #номер дня внутри праздника
    pre_kind = pd.Series(0, index=cal.index)   #к какому празднику идёт разгон
    pre_days = pd.Series(0, index=cal.index)   #сколько дней до начала праздника (1-3)

    #малые праздники; если выпал на воскресенье, эффект есть и в понедельник
    for year in years:
        for month, day in MINOR_HOLIDAYS:
            d = pd.Timestamp(year, month, day)
            if d in cal.index:
                hol_kind[d], hol_day[d] = KIND["minor"], 1
        for month, day in MINOR_HOLIDAYS:
            d = pd.Timestamp(year, month, day)
            nxt = d + pd.Timedelta(days=1)
            if d.day_of_week == 6 and nxt in cal.index and hol_kind[nxt] == 0:
                hol_kind[nxt], hol_day[nxt] = KIND["minor"], 1

    #большие праздники: блок из нескольких дней
    blocks = holiday_blocks(years)
    for kind, days in blocks:
        for i, d in enumerate(days, start=1):
            if d in cal.index:
                hol_kind[d], hol_day[d] = KIND[kind], i

    #разгон: 3 дня перед началом большого праздника
    for kind, days in blocks:
        for k in range(1, RAMP_DAYS + 1):
            d = days[0] - pd.Timedelta(days=k)
            if d in cal.index and hol_kind[d] <= KIND["minor"] and (pre_days[d] == 0 or k < pre_days[d]):
                pre_kind[d], pre_days[d] = KIND[kind], k
    pre_kind[hol_kind > KIND["minor"]] = 0
    pre_days[hol_kind > KIND["minor"]] = 0

    cal["hol_kind"], cal["hol_day"] = hol_kind, hol_day
    cal["pre_kind"], cal["pre_days"] = pre_kind, pre_days

    #зарплатные дни: 5-е и 20-е; с выходного или праздника - на рабочий день раньше
    cal["is_payday_nominal"] = cal["dom"].isin(PAYDAYS).astype(int)
    non_working = (cal["dow"] >= 5) | (cal["hol_kind"] > 0)
    payday_kind = pd.Series(0, index=cal.index)
    for d in cal.index[cal["is_payday_nominal"] == 1]:
        kind, shifted = PAYDAYS[d.day], d
        while shifted in cal.index and non_working[shifted]:
            shifted -= pd.Timedelta(days=1)
        if shifted in cal.index:
            payday_kind[shifted] = kind
    cal["payday_kind"] = payday_kind

    #простые флаги для отчётов
    cal["is_holiday"] = (cal["hol_kind"] > 0).astype(int)
    cal["is_pre_holiday"] = (cal["pre_days"] > 0).astype(int)
    cal["is_payday"] = (cal["payday_kind"] > 0).astype(int)
    return cal.reset_index(drop=True)


def add_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Уровень филиала и его недельный профиль - только по данным старше HORIZON дней."""
    df = df.sort_values(["branch_id", "date"]).copy()
    #обычный день: не праздник, не разгон, не зарплата. Только по ним считаем уровень и профиль.
    plain = (df["is_holiday"] == 0) & (df["is_pre_holiday"] == 0) & (df["is_payday"] == 0)
    df["_y_weekday"] = df["y"].where(plain & (df["dow"] < 5))
    df["_y_plain"] = df["y"].where(plain)

    by_branch = df.groupby("branch_id", sort=False, observed=True)
    #уровень: медиана обычных будней за 4 недели, заканчивающиеся 30 дней назад
    df["level"] = by_branch["_y_weekday"].transform(
        lambda s: s.shift(HORIZON).rolling(LEVEL_WINDOW, min_periods=10).median()
    )
    #недельный профиль: среднее того же дня недели за 8 недель (сдвиг 5 недель = 35 дней)
    wd_mean = df.groupby(["branch_id", "dow"], sort=False, observed=True)["_y_plain"].transform(
        lambda s: s.shift(5).rolling(8, min_periods=3).mean()
    )
    df["wd_ratio"] = wd_mean / df["level"]
    df["plan_ratio"] = df["planned_cash_load_uzs"] / df["level"]
    df["log_level"] = np.log(df["level"])
    df["age_days"] = (df["date"] - by_branch["date"].transform("min")).dt.days.clip(upper=365)
    return df.drop(columns=["_y_weekday", "_y_plain"])


def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Панель (филиал x день) -> та же таблица с признаками и целевой долей ratio = y / level."""
    cal = build_calendar(panel["date"].min(), panel["date"].max())
    df = panel.merge(cal, on="date", how="left")
    df = add_level_features(df)
    df["ratio"] = df["y"] / df["level"]
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df.reset_index(drop=True)