"""Модель: LightGBM предсказывает долю ratio = y / level, прогноз = ratio * level.

Почему доля, а не сумма:
- филиалы отличаются по масштабу в разы, а недельный и праздничный рисунок у них похож;
- снятия растут год от года, а деревья не умеют продолжать тренд. Уровень филиала
  берёт рост на себя, модели остаётся календарный рисунок.
"""
import lightgbm as lgb
import pandas as pd

from features import FEATURES

PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)


def fit(train: pd.DataFrame, objective: str = "l1") -> lgb.LGBMRegressor:
    """Обучение на строках, где известны и уровень, и факт.

    Вес строки = уровень филиала: тогда ошибка доли, умноженная на вес, - это ошибка
    в сумах, то есть ровно числитель WAPE.
    """
    rows = train.dropna(subset=["ratio", "level"])
    model = lgb.LGBMRegressor(objective=objective, **PARAMS)
    model.fit(rows[FEATURES], rows["ratio"], sample_weight=rows["level"])
    return model


def predict(model: lgb.LGBMRegressor, test: pd.DataFrame, fallback: pd.Series) -> pd.Series:
    """Прогноз в сумах. Где уровень неизвестен (слишком новый филиал) - берём fallback."""
    pred = pd.Series(model.predict(test[FEATURES]), index=test.index) * test["level"]
    return pred.clip(lower=0).fillna(fallback)


def make_lgbm(objective: str):
    """Функция (train, test) -> прогноз, в том же виде, что и бейзлайны в validate.py."""
    from validate import baseline_plan  # запасной прогноз для филиалов без уровня

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        return predict(fit(train, objective), test, fallback=baseline_plan(train, test))

    return fit_predict