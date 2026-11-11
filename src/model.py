"""Модель: LightGBM предсказывает долю ratio = y / level, прогноз = ratio * level.

Почему доля, а не сумма:
- филиалы отличаются по масштабу в разы, а недельный и праздничный рисунок у них похож;
- снятия растут год от года, а деревья не умеют продолжать тренд. Уровень филиала
  берёт рост на себя, модели остаётся календарный рисунок.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES, PLAN_FEATURE

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


def feature_list(use_plan: bool) -> list:
    return FEATURES if use_plan else [f for f in FEATURES if f != PLAN_FEATURE]


def fit(train: pd.DataFrame, objective: str = "l1", log_target: bool = False,
        use_plan: bool = True) -> lgb.LGBMRegressor:
    """Обучение на строках, где известны и уровень, и факт.

    Вес строки = уровень филиала: тогда ошибка доли, умноженная на вес, - это ошибка
    в сумах, то есть ровно числитель WAPE.

    log_target=True: модель учит логарифм доли. Эффекты в данных перемножаются
    (зарплата x канун праздника), а в логарифмах умножение превращается в сложение,
    которое бустинг строит естественно. Так модель лучше справляется с сочетаниями,
    которых почти не было в истории.
    """
    rows = train.dropna(subset=["ratio", "level"])
    rows = rows[rows["ratio"] > 0]
    target = np.log(rows["ratio"]) if log_target else rows["ratio"]
    model = lgb.LGBMRegressor(objective=objective, **PARAMS)
    model.features_ = feature_list(use_plan)
    model.fit(rows[model.features_], target, sample_weight=rows["level"])
    model.log_target_ = log_target
    return model


def predict(model: lgb.LGBMRegressor, test: pd.DataFrame, fallback: pd.Series) -> pd.Series:
    """Прогноз в сумах. Где уровень неизвестен (слишком новый филиал) - берём fallback."""
    raw = model.predict(test[model.features_])
    ratio = np.exp(raw) if model.log_target_ else raw
    pred = pd.Series(ratio, index=test.index) * test["level"]
    return pred.clip(lower=0).fillna(fallback)


def make_lgbm(objective: str, log_target: bool = False, use_plan: bool = True,
              template_plan: bool = False):
    """Функция (train, test) -> прогноз, в том же виде, что и бейзлайны в validate.py.

    template_plan=True: в тесте настоящий план заменяется недельным шаблоном, посчитанным
    по обучению. Так проверяем модель в тех же условиях, что и в бою: на период прогноза
    план банка - это одинаковый шаблон по дням недели, без поправок на конкретный день.
    """
    from validate import baseline_plan, recent  # запасной прогноз и окно последних недель

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        model = fit(train, objective, log_target, use_plan)
        if template_plan:
            last = recent(train)
            template = last.groupby(
                [last["branch_id"].astype(str), last["dow"]]
            )["planned_cash_load_uzs"].median()
            keys = pd.MultiIndex.from_arrays([test["branch_id"].astype(str), test["dow"]])
            plan = pd.Series(template.reindex(keys).values, index=test.index)
            test = test.assign(plan_ratio=plan / test["level"])
        return predict(model, test, fallback=baseline_plan(train, test))

    return fit_predict


def make_xgb():
    """XGBoost с теми же признаками, целью (логарифм доли) и весами, что и итоговый LightGBM.
    Нужен только для сравнения двух библиотек в validate.py."""
    import xgboost as xgb
    from validate import baseline_plan

    features = feature_list(use_plan=False)

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        rows = train.dropna(subset=["ratio", "level"])
        rows = rows[rows["ratio"] > 0]
        model = xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", enable_categorical=True, random_state=42,
        )
        model.fit(rows[features], np.log(rows["ratio"]), sample_weight=rows["level"])
        pred = pd.Series(np.exp(model.predict(test[features])), index=test.index) * test["level"]
        return pred.clip(lower=0).fillna(baseline_plan(train, test))

    return fit_predict