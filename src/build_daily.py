"""
Убираем дубли txn_id и тестовые операции (internal_test).
Запуск: python src/build_daily.py
"""
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_PATH = DATA_DIR / "transactions.csv"
OUT_PATH = DATA_DIR / "daily.parquet"

SCHEMA = {
    "txn_id": pl.Int64,
    "txn_ts": pl.String,
    "branch_id": pl.String,
    "channel": pl.String,
    "txn_type": pl.String,
    "amount_uzs": pl.Float64,
    "client_segment": pl.String,
}

KEYS = ["branch_id", "date", "channel", "txn_type", "client_segment"]


def scan_raw() -> pl.LazyFrame:
    return pl.scan_csv(RAW_PATH, schema=SCHEMA).with_columns(
        pl.col("txn_ts").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("ts")
    )


def quality_report(raw: pl.LazyFrame) -> None:
    stats = raw.select(
        pl.len().alias("rows"),
        pl.col("txn_id").n_unique().alias("unique_txn_id"),
        pl.col("ts").null_count().alias("bad_or_null_ts"),
        pl.col("branch_id").null_count().alias("null_branch"),
        pl.col("amount_uzs").null_count().alias("null_amount"),
        (pl.col("amount_uzs") <= 0).sum().alias("non_positive_amount"),
        pl.col("amount_uzs").max().alias("max_amount"),
        pl.col("ts").min().alias("ts_min"),
        pl.col("ts").max().alias("ts_max"),
    ).collect(engine="streaming")

    print("Качество сырых данных")
    for name, value in stats.row(0, named=True).items():
        print(f"{name:>22}: {value}")
    rows, unique = stats["rows"][0], stats["unique_txn_id"][0]
    print(f"{'duplicate_txn_id':>22}: {rows - unique}")


def aggregate(txns: pl.LazyFrame) -> pl.LazyFrame:
    """Транзакции -> суммы и количество по (филиал, день, канал, тип, сегмент)."""
    return (
        txns.with_columns(pl.col("ts").dt.date().alias("date"))
        .group_by(KEYS)
        .agg(
            pl.col("amount_uzs").sum().alias("amount_uzs"),
            pl.len().alias("n_txn"),
        )
    )


def build_daily(raw: pl.LazyFrame) -> pl.DataFrame:
    # тестовые операции сотрудников банка - не клиентский спрос
    raw = raw.filter(pl.col("client_segment") != "internal_test")

    dup_ids = (
        raw.group_by("txn_id").len().filter(pl.col("len") > 1).select("txn_id")
        .collect(engine="streaming")
        .lazy()
    )
    singles = aggregate(raw.join(dup_ids, on="txn_id", how="anti")).collect(engine="streaming")

    dups = (
        raw.join(dup_ids, on="txn_id", how="semi").collect(engine="streaming")
        .unique(subset="txn_id", keep="first")
    )
    deduped = aggregate(dups.lazy()).collect()

    return (
        pl.concat([singles, deduped])
        .group_by(KEYS)
        .agg(pl.col("amount_uzs").sum(), pl.col("n_txn").sum())
        .sort(KEYS)
    )


def main() -> None:
    raw = scan_raw()
    quality_report(raw)

    daily = build_daily(raw)
    daily.write_parquet(OUT_PATH)

    print("\nДневная таблица")
    print(f"строк: {daily.height}, сохранено в {OUT_PATH}")
    for col in ("channel", "txn_type", "client_segment"):
        print(f"\n{col}:")
        print(daily.group_by(col).agg(pl.col("n_txn").sum()).sort("n_txn", descending=True))
    print(f"\nфилиалов: {daily['branch_id'].n_unique()}")
    print(f"операций после чистки: {daily['n_txn'].sum()}")


if __name__ == "__main__":
    main()