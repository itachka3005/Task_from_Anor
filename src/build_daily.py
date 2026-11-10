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


def build_daily(raw: pl.LazyFrame) -> pl.DataFrame:
    return (
        raw.group_by(
            "branch_id",
            pl.col("ts").dt.date().alias("date"),
            "channel",
            "txn_type",
            "client_segment",
        )
        .agg(
            pl.col("amount_uzs").sum().alias("amount_uzs"),
            pl.len().alias("n_txn"),
        )
        .sort("branch_id", "date", "channel", "txn_type", "client_segment")
        .collect(engine="streaming")
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


if __name__ == "__main__":
    main()