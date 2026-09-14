from datasets import load_dataset
import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

dataset = load_dataset(
    "Dingdong-Inc/FreshRetailNet-50K",
    split="train[:50000]"
)

df: pd.DataFrame = dataset.to_pandas()

print(f"Dataset loaded successfully.")
print(f"Shape: {df.shape}")


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)

print("\nTotal missing values:", missing.sum())


# ============================================================
# 4. UNIQUE VALUES
# ============================================================

print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)

for column in df.columns:

    try:
        unique_count = df[column].nunique()

        print(
            f"{column:25} : "
            f"{unique_count} unique values"
        )

    except TypeError:
        print(
            f"{column:25} : "
            f"contains list/array values"
        )


# ============================================================
# 5. DATE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATE INFORMATION")
print("=" * 70)

try:

    df["dt"] = pd.to_datetime(
        df["dt"],
        errors="coerce"
    )

    print("Minimum date:", df["dt"].min())
    print("Maximum date:", df["dt"].max())
    print(
        "Unique dates:",
        df["dt"].nunique()
    )

except Exception as e:

    print("Could not process date column:")
    print(e)


# ============================================================
# 6. STORE / PRODUCT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("STORE / PRODUCT INFORMATION")
print("=" * 70)

for column in [
    "city_id",
    "store_id",
    "product_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id"
]:

    if column in df.columns:

        print(
            f"{column:25} : "
            f"{df[column].nunique()} unique"
        )


# ============================================================
# 7. SALES INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("SALES INFORMATION")
print("=" * 70)

if "sale_amount" in df.columns:

    print(df["sale_amount"].describe())

    print(
        "\nZero-sales rows:",
        (df["sale_amount"] == 0).sum()
    )

    print(
        "Negative-sales rows:",
        (df["sale_amount"] < 0).sum()
    )


# ============================================================
# 8. STOCKOUT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("STOCKOUT INFORMATION")
print("=" * 70)

if "stock_hour6_22_cnt" in df.columns:

    stock = df["stock_hour6_22_cnt"]

    print(stock.describe())

    stockout_rows = (stock > 0).sum()

    print(
        "\nRows with stockout:",
        stockout_rows
    )

    print(
        "Percentage with stockout:",
        round(
            stockout_rows / len(df) * 100,
            2
        ),
        "%"
    )


# ============================================================
# 9. PROMOTION / HOLIDAY INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("PROMOTION / HOLIDAY INFORMATION")
print("=" * 70)

for column in [
    "discount",
    "holiday_flag",
    "activity_flag"
]:

    if column in df.columns:

        print(f"\n{column}:")

        try:
            print(
                df[column]
                .value_counts(dropna=False)
                .head(20)
            )

        except Exception as e:

            print(
                "Could not calculate value counts:",
                e
            )


# ============================================================
# 10. WEATHER INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("WEATHER INFORMATION")
print("=" * 70)

for column in [
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level"
]:

    if column in df.columns:

        print(f"\n{column}:")

        try:
            print(
                df[column].describe()
            )

        except Exception as e:

            print(
                "Could not describe column:",
                e
            )


# ============================================================
# 11. ARRAY COLUMN CHECK
# ============================================================

print("\n" + "=" * 70)
print("ARRAY / LIST COLUMNS")
print("=" * 70)

for column in df.columns:

    try:

        sample_value = df[column].dropna().iloc[0]

        if isinstance(
            sample_value,
            (list, tuple, np.ndarray)
        ):

            print(
                f"{column:25} : ARRAY/LIST COLUMN"
            )

            print(
                "Example:",
                sample_value
            )

            try:
                print(
                    "Length:",
                    len(sample_value)
                )
            except Exception:
                pass

    except Exception:
        pass


# ============================================================
# 12. DUPLICATE ROW CHECK
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

# We cannot use df.duplicated() directly because some columns
# contain numpy arrays.

# Use only scalar/hashable columns for duplicate checking.

scalar_columns = []

for column in df.columns:

    try:

        sample_value = df[column].dropna().iloc[0]

        if isinstance(
            sample_value,
            (list, tuple, np.ndarray)
        ):
            continue

        scalar_columns.append(column)

    except Exception:
        continue


print("\nColumns used for duplicate checking:")
print(scalar_columns)


try:

    duplicate_count = df.duplicated(
        subset=scalar_columns
    ).sum()

    print(
        "\nDuplicates using scalar columns:",
        duplicate_count
    )

except Exception as e:

    print(
        "\nCould not calculate duplicates:",
        e
    )


# ============================================================
# 13. STORE + PRODUCT + DATE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("STORE + PRODUCT + DATE CHECK")
print("=" * 70)

key_columns = [
    "store_id",
    "product_id",
    "dt"
]

if all(
    column in df.columns
    for column in key_columns
):

    try:

        duplicate_keys = df.duplicated(
            subset=key_columns
        ).sum()

        print(
            "Duplicate store-product-date rows:",
            duplicate_keys
        )

    except Exception as e:

        print(
            "Could not check store-product-date duplicates:",
            e
        )


# ============================================================
# 14. PRODUCT RECORD DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PRODUCT RECORD DISTRIBUTION")
print("=" * 70)

if "product_id" in df.columns:

    product_counts = df["product_id"].value_counts()

    print(product_counts.describe())

    print("\nTop 10 products:")
    print(product_counts.head(10))


# ============================================================
# 15. STORE RECORD DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("STORE RECORD DISTRIBUTION")
print("=" * 70)

if "store_id" in df.columns:

    store_counts = df["store_id"].value_counts()

    print(store_counts.describe())

    print("\nTop stores:")
    print(store_counts.head(10))


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print(
    "Memory usage:",
    round(
        df.memory_usage(deep=True).sum() / (1024 ** 2),
        2
    ),
    "MB"
)

print("\nInspection completed successfully.")



gap_check = (
    df.groupby(["store_id", "product_id"])["dt"]
      .diff()
      .dt.days
)

print(gap_check.value_counts().sort_index().head(20))



print(
    df.groupby(["store_id", "product_id"])
      ["dt"]
      .nunique()
      .describe()
)