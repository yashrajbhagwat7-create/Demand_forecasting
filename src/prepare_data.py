import pandas as pd
from datasets import load_dataset


#==================LOAD DATA===============


def load_data():
    
    '''Load the dataset and convert it into a panda Datafram'''
    print("Loading Dataset...")
    
    datasets = load_dataset(
        "Dingdong-Inc/FreshRetailNet-50K",
        split="train[:50000]"
    )

    df = datasets.to_pandas()
    print("dataset loaded successfully")
    
    print("shape",df.shape)
    
    return df
    
    
#==============CLEAD DATA===================

def clean_data(df):
     '''perform basic cleaning and oragnization'''
     print("\nCleaning data....")
     
     df["dt"] = pd.to_datetime(
         df["dt"],
         errors="coerce"
         )
         
     df = df.sort_values(
         ["store_id","product_id","dt"]
      ).reset_index(drop=True)         

     selected_columns = [
        "city_id",
        "store_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
        "dt",
        "sale_amount",
        "hours_sale",
        "stock_hour6_22_cnt",
        "hours_stock_status",
        "discount",
        "holiday_flag",
        "activity_flag",
        "precpt",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
    ]

     df = df[selected_columns]

     print("Cleaning completed.")

     return df
    
    
    
    
def validate_data(df):
    """Check the prepared dataset for common data-quality problems."""

    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    # Missing values
    total_missing = df.isnull().sum().sum()

    print(f"\nTotal missing values: {total_missing}")

    if total_missing > 0:
        print("\nColumns containing missing values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        print("No missing values found.")

    # Invalid dates
    invalid_dates = df["dt"].isna().sum()

    print(f"\nInvalid dates: {invalid_dates}")

    # Duplicate store-product-date combinations
    duplicates = df.duplicated(
        subset=[
            "store_id",
            "product_id",
            "dt"
        ]
    ).sum()

    print(
        f"Duplicate store-product-date rows: {duplicates}"
    )

    # Negative sales
    negative_sales = (
        df["sale_amount"] < 0
    ).sum()

    print(f"Negative sales rows: {negative_sales}")

    # Date range
    print(
        f"\nDate range: "
        f"{df['dt'].min().date()} "
        f"to "
        f"{df['dt'].max().date()}"
    )

    # Stockouts
    stockout_rows = (
        df["stock_hour6_22_cnt"] > 0
    ).sum()

    stockout_percentage = (
        stockout_rows / len(df)
    ) * 100

    print(
        f"\nRows with stockout: {stockout_rows}"
    )

    print(
        f"Stockout percentage: "
        f"{stockout_percentage:.2f}%"
    )


# ============================================================
# 4. SAVE DATA
# ============================================================

def save_data(df):
    """Save the prepared dataset as a CSV file."""

    output_path = (
        "H:/demand_forecasting/data/"
        "prepared_sales.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nPrepared dataset saved to:\n"
        f"{output_path}"
    )


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("PREPARING DEMAND FORECASTING DATA")
    print("=" * 60)

    # Step 1: Load
    df = load_data()

    # Step 2: Clean
    df = clean_data(df)

    # Step 3: Validate
    validate_data(df)

    # Step 4: Save
    save_data(df)

    print("\n" + "=" * 60)
    print("PREPARATION COMPLETED")
    print("=" * 60)


# ============================================================
# 6. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()    