import pandas as pd 

#=========================
#========configuration====



GROUP_COLUMNS= ["store_id","product_id"]

LAG_WINDOWS=[1,7,14,28]

ROLLING_WINDOWS=[7,14,28]

EXTERNAL_FEATURES=[
         "discount",
         "holiday_flag",
         "activity_flag",
         "precpt",
         "avg_temperature",
         "avg_humidity",
         "avg_wind_level",
         "stock_hour6_22_cnt",
         "hours_sale",
         "hours_stock_status"]
         
         
         
#-----------------------------------------
#calendar features
#-----------------------------------------

def create_calendat_features(df:pd.DataFrame) -> pd.DataFrame:
    
    """
    create calendar based features from  the date colume.
    """
    
    df=df.copy()
    
    df["dt"]= pd.to_datetime(df["dt"])
    
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["day_of_month"] = df["dt"].dt.day
    
    
    df["week_of_year"] = (
    df["dt"]
    .dt.isocalendar()
    .week
    .astype(int)
    )
    
    
    df["month"] = df["dt"].dt.month
    
    df["is_weekend"] = (
        df["day_of_week"] >=5
        ).astype(int)
    print("feature created successfully")    
        
        
    return df


if __name__ == "__main__":

    from prepare_data import load_data
    df = load_data()

     
    create_calendat_features(df)    