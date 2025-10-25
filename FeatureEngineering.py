import pandas as pd
import DataPreProcessing

def featureengineering(df=None):
    if df is None:
        df = DataPreProcessing.datapreprocessing()
    df['total_nights'] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["arrival_month_num"] = pd.to_datetime(df["arrival_date_month"], format='%B').dt.month
    df['total_people'] = df['adults'] + df['children'] + df['babies']

    return df
