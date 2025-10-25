def datacollection():
    import pandas as pd
    df = pd.read_csv("hotel_bookings.csv")
    return df