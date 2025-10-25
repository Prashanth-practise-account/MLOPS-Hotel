import DataCollection
def datapreprocessing(df=DataCollection.datacollection()):
    number = df.select_dtypes(include=['number'])
    corr = number.corr()['is_canceled'].sort_values(ascending=False)
    df.fillna(value=0,inplace=True)
    df.drop_duplicates()
    return df
if __name__ == "__main__":
    datapreprocessing()
