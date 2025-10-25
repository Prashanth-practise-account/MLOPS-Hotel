import matplotlib.pyplot as plt
import seaborn as sns
import DataPreProcessing  # your module that returns cleaned dataframe

def eda(df):
    plt.figure()  # set figure size
    sns.countplot(x='is_canceled', data=df)
    plt.title("Booking Cancellations")
    plt.xlabel("Cancellation Status (0 = Not Canceled, 1 = Canceled)")
    plt.ylabel("Count")
    plt.savefig("booking_cancellation_plot.png")
    print("EDA plot saved as booking_cancellation_plot.png")

if __name__ == "__main__":
    df = DataPreProcessing.datapreprocessing()  # call your function
    eda(df)
