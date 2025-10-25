from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import FeatureEngineering
import joblib
def modeltrain(df = FeatureEngineering.featureengineering()):
    feature = df[["lead_time", "total_nights", "arrival_month_num"]]
    target =  df["is_canceled"]
    X_train,X_test,y_train,y_test = train_test_split(
        feature,target,test_size=0.2,random_state=42
    )
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    predict = model.predict(X_test)
    acc = accuracy_score(y_test,predict)
    print(acc*100)
    joblib.dump(model, 'hotel_model.pkl')
    print('model saved successfully')
    return model
if __name__ == "__main__":
    modeltrain()