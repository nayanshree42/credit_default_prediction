from sklearn.ensemble import RandomForestClassifier
import joblib
from data_preprocessing import load_data, preprocess_data

def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # save model + scaler
    joblib.dump(model, 'src/model.pkl')
    joblib.dump(scaler, 'src/scaler.pkl')

    return model, X_test, y_test

if __name__ == '__main__':
    train_model()