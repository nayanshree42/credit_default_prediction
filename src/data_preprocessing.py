import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path=r'C:\Users\HP\Desktop\python\credit-default-prediction\data\default of credit card clients.xls'):
    # read excel file
    df = pd.read_excel(path, header=1)
    # some datasets have extra headers, so header=1 skips the first row
    return df 

def preprocess_data(df):
    # Assume 'default.payment.next.month' is target column
    X = df.drop('default payment next month', axis=1)
    y = df['default payment next month']

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler