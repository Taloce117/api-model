import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def ingest_data(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # suppresion des lignes avec des valeurs manquantes
    df = df.dropna(inplace=True)
    print(df)
    # remplacer les valeurs non numériques par des valeurs numériques
    df = df.replace('female', 0).replace('male', 1)
    return df

def train_model(df: pd.DataFrame) -> ClassifierMixin:
    model = KNeighborsClassifier(4)
    y = df['survived']
    x = df.drop['survived']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # train model
    model.fit(x,y)
    # evaluate model
    score = model.score(X_test, y_test)
    print(f"score: {score}")
    return model

if __name__ == "__main__":
    df = ingest_data("titanic.xls")
    df = clean_data(df)
    model = train_model(df)
    joblib.dump(model, "model_titanic.joblib")