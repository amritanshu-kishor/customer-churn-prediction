import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/cleaned_churn.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)

with open("model/churn_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

print("Model trained and saved to model/churn_model.pkl")
