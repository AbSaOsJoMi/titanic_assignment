
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/train_engineered.csv")

drop_cols = ['Name','Ticket','Cabin','AgeGroup']
df_model = df.drop(columns=drop_cols, errors='ignore')

X = df_model.drop('Survived', axis=1)
y = df_model['Survived']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

importance = pd.Series(model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False).head(10))
