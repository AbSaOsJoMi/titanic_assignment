
import pandas as pd
import numpy as np

df = pd.read_csv("data/train.csv")

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Cabin'] = df['Cabin'].fillna("Unknown")

df['Sex'] = df['Sex'].str.lower()
df = df.drop_duplicates()

upper = df['Fare'].quantile(0.99)
df['Fare'] = np.where(df['Fare'] > upper, upper, df['Fare'])

df.to_csv("data/train_cleaned.csv", index=False)

print("Data cleaning complete")
