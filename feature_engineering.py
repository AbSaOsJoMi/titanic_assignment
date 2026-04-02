
import pandas as pd
import numpy as np

df = pd.read_csv("data/train_cleaned.csv")

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

df['Deck'] = df['Cabin'].str[0]

def age_group(age):
    if age < 13:
        return 'Child'
    elif age < 20:
        return 'Teen'
    elif age < 60:
        return 'Adult'
    else:
        return 'Senior'

df['AgeGroup'] = df['Age'].apply(age_group)

df['FarePerPerson'] = df['Fare'] / df['FamilySize']

df['Fare_log'] = np.log1p(df['Fare'])

df = pd.get_dummies(df, columns=['Sex','Embarked','Title','Deck'], drop_first=True)

df.to_csv("data/train_engineered.csv", index=False)

print("Feature engineering complete")
