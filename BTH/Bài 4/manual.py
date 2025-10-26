import math

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, LabelEncoder

df = pd.read_csv('./glass.csv')

df_num = df.select_dtypes('number')


def min_max_scaler(X, min = 0, max = 1):
    return (X - X.min()) / (X.max() - X.min()) * (max - min) + min

df_mm = df_num.apply(lambda col: min_max_scaler(col))
print(df_mm)


#Chuan hoa Standard
def std_scaler(X):
    return (X - X.mean()) / X.std()
df_standard = df_num.apply(lambda col: std_scaler(col))
print(df_standard)

#Chuan hoa Normalizer
def normalizer(X):
    return X / math.sqrt(X.sum())
df_norm = df_num.apply(lambda col: normalizer(col))
print(df_norm)

#So hoa cot y
y = pd.DataFrame(LabelEncoder().fit_transform(df['Type']), columns=['Type'])
print(y)