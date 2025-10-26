import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, LabelEncoder

df = pd.read_csv('./glass.csv')

df_num = df.select_dtypes('number')

#Chuan hoa max-min
MMScaler = MinMaxScaler()
df_mm = pd.DataFrame(MMScaler.fit_transform(df_num), columns=df_num.columns)
print(df_mm)


#Chuan hoa Standard
StdScaler = StandardScaler()
df_standard = pd.DataFrame(StdScaler.fit_transform(df_num), columns=df_num.columns)
print(df_standard)

#Chuan hoa Normalizer
Norm = Normalizer()
df_norm = pd.DataFrame(Norm.fit_transform(df_num), columns=df_num.columns)
print(df_norm)

#So hoa cot y
y = pd.DataFrame(LabelEncoder().fit_transform(df['Type']), columns=['Type'])
print(y)