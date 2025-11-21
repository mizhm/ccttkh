import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Normalizer

data = {
    'Gioi_tinh': ['Nam', 'Nữ', 'Nữ', 'Nam', 'Nữ'],
    'Thanh_pho': ['Hà Nội', 'HCM', 'Đà Nẵng', 'HCM', 'Hà Nội'],
    'Tuoi': [25, 30, 22, 35, 28],
    'Thu_nhap': [15000000, 20000000, 12000000, 30000000, 18000000]
}

df = pd.DataFrame(data)

print("--- Dữ liệu gốc ---")
print(df)
print("\n")

le = LabelEncoder()
df['Gioi_tinh_encoded'] = le.fit_transform(df['Gioi_tinh'])

# 1 hot encoding (dl khong co thu tu)
df_encoded = pd.get_dummies(df, columns=['Thanh_pho'])
print(df_encoded.columns)

scaler_minmax = MinMaxScaler()
df_encoded['Thu_nhap_minmax'] = scaler_minmax.fit_transform(df_encoded[['Thu_nhap']])

scaler_standard = StandardScaler()
df_encoded['Tuoi_standard'] = scaler_standard.fit_transform(df_encoded[['Tuoi']])

scaler_norm = Normalizer()
df_encoded['Tuoi_standard'] = scaler_norm.fit_transform(df_encoded[['Tuoi']])

# Bước A: Xác định danh sách cột
# cols_chu = ['Mau_sac', 'Kich_thuoc', 'Chat_lieu']
# cols_so  = ['Gia', 'Can_nang', 'Chieu_cao']

# Bước B: Số hóa (Dùng vòng lặp cho LabelEncoder)
# le = LabelEncoder()
# for col in cols_chu:
#     df_c1[col] = le.fit_transform(df_c1[col])

# Bước C: Chuẩn hóa (Truyền thẳng danh sách cols_so vào StandardScaler)
# scaler = StandardScaler()
# df_c1[cols_so] = scaler.fit_transform(df_c1[cols_so])
