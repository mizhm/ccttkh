# 📘 Data Science Cheatsheet: Pandas, NumPy, Scikit-learn

Tài liệu tổng hợp các câu lệnh cốt lõi thường dùng trong quy trình Phân tích dữ liệu và Học máy với Python.

---

## 1. Pandas 🐼

**Mục đích:** Xử lý và phân tích dữ liệu dạng bảng (DataFrame).

### 🛠️ Thiết Lập & Đọc Dữ Liệu

| Hành động     | Cú pháp                             | Ghi chú                  |
|:--------------|:------------------------------------|:-------------------------|
| **Import**    | `import pandas as pd`               |                          |
| **Đọc CSV**   | `df = pd.read_csv('file.csv')`      |                          |
| **Đọc Excel** | `df = pd.read_excel('file.xlsx')`   | Cần thư viện `openpyxl`. |
| **Ghi CSV**   | `df.to_csv('out.csv', index=False)` | Lưu file, bỏ cột index.  |

### 🔍 Khám Phá Dữ Liệu

| Hành động        | Cú pháp                     | Ghi chú                             |
|:-----------------|:----------------------------|:------------------------------------|
| **Xem đầu/đuôi** | `df.head(n)` / `df.tail(n)` | Mặc định n=5.                       |
| **Cấu trúc**     | `df.info()`                 | Kiểu dữ liệu, bộ nhớ, giá trị null. |
| **Thống kê**     | `df.describe()`             | Mean, std, min, max (cột số).       |
| **Kích thước**   | `df.shape`                  | Trả về (hàng, cột).                 |
| **Tên cột**      | `df.columns`                | Danh sách tên các cột.              |

### 🎯 Chọn Lọc (Indexing & Selection)

| Hành động          | Cú pháp                     | Ghi chú                     |
|:-------------------|:----------------------------|:----------------------------|
| **Chọn cột**       | `df['col_name']`            | Trả về Series.              |
| **Chọn nhiều cột** | `df[['col1', 'col2']]`      | Trả về DataFrame.           |
| **Lọc điều kiện**  | `df[df['age'] > 20]`        | Lọc hàng theo Boolean mask. |
| **Theo Label**     | `df.loc[row_lbl, col_lbl]`  | Chọn theo tên nhãn.         |
| **Theo Vị trí**    | `df.iloc[row_idx, col_idx]` | Chọn theo chỉ số (index).   |

### 🧹 Làm Sạch & Biến Đổi

| Hành động       | Cú pháp                              | Ghi chú                        |
|:----------------|:-------------------------------------|:-------------------------------|
| **Check Null**  | `df.isnull().sum()`                  | Đếm số lượng NaN mỗi cột.      |
| **Xóa Null**    | `df.dropna()`                        | Xóa hàng có NaN.               |
| **Điền Null**   | `df.fillna(value)`                   | Điền NaN bằng giá trị cụ thể.  |
| **Sắp xếp**     | `df.sort_values(by='col')`           | `ascending=False` để giảm dần. |
| **Đổi tên cột** | `df.rename(columns={'old': 'new'})`  |                                |
| **Apply hàm**   | `df['col'].apply(lambda x: x*2)`     | Áp dụng hàm cho từng phần tử.  |
| **Groupby**     | `df.groupby('col')['target'].mean()` | Gom nhóm và tính toán.         |

---

## 2. NumPy 🔢

**Mục đích:** Tính toán khoa học, xử lý mảng đa chiều (Matrix/Vector).

### 🧱 Khởi Tạo Mảng

| Hành động    | Cú pháp                        | Ghi chú                     |
|:-------------|:-------------------------------|:----------------------------|
| **Import**   | `import numpy as np`           |                             |
| **Từ List**  | `np.array([1, 2, 3])`          |                             |
| **Mảng 0**   | `np.zeros((2, 3))`             | Mảng 2x3 toàn số 0.         |
| **Mảng 1**   | `np.ones((2, 3))`              | Mảng 2x3 toàn số 1.         |
| **Tuần tự**  | `np.arange(start, stop, step)` | Giống range() của Python.   |
| **Chia đều** | `np.linspace(0, 1, 5)`         | 5 điểm cách đều từ 0 đến 1. |

### 📐 Thuộc Tính & Biến Đổi

| Hành động      | Cú pháp                  | Ghi chú                     |
|:---------------|:-------------------------|:----------------------------|
| **Kích thước** | `arr.shape`              | (hàng, cột).                |
| **Số chiều**   | `arr.ndim`               | 1, 2, 3...                  |
| **Reshape**    | `arr.reshape(3, 2)`      | Đổi cấu trúc mảng.          |
| **Transpose**  | `arr.T`                  | Chuyển vị (hàng thành cột). |
| **Nối mảng**   | `np.concatenate((a, b))` | Nối a và b.                 |

### ➕ Phép Toán (Vectorization)

| Hành động        | Cú pháp                     | Ghi chú                      |
|:-----------------|:----------------------------|:-----------------------------|
| **Cơ bản**       | `arr + 5`, `arr * 2`        | Thực hiện trên từng phần tử. |
| **Nhân Ma trận** | `np.dot(a, b)` hoặc `a @ b` | Tích vô hướng.               |
| **Thống kê**     | `arr.mean()`, `arr.sum()`   | Trung bình, Tổng.            |
| **Max/Min**      | `arr.max()`, `arr.min()`    | Giá trị lớn nhất/nhỏ nhất.   |
| **Theo trục**    | `arr.sum(axis=0)`           | 0=cột, 1=hàng.               |

---

## 3. Scikit-learn (Sklearn) 🤖

**Mục đích:** Xây dựng, huấn luyện và đánh giá mô hình Machine Learning.

### ✂️ Chia & Xử Lý Dữ Liệu

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Chia tập Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Chuẩn hóa dữ liệu (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit & Transform train
X_test_scaled = scaler.transform(X_test)  # Chỉ Transform test
```

### 🧠 Quy Trình Huấn Luyện (Ví dụ: Logistic Regression)

```python

from sklearn.linear_model import LogisticRegression

# 1. Khởi tạo mô hình
model = LogisticRegression()

# 2. Huấn luyện (Fit)
model.fit(X_train_scaled, y_train)

# 3. Dự đoán (Predict)
y_pred = model.predict(X_test_scaled)
```

### 📊 Đánh Giá Mô Hình

```python

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Độ chính xác
acc = accuracy_score(y_test, y_pred)

# Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)

# Báo cáo chi tiết (Precision, Recall, F1)
print(classification_report(y_test, y_pred))
```
### ⚙️ Tinh Chỉnh (Hyperparameter Tuning)

```python

from sklearn.model_selection import GridSearchCV

params = {'C': [0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), params, cv=5)
grid.fit(X_train_scaled, y_train)

print(grid.best_params_)
```

# 📝 Tài Liệu Ôn Tập: Xử Lý Dữ Liệu & K-Fold Cross-Validation

Tài liệu này tổng hợp code mẫu giải quyết 5 yêu cầu cụ thể trong đề cương ôn tập của bạn.

---

## 4. Chuẩn Hóa và Số Hóa Dữ Liệu

### A. Số hóa dữ liệu (Encoding)
Chuyển dữ liệu từ dạng chữ (Categorical) sang dạng số để máy học được.

**Trường hợp 1: Label Encoder** (Dùng cho cột nhãn mục tiêu `y` hoặc biến thứ bậc)
```python
from sklearn.preprocessing import LabelEncoder

# Giả sử y là: ['Male', 'Female', 'Male']
le = LabelEncoder()
y_encoded = le.fit_transform(y) 
# Kết quả: [1, 0, 1]
```
**Trường hợp 2: One-Hot Encoder** (Dùng cho cột đặc trưng `X`)
```python
from sklearn.preprocessing import OneHotEncoder

# Giả sử X là: [['Small'], ['Medium'], ['Large']]
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(X).toarray()
# Kết quả: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
```
**Trường hợp 3: Ordinal Encoder** (Dùng cho cột đặc trưng `X`)
```python
from sklearn.preprocessing import OrdinalEncoder

# Giả sử X là: [['Low'], ['Medium'], ['High']]
oe = OrdinalEncoder()
X_encoded = oe.fit_transform(X)
# Kết quả: [[0], [1], [2]]
```
### B. Chuẩn hóa dữ liệu (Scaling)

Đưa dữ liệu về cùng một miền giá trị (thường dùng trước SVM, Logistic Regression).
```python

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Cách 1: StandardScaler (Về phân phối chuẩn: mean=0, std=1) - Khuyên dùng cho SVM/Logistic
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cách 2: MinMaxScaler (Về khoảng [0, 1])
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)

from sklearn.preprocessing import Normalizer

# Cách 1: Normalizer (Về khoảng [0, 1])
normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X)
```
### C. Biến Đổi Cosin (Discrete Cosine Transform - DCT)

Trong xử lý dữ liệu (đặc biệt là nén dữ liệu hoặc trích xuất đặc trưng), biến đổi Cosin thường được dùng để giảm chiều dữ liệu hoặc làm mịn.
```python

from scipy.fftpack import dct
import numpy as np

# Hàm thực hiện DCT trên từng hàng của dữ liệu X
# axis=1: thực hiện theo hàng
# type=2: loại DCT phổ biến nhất
# norm='ortho': chuẩn hóa trực giao
X_dct = dct(X, axis=1, type=2, norm='ortho')

# Nếu muốn lấy n thành phần đầu tiên (giảm chiều)
n_components = 5
X_dct_reduced = X_dct[:, :n_components]
```


## 5. Ứng dụng

### A. Trích xuất đặc trưng

#### 3. Chia Dữ Liệu Train/Test Theo %

Chia dữ liệu thành 2 phần cố định (Ví dụ: 70% học, 30% thi).

```python
from sklearn.model_selection import train_test_split

# test_size=0.3: dành 30% cho tập test
# random_state=42: giữ cố định cách chia để kết quả không đổi mỗi lần chạy
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)
```

#### 4 & 5. Quy Trình K-Fold Cross-Validation (Trọng Tâm)

Đây là phần quan trọng nhất: Chia dữ liệu thành K phần, lần lượt dùng 1 phần để test và K-1 phần để train, sau đó tính trung bình các chỉ số đánh giá.

##### Các thư viện cần thiết

```python
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
```

##### Code mẫu thực hiện vòng lặp K-Fold

Đoạn code này áp dụng cho cả SVM và Logistic Regression.

```python
# 1. Khởi tạo K-Fold
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# 2. Chọn mô hình (Bỏ comment mô hình bạn muốn dùng)
model = SVC(kernel='linear')               # SVM
# model = LogisticRegression(max_iter=1000)  # Logistic Regression

# 3. Tạo danh sách lưu kết quả
acc_scores = []
pre_scores = []
rec_scores = []

print(f"Bắt đầu chạy {k}-Fold Cross-Validation...")

# 4. Vòng lặp K-Fold
# Lưu ý: X và y phải là dạng numpy array. Nếu là DataFrame, dùng X.values, y.values
X_arr = np.array(X)
y_arr = np.array(y)

for fold_idx, (train_index, test_index) in enumerate(kf.split(X_arr)):
    # A. Lấy dữ liệu theo index của fold hiện tại
    X_train_fold, X_test_fold = X_arr[train_index], X_arr[test_index]
    y_train_fold, y_test_fold = y_arr[train_index], y_arr[test_index]

    # B. Huấn luyện mô hình
    model.fit(X_train_fold, y_train_fold)

    # C. Dự đoán
    y_pred_fold = model.predict(X_test_fold)

    # D. Tính các chỉ số
    # average='macro' hoặc 'weighted' nếu bài toán phân loại nhiều lớp (multi-class)
    # average='binary' nếu chỉ có 2 lớp (0 và 1)
    acc = accuracy_score(y_test_fold, y_pred_fold)
    pre = precision_score(y_test_fold, y_pred_fold, average='macro', zero_division=0)
    rec = recall_score(y_test_fold, y_pred_fold, average='macro', zero_division=0)

    # Lưu vào danh sách
    acc_scores.append(acc)
    pre_scores.append(pre)
    rec_scores.append(rec)

    print(f"Fold {fold_idx+1}: Accuracy={acc:.4f}, Precision={pre:.4f}, Recall={rec:.4f}")

# 5. Tính trung bình kết quả cuối cùng
print("-" * 30)
print(f"KẾT QUẢ TRUNG BÌNH ({k} folds):")
print(f"Accuracy : {np.mean(acc_scores):.4f}")
print(f"Precision: {np.mean(pre_scores):.4f}")
print(f"Recall   : {np.mean(rec_scores):.4f}")
```

##### Giải thích các tham số quan trọng trong metrics:

- `average='binary'`: Dùng cho bài toán 2 lớp (VD: Đúng/Sai).
- `average='macro'`: Dùng cho bài toán nhiều lớp (VD: Hoa A, Hoa B, Hoa C), tính trung bình các lớp không trọng số.
- `zero_division=0`: Tránh lỗi chia cho 0 nếu mô hình không dự đoán được lớp nào đó.