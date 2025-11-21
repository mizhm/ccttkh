import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data = load_wine()
X = data.data
y = data.target
print(f"Dữ liệu gốc: {X.shape} mẫu, {len(np.unique(y))} lớp.\n")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Đã thực hiện chuẩn hóa (StandardScaler).")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"Đã chia dữ liệu theo tỷ lệ 80-20:")
print(f"- Train set: {X_train.shape}")
print(f"- Test set:  {X_test.shape}\n")

k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
print("Cấu hình K-Fold với K=5.\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='linear')
}

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted')
}

print("--- KẾT QUẢ ĐÁNH GIÁ K-FOLD (Trung bình 5 lần lặp) ---")
results = []

for name, model in models.items():
    scores = cross_validate(model, X_scaled, y, cv=k_fold, scoring=scoring)

    mean_acc = scores['test_accuracy'].mean()
    mean_prec = scores['test_precision'].mean()
    mean_rec = scores['test_recall'].mean()

    results.append({
        "Thuật toán": name,
        "Accuracy": mean_acc,
        "Precision": mean_prec,
        "Recall": mean_rec
    })

df_results = pd.DataFrame(results)
print(df_results.to_markdown(index=False))
