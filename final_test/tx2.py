import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft2, fft2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, kfold):
    predictions = []
    metrics = {'accuracy': [], 'precision': [], 'recall': []}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_fold_train, y_fold_train)
        
        y_pred = model_clone.predict(X_test)
        predictions.append(y_pred)
        
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
        metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
    
    return predictions, metrics


def print_fold_results(predictions, y_test, model_name):
    print(f"\n--- {model_name} ---")
    for fold, y_pred in enumerate(predictions, 1):
        correct = (y_pred == y_test.values).sum()
        incorrect = len(y_test) - correct
        accuracy = correct / len(y_test) * 100
        print(f"Fold {fold}: Đúng={correct}, Sai={incorrect}, Accuracy={accuracy:.2f}%")
    
    avg_correct = np.mean([(pred == y_test.values).sum() for pred in predictions])
    avg_incorrect = len(y_test) - avg_correct
    print(f"Trung bình: Đúng={avg_correct:.1f}, Sai={avg_incorrect:.1f}")


def print_metrics_results(metrics, model_name):
    print(f"\n--- {model_name} ---")
    for fold in range(len(metrics['accuracy'])):
        print(f"Fold {fold+1}: Accuracy={metrics['accuracy'][fold]:.4f}, "
              f"Precision={metrics['precision'][fold]:.4f}, "
              f"Recall={metrics['recall'][fold]:.4f}")
    
    print(f"\nTrung bình {model_name}:")
    print(f"  Accuracy:  {np.mean(metrics['accuracy']):.4f}")
    print(f"  Precision: {np.mean(metrics['precision']):.4f}")
    print(f"  Recall:    {np.mean(metrics['recall']):.4f}")


def plot_metrics_comparison(bayes_metrics, knn_metrics):
    folds = [f'Fold {i}' for i in range(1, 6)]
    x = np.arange(len(folds))
    width = 0.35
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_list = ['accuracy', 'precision', 'recall']
    titles = ['Accuracy theo từng Fold', 'Precision theo từng Fold', 'Recall theo từng Fold']
    
    for i, (metric, title) in enumerate(zip(metrics_list, titles)):
        axes[i].bar(x - width/2, bayes_metrics[metric], width, label='Naive Bayes', color='steelblue')
        axes[i].bar(x + width/2, knn_metrics[metric], width, label='KNN', color='darkorange')
        axes[i].set_xlabel('Fold')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(title)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(folds)
        axes[i].legend()
        axes[i].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=150)
    plt.show()
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    metrics_names = ['Accuracy', 'Precision', 'Recall']
    bayes_avg = [np.mean(bayes_metrics[m]) for m in metrics_list]
    knn_avg = [np.mean(knn_metrics[m]) for m in metrics_list]
    
    x2 = np.arange(len(metrics_names))
    ax2.bar(x2 - width/2, bayes_avg, width, label='Naive Bayes', color='steelblue')
    ax2.bar(x2 + width/2, knn_avg, width, label='KNN', color='darkorange')
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Score')
    ax2.set_title('So sánh trung bình Accuracy, Precision, Recall')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics_names)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('metrics_average_comparison.png', dpi=150)
    plt.show()


df = pd.read_csv('../data/Wine.csv')
print("Dữ liệu gốc:")
print(df.head())

columns = df.columns.tolist()
class_col = columns[-1]
feature_cols = columns[:-1]
first_two_cols = feature_cols[:2]
remaining_cols = feature_cols[2:]

# CÂU 1: Chuẩn hóa dữ liệu
print_header("CÂU 1: CHUẨN HÓA DỮ LIỆU")

df_minmax = pd.DataFrame(
    MinMaxScaler(feature_range=(0, 1)).fit_transform(df[first_two_cols]),
    columns=first_two_cols
)

df_standardized = pd.DataFrame(
    StandardScaler().fit_transform(df[remaining_cols]),
    columns=remaining_cols
)

label_encoder = LabelEncoder()
df_class = pd.DataFrame({class_col: label_encoder.fit_transform(df[class_col])})

df_final = pd.concat([df_minmax, df_standardized, df_class], axis=1)

print("\nDữ liệu sau khi chuẩn hóa:")
print(df_final.head())
print(f"\n2 cột đầu ({first_two_cols}): Min-Max Scaling [0,1]")
print(f"Các cột còn lại ({remaining_cols}): Standardize (Z-score)")
print(f"Cột class ({class_col}): Label Encoding - Classes: {list(label_encoder.classes_)}")

# CÂU 2: Biến đổi Fourier Transform 2D
print_header("CÂU 2: BIẾN ĐỔI FOURIER TRANSFORM 2D (scipy)")

data_2d = df_final[feature_cols].head(10).values

print("\n1. Dữ liệu 10 dòng đầu TRƯỚC KHI biến đổi:")
print(data_2d)

fft_2d_data = fft2(data_2d)
print("\n2. Dữ liệu SAU KHI biến đổi Fourier THUẬN (fft2):")
print(fft_2d_data)

ifft_2d_data = ifft2(fft_2d_data)
print("\n3. Dữ liệu SAU KHI biến đổi Fourier NGHỊCH (ifft2):")
print(ifft_2d_data.real)

# CÂU 3: Chia dữ liệu 95% train, 5% test
print_header("CÂU 3: CHIA DỮ LIỆU TRAIN/TEST (95%/5%)")

X = df_final[feature_cols]
y = df_final[class_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

print(f"\nTổng số mẫu: {len(df_final)}")
print(f"Số mẫu train: {len(X_train)} ({len(X_train)/len(df_final)*100:.1f}%)")
print(f"Số mẫu test: {len(X_test)} ({len(X_test)/len(df_final)*100:.1f}%)")
print(f"\nX_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# CÂU 4: Kiểm tra chéo 5-fold
print_header("CÂU 4: KIỂM TRA CHÉO 5-FOLD TRÊN TẬP TRAIN")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\nSố lượng mẫu train: {len(X_train)}, Số fold: 5\n")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"Fold {fold}: Train={len(train_idx)} mẫu, Validation={len(val_idx)} mẫu")

# CÂU 5 & 6: Phân lớp bằng Naive Bayes và KNN
print_header("CÂU 5: PHÂN LỚP BẰNG NAIVE BAYES VỚI 5-FOLD")

models = {
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = {}
for model_name, model in models.items():
    if model_name == 'KNN':
        print_header("CÂU 6: PHÂN LỚP BẰNG KNN VỚI 5-FOLD")
    
    predictions, metrics = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, kfold)
    results[model_name] = {'predictions': predictions, 'metrics': metrics}
    
    print_fold_results(predictions, y_test, model_name)

# CÂU 7: Accuracy, Precision, Recall + Đồ thị
print_header("CÂU 7: ACCURACY, PRECISION, RECALL CHO 5-FOLD")

for model_name in models:
    print_metrics_results(results[model_name]['metrics'], model_name)

plot_metrics_comparison(results['Naive Bayes']['metrics'], results['KNN']['metrics'])
