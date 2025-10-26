import math
import pandas as pd


def calculate_priors(y_train):
    """
    Tính xác suất không điều kiện P(Y) (prior probabilities).
    """
    n_samples_total = len(y_train)
    classes = set(y_train)
    class_counts = {c: 0 for c in classes}

    for label in y_train:
        class_counts[label] += 1

    priors = {}
    for c, count in class_counts.items():
        priors[c] = count / n_samples_total

    return priors, class_counts, classes


def calculate_conditionals(X_train, y_train, classes, class_counts, k=1):
    """
    Tính xác suất có điều kiện P(X|Y) với làm mịn Laplace (k=1).
    """
    n_features = len(X_train[0])
    conditionals = {}
    feature_values = {}

    for i in range(n_features):
        conditionals[i] = {}
        feature_values[i] = set(row[i] for row in X_train)

        for val in feature_values[i]:
            conditionals[i][val] = {}
            for c in classes:
                conditionals[i][val][c] = 0

    for i in range(len(y_train)):
        label = y_train[i]
        features = X_train[i]
        for j in range(n_features):
            feature_val = features[j]
            conditionals[j][feature_val][label] += 1

    for i in range(n_features):
        num_unique_vals = len(feature_values[i])

        for val in feature_values[i]:
            for c in classes:
                # P(X|Y) = (Đếm(X, Y) + k) / (Đếm(Y) + k * V)
                numerator = conditionals[i][val][c] + k
                denominator = class_counts[c] + k * num_unique_vals
                conditionals[i][val][c] = numerator / denominator

    return conditionals, feature_values


def predict_naive_bayes(X_test, priors, conditionals, classes, feature_values, class_counts, k=1):
    """
    Dự đoán nhãn cho X_test dựa trên các xác suất đã tính.
    """
    predictions = []

    for x_new in X_test:
        log_posteriors = {}

        for c in classes:
            log_posteriors[c] = math.log(priors[c])

            n_features = len(x_new)
            for i in range(n_features):
                feature_val = x_new[i]

                if feature_val in conditionals[i]:
                    prob = conditionals[i][feature_val][c]
                else:
                    num_unique_vals = len(feature_values[i])
                    prob = k / (class_counts[c] + k * num_unique_vals)

                log_posteriors[c] += math.log(prob)

        best_class = max(log_posteriors, key=log_posteriors.get)
        predictions.append(best_class)

    return predictions


file_path = 'data1.csv'
df = pd.read_csv(file_path)
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
df_clean = df.drop(columns=['ID'])

feature_columns = ['Age', 'Income', 'Student', 'Credit']
target_column = 'Buy'
X_train = df_clean[feature_columns].values.tolist()
y_train = df_clean[target_column].values.tolist()

k_laplace = 1
priors, class_counts, classes = calculate_priors(y_train)
conditionals, feature_values = calculate_conditionals(X_train, y_train, classes, class_counts, k_laplace)

print(f"Xác suất không điều kiện P(Buy): {priors}")
print(f"Xác suất có điều kiện P(Age='Young' | Buy='yes'): {conditionals[0]['Young']['yes']:.4f}")

X_test = [
    ['Young', 'Medium', 'Yes', 'Fair']
]
predictions = predict_naive_bayes(
    X_test,
    priors,
    conditionals,
    classes,
    feature_values,
    class_counts,
    k_laplace
)
print(f"Dự đoán cho {X_test[0]}: {predictions[0]}")
