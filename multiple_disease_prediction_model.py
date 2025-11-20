import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
n_samples = 5000

def generate_heart_data(n: int) -> pd.DataFrame:
    data = {
        'age': np.random.normal(58, 15, n).clip(25, 85).astype(int),
        'sex': np.random.choice([1, 0.7], n),
        'cp': np.random.choice([0, 1, 2, 3], n, p=[0.1, 0.2, 0.3, 0.4]),
        'trestbps': np.random.normal(135, 25, n).clip(90, 180).astype(int),
        'chol': np.random.normal(250, 50, n).clip(120, 400).astype(int),
        'exang': np.random.binomial(1, 0.4, n),
        'oldpeak': np.random.exponential(1.2, n).clip(0, 6),
        'ca': np.random.choice([0, 1, 2, 3], n, p=[0.4, 0.35, 0.2, 0.05]),
        'thal': np.random.choice([0, 1, 2, 3], n, p=[0.03, 0.17, 0.3, 0.5])
    }
    target = []
    for i in range(n):
        score = 0
        score += 8 if data['cp'][i] == 3 and data['thal'][i] == 2 else 4 if data['cp'][i] >= 2 else 0
        score += 4 + data['ca'][i] if data['ca'][i] > 0 else 0
        score += 6 if data['exang'][i] == 1 and data['oldpeak'][i] > 2 else 3 if data['exang'][i] == 1 else 0
        score += (data['age'][i] > 65) * 2 + (data['sex'][i] == 1) * 1.5
        score += (data['trestbps'][i] > 150) * 2 + (data['chol'][i] > 280) * 2
        score += (np.random.random() < min(0.95, max(0.02, score / 12)))
        target.append(int(score >= 8))
    data['target'] = target
    return pd.DataFrame(data)

def generate_diabetes_data(n: int) -> pd.DataFrame:
    data = {
        'Pregnancies': np.random.poisson(2.8, n).clip(0, 12),
        'Glucose': np.random.normal(130, 40, n).clip(60, 200).astype(int),
        'BMI': np.random.normal(33, 8, n).clip(15, 55),
        'Age': np.random.normal(38, 15, n).clip(18, 80).astype(int),
        'DiabetesPedigreeFunction': np.random.gamma(1.8, 0.4, n).clip(0.05, 2.5)
    }
    target = []
    for i in range(n):
        score = 0
        score += (data['Glucose'][i] >= 140) * 8 + (data['Glucose'][i] >= 126) * 5
        score += (data['BMI'][i] >= 35) * 6 + (data['BMI'][i] >= 30) * 3
        score += (data['Age'][i] >= 50 and data['DiabetesPedigreeFunction'][i] > 0.8) * 5
        target.append(1 if score >= 10 else 0 if score <= 1 else int(np.random.binomial(1, score / 12)))
    data['Outcome'] = target
    return pd.DataFrame(data)

def generate_parkinsons_data(n: int) -> pd.DataFrame:
    target = np.random.binomial(1, 0.5, n)
    data = {
        'MDVP_Jitter_percent': [],
        'MDVP_Shimmer': [],
        'NHR': [],
        'HNR': [],
        'RPDE': [],
        'DFA': [],
        'PPE': []
    }
    for i in range(n):
        if target[i]:
            data['MDVP_Jitter_percent'].append(np.random.uniform(0.015, 0.035))
            data['MDVP_Shimmer'].append(np.random.uniform(0.04, 0.08))
            data['NHR'].append(np.random.uniform(0.03, 0.15))
            data['HNR'].append(np.random.uniform(8, 18))
            data['RPDE'].append(np.random.uniform(0.55, 0.7))
            data['DFA'].append(np.random.uniform(0.72, 0.85))
            data['PPE'].append(np.random.uniform(0.25, 0.55))
        else:
            data['MDVP_Jitter_percent'].append(np.random.uniform(0.002, 0.008))
            data['MDVP_Shimmer'].append(np.random.uniform(0.01, 0.025))
            data['NHR'].append(np.random.uniform(0.001, 0.015))
            data['HNR'].append(np.random.uniform(22, 35))
            data['RPDE'].append(np.random.uniform(0.25, 0.45))
            data['DFA'].append(np.random.uniform(0.57, 0.68))
            data['PPE'].append(np.random.uniform(0.04, 0.18))
    data['status'] = target
    return pd.DataFrame(data)

def generate_ckd_data(n: int) -> pd.DataFrame:
    target = np.random.binomial(1, 0.4, n)
    data = {
        'sc': [], 'bu': [], 'hemo': [], 'pcv': [], 'al': [],
        'sg': [], 'htn': [], 'dm': []
    }
    for i in range(n):
        if target[i]:
            data['sc'].append(np.random.uniform(2, 8))
            data['bu'].append(np.random.uniform(30, 150))
            data['hemo'].append(np.random.uniform(6, 10))
            data['pcv'].append(np.random.randint(15, 30))
            data['al'].append(np.random.randint(2, 5))
            data['sg'].append(np.random.choice([1.005, 1.010]))
            data['htn'].append(np.random.binomial(1, 0.7))
            data['dm'].append(np.random.binomial(1, 0.5))
        else:
            data['sc'].append(np.random.uniform(0.4, 1.2))
            data['bu'].append(np.random.uniform(10, 25))
            data['hemo'].append(np.random.uniform(12, 17))
            data['pcv'].append(np.random.randint(35, 50))
            data['al'].append(0)
            data['sg'].append(np.random.choice([1.015, 1.020, 1.025]))
            data['htn'].append(np.random.binomial(1, 0.2))
            data['dm'].append(np.random.binomial(1, 0.15))
    data['classification'] = target
    return pd.DataFrame(data)

# Dispatcher
data_generators = {
    'heart': generate_heart_data,
    'diabetes': generate_diabetes_data,
    'parkinsons': generate_parkinsons_data,
    'ckd': generate_ckd_data
}

# Generate CSVs
for disease, generator in data_generators.items():
    df = generator(n_samples)
    df.to_csv(f'{disease}.csv', index=False)

def train_model(csv_file: str, target_col: str):
    df = pd.read_csv(csv_file)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42,
        class_weight='balanced', max_features='sqrt')
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"{csv_file} Accuracy: {acc:.2%}")
    return {'model': model, 'scaler': scaler}, acc
