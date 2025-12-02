"""
A complete ML training pipeline for heart disease prediction.
It will:
 - look for 'heart.csv' in the same folder. If not found, it will generate synthetic demo data.
 - run preprocessing, train multiple models, evaluate them, save the best model and scaler/encoders.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

RANDOM_SEED = 42

# -------------------------
# 1. Load data or generate demo data
# -------------------------
DATA_FILE = "heart.csv"

def generate_synthetic_heart_data(n=1000, random_state=RANDOM_SEED):
    """
    Creates a simple synthetic dataset that mimics typical heart disease attributes.
    Columns match common UCI Heart dataset fields.
    """
    rng = np.random.RandomState(random_state)
    age = rng.randint(29, 77, n)
    sex = rng.binomial(1, 0.7, n)  # 1 male, 0 female (example distribution)
    cp = rng.randint(0, 4, n)  # chest pain 0-3
    trestbps = rng.normal(130, 20, n).astype(int)
    chol = rng.normal(245, 50, n).astype(int)
    fbs = rng.binomial(1, 0.15, n)
    restecg = rng.randint(0, 2, n)
    thalach = (200 - age + rng.normal(0, 10, n)).astype(int)
    exang = rng.binomial(1, 0.3, n)
    oldpeak = np.round(np.abs(rng.normal(1.0 + exang*1.2, 1.0, n)), 2)
    slope = rng.randint(0, 3, n)
    ca = rng.randint(0, 4, n)
    thal = rng.randint(1, 4, n)  # 1-3 typical
    # create target with a simple rule + noise
    risk_score = (age/50) + (cp*0.5) + (trestbps/140) + (chol/250) + exang + (oldpeak*0.6) + (ca*0.5)
    prob = 1 / (1 + np.exp(- (risk_score - 3)))  # sigmoid-ish
    target = rng.binomial(1, prob)
    df = pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal, 'target': target
    })
    return df

if os.path.exists(DATA_FILE):
    print(f"Loading dataset from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
else:
    print(f"{DATA_FILE} not found — generating synthetic demo dataset.")
    df = generate_synthetic_heart_data(1000)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# -------------------------
# 2. Preprocessing
# -------------------------
# Common column names used in many heart disease datasets:
# age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target

# Basic check
print("\nMissing values per column:\n", df.isna().sum())

# Separate features and target
target_col = 'condition'
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify numeric and categorical features
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
# treat some integer-coded variables as categorical if appropriate
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numeric_cols = [c for c in numeric_cols if c not in categorical_cols]

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

from sklearn.preprocessing import OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# -------------------------
# 3. Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

# -------------------------
# 4. Models to compare
# -------------------------
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_SEED),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(probability=True, random_state=RANDOM_SEED)
}

results = []

# Helper: evaluate
def evaluate_model(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    rocauc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=rocauc, y_pred=y_pred, y_proba=y_proba)

# Train and evaluate each model
for name, clf in models.items():
    print(f"\nTraining {name} ...")
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', clf)])
    pipe.fit(X_train, y_train)
    # cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
    evals = evaluate_model(pipe, X_test, y_test)
    evals['cv_mean_acc'] = np.mean(cv_scores)
    evals['cv_std_acc'] = np.std(cv_scores)
    evals['model_name'] = name
    results.append(evals)
    print(f"{name} CV accuracy: {evals['cv_mean_acc']:.4f} ± {evals['cv_std_acc']:.4f}")
    print(f"Test accuracy: {evals['accuracy']:.4f}, Precision: {evals['precision']:.4f}, Recall: {evals['recall']:.4f}, F1: {evals['f1']:.4f}, ROC-AUC: {evals['roc_auc']}")

# Summarize results
df_results = pd.DataFrame([{
    'model': r['model_name'],
    'cv_acc': r['cv_mean_acc'],
    'test_acc': r['accuracy'],
    'precision': r['precision'],
    'recall': r['recall'],
    'f1': r['f1'],
    'roc_auc': r['roc_auc']
} for r in results]).sort_values('test_acc', ascending=False)

print("\nModel comparison:\n", df_results)

# -------------------------
# 5. Pick best model (by F1 or test_acc) and save
# -------------------------
best_idx = df_results['f1'].idxmax()
best_model_name = df_results.loc[best_idx, 'model']
print("\nBest model (by F1):", best_model_name)

best_clf = models[best_model_name]
best_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', best_clf)])
best_pipe.fit(X_train, y_train)

# Save artifacts
joblib.dump(best_pipe, 'best_model.joblib')
print("Saved best model to best_model.joblib")

# Also save an example of feature importances if available
if hasattr(best_clf, 'feature_importances_'):
    # We need feature names after preprocessing (for OneHot encoding we extract feature names)
    # create preprocessor-fit to get names:
    preprocessor.fit(X_train)
    # numeric feature names remain
    num_features = numeric_cols
    # get categorical feature names from OneHotEncoder
    cat_features = []
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    # If OneHotEncoder is used, we can get categories_
    cat_cols = categorical_cols
    try:
        ohe_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    except Exception:
        ohe_feature_names = []
    all_features = num_features + ohe_feature_names
    fi = best_clf.feature_importances_
    fi_df = pd.DataFrame({'feature': all_features, 'importance': fi})
    fi_df = fi_df.sort_values('importance', ascending=False).head(30)
    fi_df.to_csv('feature_importances.csv', index=False)
    print("Saved feature_importances.csv")

# -------------------------
# 6. Plot ROC curve for best model
# -------------------------
if hasattr(best_pipe, 'predict_proba'):
    y_proba_best = best_pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba_best)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Best Model')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_best_model.png')
    print("Saved ROC curve to roc_best_model.png")

# -------------------------
# 7. Save a sample validation CSV with predictions
# -------------------------
test_with_preds = X_test.copy()
test_with_preds['true_target'] = y_test.values
test_with_preds['predicted'] = best_pipe.predict(X_test)
if hasattr(best_pipe, 'predict_proba'):
    test_with_preds['pred_proba'] = best_pipe.predict_proba(X_test)[:,1]
test_with_preds.to_csv('test_predictions.csv', index=False)
print("Saved test_predictions.csv")

print("\nAll artifacts saved: best_model.joblib, (feature_importances.csv if available), roc_best_model.png, test_predictions.csv")
