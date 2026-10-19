
import joblib
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# 1. Define project paths
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "churn_demo.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"

# 2. Create the models folder if it does not exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 3. Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Make sure churn_demo.csv is inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

# 4. Validate dataset
target = "churn"

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

if df.empty:
    raise ValueError("The dataset is empty.")

if df[target].nunique() != 2:
    raise ValueError("This training script expects a binary churn target.")

X = df.drop(columns=[target])
y = df[target]

# Ensure the target has no missing values
if y.isna().any():
    raise ValueError("The target column 'churn' contains missing values.")

# 5. Identify numeric and categorical features
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

if not numeric_features and not categorical_features:
    raise ValueError("No usable feature columns found.")

print(f"Dataset shape: {df.shape}")
print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# 6. Build preprocessing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

# 7. Build complete ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

# 8. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 9. Train model
print("\nTraining model...")
model.fit(X_train, y_train)

# 10. Make predictions
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

# 11. Evaluate model
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")

# ROC-AUC requires both classes in the test set
if y_test.nunique() == 2:
    print(f"ROC-AUC:  {roc_auc_score(y_test, proba):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, pred, zero_division=0))

# 12. Save trained model
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully to:\n{MODEL_PATH}")
