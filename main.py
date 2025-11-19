import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
import joblib
from ydata_profiling import ProfileReport

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
df = pd.read_csv('Expresso_churn_dataset.csv')

# Optional: Explore dataset
print("--- DataFrame Head ---")
print(df.head())
print("\n--- General Info ---")
df.info()
print("\n--- Descriptive Statistics ---")
print(df.describe(include='all'))

# -----------------------------
# 2️⃣ Create Pandas Profiling Report (Optional)
# -----------------------------
profile = ProfileReport(df.sample(n=100000, random_state=42), title='Expresso Churn Sampled Report')
profile.to_file("expresso_report.html")
print("Profiling report saved as expresso_report.html")

# -----------------------------
# 3️⃣ Preprocessing
# -----------------------------
# Drop duplicates and unique ID
df.drop_duplicates(inplace=True)
df = df.drop(columns=['user_id'])

# Define columns
categorical_cols = ["REGION", "TENURE", "MRG", "TOP_PACK"]
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_cols.remove("CHURN")  # target column

# Fill missing values
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# -----------------------------
# 4️⃣ Split Features & Target
# -----------------------------
X = df.drop("CHURN", axis=1)
y = df["CHURN"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5️⃣ Target Encoding for categorical columns
# -----------------------------
target_enc = ce.TargetEncoder(cols=categorical_cols)
X_train[categorical_cols] = target_enc.fit_transform(X_train[categorical_cols], y_train)
X_test[categorical_cols] = target_enc.transform(X_test[categorical_cols])

# -----------------------------
# 6️⃣ Train RandomForest
# -----------------------------
model = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

# -----------------------------
# 7️⃣ Save model + encoder + metadata for Streamlit
# -----------------------------

#  Save full feature order (needed for prediction alignment)
feature_cols = X_train.columns.tolist()

# Save unique categories for dropdowns in Streamlit
categories_dict = {
    col: df[col].unique().tolist() for col in categorical_cols
}

#  Save everything in one pickle file
joblib.dump(
    (model, target_enc, categorical_cols, feature_cols, categories_dict),
    "model.pkl"
)

print("Model saved as model.pkl")
