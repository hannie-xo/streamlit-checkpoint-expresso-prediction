import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Load saved model and encoder
# ------------------------------
model, target_enc, categorical_cols, feature_cols, categories_dict = joblib.load(
    r"C:\Users\User\OneDrive\Documents\Expresso Churn App\expresso_churn_predictor\model.pkl"
)


st.title("Expresso Churn Prediction App")
st.write("Please enter customer details for churn prediction:")

# ------------------------------
# Categorical features input
# ------------------------------
st.subheader("Categorical Features")
cat_inputs = {}

for col in categorical_cols:
    options = categories_dict[col]  
    val = st.selectbox(col, options)  
    cat_inputs[col] = val

cat_df = pd.DataFrame([cat_inputs])
cat_encoded = target_enc.transform(cat_df)  # encode

# ------------------------------
# Numeric features input
# ------------------------------
st.subheader("Numeric Features")

# Extract numeric columns (all features except categorical + CHURN)
numeric_cols = [col for col in feature_cols if col not in categorical_cols]

numeric_inputs = {}
for col in numeric_cols:
    val = st.number_input(col, value=0.0, format="%.2f")
    numeric_inputs[col] = val

num_df = pd.DataFrame([numeric_inputs])

# ------------------------------
# Combine categorical + numeric
# ------------------------------
df_input = pd.concat([num_df, cat_encoded], axis=1)


df_input = df_input[feature_cols]

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Churn"):
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    st.write("---")
    st.subheader("Prediction Result:")

    if prediction == 1:
        st.error(f'Customer is likely to churn.\nProbability: {probability:.2%}')
    else:
        st.success(f'Customer is unlikely to churn.\nProbability: {probability:.2%}')

