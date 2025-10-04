import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# --- Caching Functions to Improve Performance ---

# Use st.cache_data to load data and process it only once
@st.cache_data
def load_and_process_data():
    """Loads the dataset and performs all necessary preprocessing."""
    # 1. Load Data
    # CORRECTED PATH: Relative to the project root
    df = pd.read_csv("data/hmeq.csv")

    # 2. Handle Missing Values
    num_cols = df.select_dtypes(include=['float64','int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    # 3. Encode Categoricals
    df_processed = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df, df_processed

# Use st.cache_resource to train the model and cache it
@st.cache_resource
def train_model(df_processed):
    """Trains the Decision Tree model."""
    # 4. Split Data
    X = df_processed.drop("BAD", axis=1)
    y = df_processed["BAD"]

    # No need for train_test_split here as we train on all data for the app
    # But we need X.columns for the prediction part later
    
    # 5. Train Decision Tree Model
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
    tree.fit(X, y)
    return tree, X.columns

# --- Main Application ---

st.set_page_config(page_title="Loan Default Predictor", layout="wide")

st.title("💰 Loan Default Prediction App")
st.write(
    "This app predicts the risk of a loan applicant defaulting based on their financial profile. "
    "Enter the applicant's details in the sidebar to get a prediction."
)

# Load data and train model using cached functions
df_original, df_processed = load_and_process_data()
model, trained_columns = train_model(df_processed)

# --- Sidebar for User Input ---

st.sidebar.header("Applicant Information")

def get_user_input():
    """Creates sidebar widgets and returns user input as a DataFrame."""
    # Use median values from the original dataframe as defaults
    # to provide a realistic starting point for the user.
    loan = st.sidebar.number_input("Loan Amount ($)", min_value=1000, value=int(df_original['LOAN'].median()), step=500)
    mortdue = st.sidebar.number_input("Mortgage Due ($)", min_value=0, value=int(df_original['MORTDUE'].median()), step=1000)
    value = st.sidebar.number_input("Property Value ($)", min_value=0, value=int(df_original['VALUE'].median()), step=1000)
    yoj = st.sidebar.number_input("Years at Job", min_value=0, max_value=50, value=int(df_original['YOJ'].median()), step=1)
    derog = st.sidebar.number_input("Derogatory Reports", min_value=0, max_value=10, value=int(df_original['DEROG'].median()))
    delinq = st.sidebar.number_input("Delinquent Lines", min_value=0, max_value=20, value=int(df_original['DELINQ'].median()))
    clage = st.sidebar.number_input("Age of Oldest Trade Line (months)", min_value=0.0, value=df_original['CLAGE'].median(), step=1.0)
    ninq = st.sidebar.number_input("Recent Credit Inquiries", min_value=0, value=int(df_original['NINQ'].median()))
    clno = st.sidebar.number_input("Number of Credit Lines", min_value=0, value=int(df_original['CLNO'].median()))
    debtinc = st.sidebar.number_input("Debt-to-Income Ratio", min_value=0.0, value=df_original['DEBTINC'].median(), step=0.1)
    
    # For categorical variables, get the unique values from the original data
    reason = st.sidebar.selectbox("Reason for Loan", options=df_original['REASON'].unique())
    job = st.sidebar.selectbox("Job Category", options=df_original['JOB'].unique())

    # Create a dictionary from the inputs
    input_data = {
        'LOAN': loan, 'MORTDUE': mortdue, 'VALUE': value, 'YOJ': yoj, 'DEROG': derog,
        'DELINQ': delinq, 'CLAGE': clage, 'NINQ': ninq, 'CLNO': clno, 'DEBTINC': debtinc,
        'REASON': reason, 'JOB': job
    }
    
    # Convert to a DataFrame
    input_df = pd.DataFrame([input_data])
    return input_df

# Get input from the user
user_input_df = get_user_input()

# --- Prediction and Display ---

st.subheader("Prediction")

# One-hot encode the user input
input_processed = pd.get_dummies(user_input_df)

# Align columns with the training data
# This ensures the input has the same features as the model was trained on
input_aligned = input_processed.reindex(columns=trained_columns, fill_value=0)

if st.sidebar.button("Predict"):
    # Make prediction
    prediction = model.predict(input_aligned)[0]
    prediction_proba = model.predict_proba(input_aligned)[0]

    # Display prediction result
    if prediction == 1:
        st.error(f"Prediction: High Risk of Default (Probability: {prediction_proba[1]:.2f})")
        st.warning("It is recommended to decline this loan application.")
    else:
        st.success(f"Prediction: Low Risk of Default (Probability: {prediction_proba[0]:.2f})")
        st.info("This loan application is likely safe to approve.")
    
    # Display the user input data for confirmation
    st.write("---")
    st.subheader("Applicant Data Entered:")
    st.table(user_input_df)

else:
    st.info("Click the 'Predict' button in the sidebar to see the loan default risk.")

# Optional: Add an expander to show the decision tree visualization
with st.expander("Show Model Visualization"):
    # CORRECTED PATH: Relative to the project root
    st.image("outputs/tree_plot.png")