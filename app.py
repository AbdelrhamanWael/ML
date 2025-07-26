import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Set page config
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    # Load your dataset
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return df

def preprocess_data(df):
    # Data preprocessing steps from your notebook
    df = df.drop(['customerID'], axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df

def train_model(X, y):
    # Train a simple Random Forest model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_smote, y_train_smote)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, classification_report(y_test, y_pred)

def main():
    st.title("📊 Telco Customer Churn Analysis")
    
    # Load data
    df = load_data()
    df = preprocess_data(df)
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Data Overview", "Exploratory Analysis", "Churn Prediction"]
    )
    
    if page == "Data Overview":
        st.header("Dataset Overview")
        st.write("### First 5 rows of the dataset")
        st.dataframe(df.head())
        
        st.write("### Dataset Information")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        
        st.write("### Column Types")
        st.write(df.dtypes)
        
        st.write("### Missing Values")
        st.write(df.isnull().sum())
    
    elif page == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        
        # Churn distribution
        st.write("### Churn Distribution")
        fig, ax = plt.subplots()
        df['Churn'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Churn Distribution')
        ax.set_xlabel('Churn')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        
        # Numerical features distribution
        st.write("### Distribution of Numerical Features")
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            st.pyplot(fig)
        
        # Categorical features
        st.write("### Categorical Features vs Churn")
        cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                   'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                   'Contract', 'PaperlessBilling', 'PaymentMethod']
        
        selected_cat = st.selectbox("Select a categorical feature", cat_cols)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x=selected_cat, hue='Churn', ax=ax)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    
    elif page == "Churn Prediction":
        st.header("Churn Prediction")
        
        with st.spinner("Training the model..."):
            # Prepare features and target
            X = df.drop('Churn', axis=1)
            y = df['Churn']
            
            # Convert categorical variables to numerical
            cat_cols = X.select_dtypes(include=['object']).columns
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Train model
            model, accuracy, report = train_model(X, y)
        
        st.success("Model trained successfully!")
        
        st.write("### Model Performance")
        st.write(f"Accuracy: {accuracy:.2f}")
        
        st.write("### Classification Report")
        st.text(report)
        
        # Feature importance
        st.write("### Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
        ax.set_title('Top 10 Important Features')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
