import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stSelectbox>div>div>div {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# File paths
DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
MODEL_PATH = 'churn_model.joblib'

@st.cache_data
def load_data():
    # Load your dataset
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Data preprocessing steps
    df = df.drop(['customerID'], axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    
    return df

def prepare_features(df):
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Define categorical and numerical features
    categorical_features = [col for col in X.columns if X[col].dtype == 'object']
    numerical_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    
    # Create transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse=False)
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor, categorical_features, numerical_features

def train_model(X, y, preprocessor):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess the data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_preprocessed, y_train)
    
    # Train the model
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_smote, y_train_smote)
    
    # Make predictions
    y_pred = model.predict(X_test_preprocessed)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, class_report, conf_matrix, preprocessor

def save_model(model, preprocessor):
    """Save the trained model and preprocessor"""
    model_data = {
        'model': model,
        'preprocessor': preprocessor
    }
    joblib.dump(model_data, MODEL_PATH)

def load_saved_model():
    """Load the saved model and preprocessor"""
    if os.path.exists(MODEL_PATH):
        model_data = joblib.load(MODEL_PATH)
        return model_data['model'], model_data['preprocessor']
    return None, None

def plot_confusion_matrix(conf_matrix):
    """Plot confusion matrix"""
    fig = px.imshow(
        conf_matrix,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Not Churn', 'Churn'],
        y=['Not Churn', 'Churn'],
        text_auto=True,
        aspect="auto"
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    return fig

def show_metrics(accuracy, class_report, conf_matrix):
    """Display model metrics"""
    st.subheader("Model Performance")
    
    # Display accuracy
    st.metric("Accuracy", f"{accuracy:.2%}")
    
    # Display classification report
    st.subheader("Classification Report")
    class_report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(class_report_df)
    
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    conf_fig = plot_confusion_matrix(conf_matrix)
    st.plotly_chart(conf_fig, use_container_width=True)

def get_user_input(df, categorical_features, numerical_features):
    """Get user input for prediction"""
    st.sidebar.header("Customer Information")
    
    # Create input fields based on feature types
    input_data = {}
    
    # Add numerical inputs
    for feature in numerical_features:
        if feature == 'tenure':
            input_data[feature] = st.sidebar.slider("Tenure (months)", 0, 100, 12)
        elif feature == 'MonthlyCharges':
            input_data[feature] = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=50.0, step=1.0)
        elif feature == 'TotalCharges':
            input_data[feature] = st.sidebar.number_input("Total Charges", min_value=0.0, value=500.0, step=10.0)
    
    # Add categorical inputs
    for feature in categorical_features:
        if feature != 'Churn':  # Skip target variable
            options = sorted(df[feature].unique())
            input_data[feature] = st.sidebar.selectbox(
                f"{feature.replace('_', ' ').title()}",
                options
            )
    
    return pd.DataFrame([input_data])

def main():
    st.title("📊 Telco Customer Churn Analysis")
    
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Prepare features and target
    X, y, preprocessor, categorical_features, numerical_features = prepare_features(df)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Data Overview", "Exploratory Analysis", "Churn Prediction", "Train Model"]
    )
    
    if page == "Data Overview":
        st.header("📋 Dataset Overview")
        
        # Show basic info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Rows", df.shape[0])
        with col2:
            st.metric("Number of Columns", df.shape[1])
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Show data types
        st.subheader("Data Types")
        st.write(df.dtypes)
        
        # Show missing values
        st.subheader("Missing Values")
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Values']
        st.dataframe(missing_data[missing_data['Missing Values'] > 0])
    
    elif page == "Exploratory Analysis":
        st.header("🔍 Exploratory Data Analysis")
        
        # Churn distribution
        st.subheader("Churn Distribution")
        fig = px.pie(
            df, 
            names='Churn', 
            title='Churn Distribution',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Numerical features distribution
        st.subheader("Numerical Features Distribution")
        num_col = st.selectbox("Select a numerical feature", numerical_features)
        
        fig = px.histogram(
            df, 
            x=num_col, 
            color='Churn',
            marginal='box',
            title=f'Distribution of {num_col} by Churn',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Categorical features distribution
        st.subheader("Categorical Features Distribution")
        cat_col = st.selectbox("Select a categorical feature", categorical_features)
        
        fig = px.histogram(
            df, 
            x=cat_col, 
            color='Churn',
            title=f'Distribution of {cat_col} by Churn',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Train Model":
        st.header("🤖 Train Churn Prediction Model")
        
        if st.button("Train Model"):
            with st.spinner("Training model... This may take a few minutes."):
                model, accuracy, class_report, conf_matrix, preprocessor = train_model(X, y, preprocessor)
                save_model(model, preprocessor)
                st.success("Model trained and saved successfully!")
                show_metrics(accuracy, class_report, conf_matrix)
        else:
            st.info("Click the button above to train the model.")
    
    elif page == "Churn Prediction":
        st.header("🔮 Predict Customer Churn")
        
        # Load saved model or show message
        model, preprocessor = load_saved_model()
        
        if model is None or preprocessor is None:
            st.warning("Please train the model first from the 'Train Model' page.")
            return
        
        # Get user input
        st.sidebar.subheader("Enter Customer Details")
        input_df = get_user_input(df, categorical_features, numerical_features)
        
        # Display user input
        st.subheader("Customer Information")
        st.dataframe(input_df)
        
        # Make prediction
        if st.button("Predict Churn"):
            try:
                # Preprocess input
                input_processed = preprocessor.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_processed)[0]
                prediction_proba = model.predict_proba(input_processed)[0]
                
                # Display result
                st.subheader("Prediction Result")
                
                col1, col2 = st.columns(2)
                with col1:
                    if prediction == 'Yes':
                        st.error("Churn: Yes")
                    else:
                        st.success("Churn: No")
                
                with col2:
                    st.metric("Confidence", f"{max(prediction_proba)*100:.1f}%")
                
                # Show probability distribution
                proba_df = pd.DataFrame({
                    'Churn': ['No', 'Yes'],
                    'Probability': prediction_proba
                })
                
                fig = px.bar(
                    proba_df, 
                    x='Churn', 
                    y='Probability',
                    title='Prediction Probabilities',
                    color='Churn',
                    color_discrete_map={'No': '#2ecc71', 'Yes': '#e74c3c'},
                    text_auto=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
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
