# Telco Customer Churn Analysis

This project analyzes customer churn data for a telecommunications company using machine learning. The application is built with Streamlit and provides an interactive interface for exploring the data and making predictions.

## Features

- Data exploration and visualization
- Interactive charts and graphs
- Customer churn prediction using Random Forest
- Feature importance analysis

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd Telco-Customer-Churn
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8501
   ```

## Project Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Dataset
- `Abdelrhamanwael_Telco Customer Churn.ipynb`: Original Jupyter notebook with analysis

## Deployment

### Local Deployment
Follow the installation and running instructions above to run the app locally.

### Cloud Deployment (Streamlit Cloud)
1. Create a GitHub repository for your project
2. Push your code to the repository
3. Go to [Streamlit Cloud](https://streamlit.io/cloud)
4. Sign in with your GitHub account
5. Click "New app" and select your repository
6. Set the main file path to `app.py`
7. Click "Deploy!"

### Docker Deployment
1. Build the Docker image:
   ```
   docker build -t telco-churn-app .
   ```
2. Run the container:
   ```
   docker run -p 8501:8501 telco-churn-app
   ```
3. Access the app at `http://localhost:8501`

## Usage

1. **Data Overview**: Explore the dataset structure and basic statistics
2. **Exploratory Analysis**: Visualize data distributions and relationships
3. **Churn Prediction**: View model performance and feature importance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
