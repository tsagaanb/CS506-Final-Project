

# Starbucks Wait Time Prediction Web App

## Overview
This project predicts the wait time for Starbucks orders at George Sherman Union (GSU) based on the day of the week and time of day. By analyzing patterns in wait times, the app helps students make informed decisions about when to place their orders, saving time and enhancing convenience.

## Table of Contents
- How to Build and Run the Code
- Test Code and GitHub Workflow
- Data Processing and Modeling
- Visualizations
- Results

## How to Build and Run the Code

### Prerequisites
Ensure you have the following installed:
- Python 3.9 or later
- pip (Python package manager)

### Steps
1. **Clone the Repository**
   ```bash
   git clone git@github.com:tsagaanb/CS506-Final-Project.git
   cd CS506-Final-Project
   ```

2. **Install Dependencies**
   Use the provided Makefile for streamlined setup:
   ```bash
   make install
   ```
   This command creates a virtual environment and installs all dependencies from `requirements.txt`.

3. **Run the Flask Application**
   ```bash
   make run
   ```
   The application will start running at `http://127.0.0.1:3000/`.

4. **Alternative Manual Setup**
   If you prefer manual setup:
   - Set up a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate   # On Mac/Linux
     venv\Scripts\activate      # On Windows
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Run the application:
     ```bash
     python app.py
     ```

## Test Code and GitHub Workflow

### Running Tests
The `tests/` folder contains a test script for verifying model predictions and application functionality. Run all tests using:
```bash
make test
```

### GitHub Workflow
The project includes a CI/CD workflow (`.github/workflows/main.yml`) that:
- Runs tests automatically on every push to the repository.
- Validates data preprocessing logic, model predictions, and Flask pipeline functionality.

## Data Processing and Modeling

### Data Collection
- **Source:** Data was collected via the Grubhub app at 15-minute intervals during business hours (Monday through Sunday).
- **Features:** 
  - `DayOfWeek`: Categorical representation of the day.
  - `Hour`: Numeric representation of the hour.
  - `IsWeekend`: Binary feature for weekends.
  - `PartOfDay`: Morning, Afternoon, or Evening.

### Data Processing
1. **Feature Engineering:**
   - Cyclic encoding for `Hour` using sine and cosine transformations.
   - Binary indicator for weekends (`IsWeekend`).
   - Categorical encoding for `PartOfDay`.

2. **Preprocessing Pipeline:** 
   - One-hot encoding for categorical variables.
   - Standardization for numerical features.

### Model
We evaluated multiple models:
- Linear Regression: Baseline with MSE ≈ 13.5.
- K-Nearest Neighbors (KNN): Non-linear patterns, MSE ≈ 6.2.
- Decision Tree Regression: Complex patterns, MSE ≈ 5.68.
- Gradient Boosting Regressor: Final model with MSE ≈ 5.58.

### Model Performance 
| Metric                 | Training Data | Testing Data |
|------------------------|---------------|--------------|
| Mean Squared Error (MSE) | 5.61          | 10.70        |
| Root Mean Squared Error (RMSE) | 2.36          | 3.27         |
| Mean Absolute Error (MAE) | 2.28          | 2.28         |


## Visualizations

### Key Visualizations
1. **Daily Average Wait Times:**
Interactive graphs displaying average wait times for each day of the week help users visualize trends.

2. **Actual vs Predicted Wait Times:**
For every day of the week, graphs compare actual wait times with model-predicted values, providing transparency into model performance.

3. **Feature Importance:**
A plot showing the importance of features such as DayOfWeek, Hour, and PartOfDay, highlighting what drives the model's predictions.

### Sample Outputs
Generated plots are saved in the `static/plots/` directory and displayed on the `/our_model` route.

## Progress Since Midterm Report
- We collected data from November 1st to November 14th (2 weeks) and used that as our testing data. We fit our testing data to the model and tried to improve the test MSE.
- We experimented with various pipelining techniques and regressors, including RandomForestRegressor, XGBRegressor, GradientBoostingRegressor, VotingRegressor, and Cross-Validation with different hyperparameter tuning approaches, to enhance our decision tree model and minimize the test Mean Squared Error (MSE). However, most methods resulted in negligible improvements. The only method that showed a meaningful improvement was GradientBoostingRegressor, which reduced the test MSE from 10.75 to 10.70.

## Results

### Achieved Goals
- **Prediction Functionality:** Users can input a day and time to get predicted wait times.
- **Interactive Visualizations:** 
  - Actual vs predicted wait times for each day of the week.
  - Predicted wait times for an entire day based on user input.
- **Model Performance:** The Gradient Boosting model achieved a testing MSE of **10.70**.

### Final Visualizations
Interactive plots and graphs are embedded within the app, providing users with insights and transparency.
