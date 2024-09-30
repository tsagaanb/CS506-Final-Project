# Starbucks Wait Time Prediction at GSU

## Description
This project aims to predict the wait time of Starbucks orders at Georgia State University (GSU) based on the time of day and day of the week. By analyzing patterns in wait times, the project will help students make informed decisions about when to place their orders.

## Clear Goal(s)
- Predicting the wait time of Starbucks at GSU depending on the time of the day and day of the week.

## Data Collection
- **Data to be Collected:** The number of students in line and the estimated wait time at Starbucks.
- **Collection Method:** Data will be collected through the Grubhub application at 10-minute intervals during the open hours, Monday through Sunday.

## Data Modeling
- **Modeling Techniques:**
  - ARIMA (AutoRegressive Integrated Moving Average) for time series forecasting.
  - Linear regression to understand trends in wait times based on time of day.
  - Machine learning methods like Random Forests or XGBoost to capture nonlinear relationships, using features such as time of day and day of the week.

## Data Visualization
- **Visualization Methods:**
  - Line graphs showing wait time trends throughout the day.
  - Heatmaps displaying wait times by hour and day.
  - Box plots illustrating the distribution of wait times across different times.

## Test Plan
- **Testing Approach:**
  - Withhold 20% of the collected data for testing.
  - Train on data collected over a specified period (e.g., one month) and validate the model on the data collected during the subsequent month.
  - Use k-fold cross-validation to ensure robustness of the model across different time frames.
  - **Training Data Collection:** Throughout October (October 4th - October 31st).
  - **Testing Data Collection:** Throughout November (November 1st - November 28th).
