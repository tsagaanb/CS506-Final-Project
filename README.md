# MidSemester Report: Starbucks Wait Time Prediction at George Sherman Union
## Description
This project aims to predict the wait time of Starbucks orders at George Sherman Union (GSU) based on the time of day and day of the week. In other words, the user should be able to input any time during Starbucks' business hours and have returned the estimated wait time based on the current minute and day. By analyzing patterns in wait times, the project will help students make informed decisions about when to place their orders.

## Clear Goal(s)
- Predicting the wait time of Starbucks at GSU depending on the time of the day and day of the week.

## Data Collection
To make accurate predictions, we collected the following key data points:

- **Number of students in line**
- **Estimated wait time for Starbucks orders**

Data was collected through the Grubhub app at 15-minute intervals during all business hours, Monday through Sunday. This approach allowed us to build a dataset that captures typical wait-time patterns.

## Data Processing and Visualization

### Data Processing
   - We combined `Date` and `Time` columns into a single `DateTime` format and set it as the index for time-based analysis.
   - We extracted `DayOfWeek` and `TimeOfDay` to identify daily and weekly patterns in wait times.
   - Finally, we grouped the data by day of the week and 15-minute intervals to calculate average wait times.

### Visualization
   - We graphed the average wait times of every day of the week.
   - We graphed the Mean Squared Errors of all the models
   - We picked the best model (Decision Tree) and graphed the predicted wait time and the actual wait time for every day of the week

## Feature Preparation and Preprocessing
Our feature set includes the DayOfWeek and Hour columns, and our target is the wait time in minutes. Since our model needs numeric data, we preprocess it by one-hot encoding the DayOfWeek feature to create separate columns for each day and standardizing the Hour feature. This preprocessing is encapsulated in a pipeline, making the data ready for any machine learning model we choose.

## Modeling Methods

1. **Linear Regression (Baseline)**  
   Linear Regression served as our baseline model. Due to the non-linear nature of the data, this model struggled, resulting in a higher Mean Squared Error (MSE) of approximately 13.5.

2. **K-Nearest Neighbors (KNN) for Nonlinear Patterns**  
   KNN was chosen to better capture non-linear patterns. This approach yielded an MSE of about 6.2, improving on Linear Regression but still not optimal.

3. **Decision Tree Regression**  
   To capture complex patterns, we implemented Decision Tree Regression. This model achieved a significantly lower MSE of approximately 5.68, showing stronger performance.

4. **Random Forest Regression for Enhanced Performance**  
   Random Forest was tested to improve accuracy further by capturing intricate patterns in the data. However, it achieved an MSE similar to Decision Tree at around 5.7.

## Preliminary Results
The Mean Squared Errors from each model were as follows:

- **Linear Regression**: MSE ≈ 13.5
- **K-Nearest Neighbors (KNN)**: MSE ≈ 6.2
- **Decision Tree**: MSE ≈ 5.68
- **Random Forest**: MSE ≈ 5.7

## Conclusion
The Decision Tree model emerged as the most effective for our dataset, achieving the lowest MSE. Based on these results, we chose Decision Tree Regression as our primary model for predicting Starbucks wait times. Moving forward, we plan to refine the model further by incorporating additional factors, such as events or promotions, that could influence wait times.

## Future Plans: Test Plan
- **Testing Approach:**
  - **Testing Data Collection:** Throughout November (November 1st - November 28th).
  - Use the testing data to see if our model is achieving promising results.
  - Upgrade our model to fit the training data better if we do not achieve the results that we wanted and test on the testing data again.
 

