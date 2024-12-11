# **Final Report: Starbucks Wait Time Prediction at George Sherman Union**

## **Project Overview**
This project predicts Starbucks wait times at George Sherman Union (GSU) based on the time of day and the day of the week. By analyzing historical wait-time patterns, the system allows users to input a specific time during Starbucks' business hours and receive an estimated wait time. The goal is to assist students in making informed decisions about when to visit Starbucks, optimizing their time and convenience.

---

## **Objective**
To accurately predict the wait time for Starbucks at GSU using machine learning models and time-based features. The project aims to minimize prediction error while identifying patterns that influence wait times throughout the week.

---

## **Data Collection and Methodology**

### **Key Data Points**
- **Number of students in line**  
- **Estimated wait time for Starbucks orders**  

### **Data Collection Process**
- Data was collected through the Grubhub app at 15-minute intervals during Starbucks' business hours (Monday through Sunday).  
- The data spanned multiple weeks to ensure robust coverage of typical business patterns. This includes peak hours (morning rush, lunch hours) and non-peak hours for comparison.  
- **Total Data Points Collected:** [Insert total number]  
- Outliers and anomalies, such as extreme wait times due to special events, were analyzed separately to understand their impact on predictions.  

---

## **Data Processing and Visualization**

### **Data Processing Steps**
1. **Date-Time Merging:** Combined `Date` and `Time` columns into a single `DateTime` column for chronological analysis.  
2. **Feature Engineering:** Extracted new features like `DayOfWeek` (categorical) and `TimeOfDay` (hourly bins) to provide more granular insights.  
3. **Data Aggregation:** Grouped data by day of the week and 15-minute intervals to compute average wait times for trend analysis.  

### **Visualization**
- **Daily Patterns:**  
  Visualized average wait times for each day of the week, showing clear trends in customer behavior (e.g., higher wait times on Mondays and Fridays).  
- **Model Performance Comparison:**  
  Plotted the Mean Squared Errors (MSE) of all tested models to visually compare their accuracy.  
- **Best Model Analysis:**  
  Graphed predicted vs. actual wait times for each day of the week using the Decision Tree Regression model to highlight its precision.  

---

## **Feature Preparation and Preprocessing**

### **Features**
1. **Day of the Week (One-Hot Encoded):** Each day represented as a separate binary feature.  
2. **Hour of the Day (Standardized):** Converted into a numeric scale and normalized for uniformity across the dataset.  

### **Target Variable**
- Wait time in minutes (continuous numeric variable).  

### **Preprocessing Workflow**
- Implemented a preprocessing pipeline to:
  - Handle missing or inconsistent data.
  - Apply one-hot encoding for categorical features.
  - Standardize numeric features to improve model performance.
- The processed dataset was split into training (80%) and testing (20%) subsets to validate model accuracy.

---

## **Modeling and Results**

### **Models Tested**
1. **Linear Regression (Baseline):**  
   - **MSE:** ~13.5  
   - Observations: Struggled with non-linear patterns in the dataset. Served as a performance benchmark.  

2. **K-Nearest Neighbors (KNN):**  
   - **MSE:** ~6.2  
   - Observations: Captured some non-linear relationships but was limited by its sensitivity to the number of neighbors.  

3. **Decision Tree Regression:**  
   - **MSE:** ~5.68  
   - Observations: Best overall performance due to its ability to model complex patterns in wait times effectively.  

4. **Random Forest Regression:**  
   - **MSE:** ~5.7  
   - Observations: Marginally better than KNN but on par with Decision Tree. Computationally more expensive without significant improvement.  

### **Model Comparison**
| Model              | Mean Squared Error (MSE) |
|---------------------|--------------------------|
| Linear Regression   | ~13.5                   |
| K-Nearest Neighbors | ~6.2                    |
| Decision Tree       | ~5.68                   |
| Random Forest       | ~5.7                    |

### **Final Model**
The **Decision Tree Regression model** was selected due to its ability to capture intricate wait-time patterns with the lowest MSE.

---

## **Results and Analysis**

### **Key Findings**
1. **Time-Based Trends:**  
   - Wait times are consistently higher during weekday mornings and lunchtime.  
   - Evenings and weekends exhibit lower wait times, except during specific events.  

2. **Model Performance:**  
   - The Decision Tree model provided the most accurate predictions, with minimal variance between predicted and actual wait times.  

3. **Outliers:**  
   - Extreme wait times were linked to specific events or promotions, providing insights into additional variables that could enhance the model.

---

## **Conclusion**
The project successfully developed a predictive model for Starbucks wait times at GSU, with the Decision Tree Regression model achieving the lowest MSE (~5.68). By leveraging historical data and machine learning techniques, the system offers a practical tool for optimizing student schedules.

---

## **Next Steps**
1. **Dataset Expansion:**  
   - Incorporate more granular data, such as minute-level intervals and customer order sizes.  
2. **Live Data Integration:**  
   - Develop a real-time prediction system using live Grubhub or in-store data streams.  
3. **Additional Features:**  
   - Include external factors like promotions, weather conditions, and campus events for improved accuracy.  

---

