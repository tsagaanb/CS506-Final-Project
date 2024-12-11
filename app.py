from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/plots'

# Load data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Preprocess data
def preprocess_data(data):
    data['Date'] = data['Date'].astype(str)
    data['Time'] = data['Time'].astype(str)
    data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')
    data.dropna(subset=['DateTime'], inplace=True)

    # Extract day of the week and hour of the day as features
    data['DayOfWeek'] = data['DateTime'].dt.day_name()
    data['Hour'] = data['DateTime'].dt.hour
    data['IsWeekend'] = data['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)
    data['PartOfDay'] = pd.cut(data['Hour'], bins=[-1, 11, 16, 24], labels=['Morning', 'Afternoon', 'Evening'])

    # Cyclic features for hour
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)

    # Convert 'Wait Time (mins)' to numeric and drop rows with missing target values
    data['Wait Time (mins)'] = pd.to_numeric(data['Wait Time (mins)'], errors='coerce')
    data.dropna(subset=['Wait Time (mins)'], inplace=True)
    return data

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Features and target
X_train = train_data[['DayOfWeek', 'Hour_sin', 'Hour_cos', 'IsWeekend', 'PartOfDay']]
y_train = train_data['Wait Time (mins)']
X_test = test_data[['DayOfWeek', 'Hour_sin', 'Hour_cos', 'IsWeekend', 'PartOfDay']]
y_test = test_data['Wait Time (mins)']

# Train Gradient Boosting model
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['DayOfWeek', 'PartOfDay']),
        ('num', StandardScaler(), ['Hour_sin', 'Hour_cos', 'IsWeekend'])
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(
        random_state=42,
        n_estimators=150,          # Reduce number of trees
        learning_rate=0.1,         # Moderate learning rate
        max_depth=4,               # Shallower trees to prevent overfitting
        min_samples_split=15,      # Require more samples to split
        min_samples_leaf=10,       # Require more samples per leaf node
        subsample=0.8,             # Use a subset of samples for training each tree
        max_features='sqrt'        # Use a subset of features for each split
    ))
])


pipeline.fit(X_train, y_train)
y_test_pred = pipeline.predict(X_test)
y_train_pred = pipeline.predict(X_train)

# Calculate metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = train_mse ** 0.5
test_rmse = test_mse ** 0.5
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# Save plots for accuracy
def save_plot(actual, predicted, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual', marker='o')
    plt.plot(predicted, label='Predicted', marker='x')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    plt.close()


# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    day = request.form['day']  # Get the selected day
    time = request.form['time']  # Get the selected time in HH:MM format

    # Extract the hour from the time string (e.g., "13:30" -> 13)
    hour = int(time.split(':')[0])
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    is_weekend = int(day in ['Saturday', 'Sunday'])
    part_of_day = 'Morning' if hour < 12 else 'Afternoon' if hour < 17 else 'Evening'

    # Prepare input data for the prediction
    input_data = pd.DataFrame({'DayOfWeek': [day], 'Hour_sin': [hour_sin], 'Hour_cos': [hour_cos], 'IsWeekend': [is_weekend], 'PartOfDay': [part_of_day]})
    prediction = pipeline.predict(input_data)[0]  # Predict wait time

    # Generate predicted wait times for the entire day
    hours = list(range(7, 20 if day not in ["Saturday", "Sunday"] else 19))  # Operating hours
    input_day_data = pd.DataFrame({
        'DayOfWeek': [day] * len(hours),
        'Hour_sin': np.sin(2 * np.pi * np.array(hours) / 24),
        'Hour_cos': np.cos(2 * np.pi * np.array(hours) / 24),
        'IsWeekend': [is_weekend] * len(hours),
        'PartOfDay': ['Morning' if h < 12 else 'Afternoon' if h < 17 else 'Evening' for h in hours]
    })
    daily_predictions = pipeline.predict(input_day_data)

    # Plot the predictions for the chosen day
    plt.figure(figsize=(10, 6))
    plt.plot(hours, daily_predictions, marker='o', label='Predicted Wait Time')
    plt.title(f'Predicted Wait Times for {day}')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Predicted Wait Time (mins)')
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plot_filename = f'static/plots/{day}_predictions.png'
    plt.savefig(plot_filename)
    plt.close()

    # Render the home template with the prediction and plot
    return render_template(
        'home.html',
        prediction=round(prediction, 2),
        day=day,
        time=time,
        plot_filename=plot_filename
    )

@app.route('/our_model')
def our_model():
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Generate plots for each day
    for day in days_of_week:
        day_data = test_data[test_data['DayOfWeek'] == day]
        if day_data.empty:
            continue

        avg_day_data = day_data.groupby('Hour').agg({'Wait Time (mins)': 'mean'}).reset_index()
        avg_day_data['Predicted Wait Time'] = pipeline.predict(
            pd.DataFrame({
                'DayOfWeek': [day] * len(avg_day_data),
                'Hour_sin': np.sin(2 * np.pi * avg_day_data['Hour'] / 24),
                'Hour_cos': np.cos(2 * np.pi * avg_day_data['Hour'] / 24),
                'IsWeekend': [int(day in ['Saturday', 'Sunday'])] * len(avg_day_data),
                'PartOfDay': ['Morning' if h < 12 else 'Afternoon' if h < 17 else 'Evening' for h in avg_day_data['Hour']]
            })
        )

        plt.figure(figsize=(8, 6))
        plt.plot(avg_day_data['Hour'], avg_day_data['Wait Time (mins)'], label='Actual Wait Time', marker='o', linestyle='-', color='blue')
        plt.plot(avg_day_data['Hour'], avg_day_data['Predicted Wait Time'], label='Predicted Wait Time', marker='x', linestyle='--', color='red')
        plt.title(f'Actual vs Predicted Wait Time on {day}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Wait Time (mins)')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{day}_comparison.png')
        plt.savefig(plot_path)
        plt.close()

        print(f"Plot saved for {day} at {plot_path}")

    # Pass metrics and days to template
    return render_template(
        'our_model.html',
        train_mse=train_mse,
        test_mse=test_mse,
        train_rmse=train_rmse,
        test_rmse=test_rmse,
        train_mae=train_mae,
        test_mae=test_mae,
        days_of_week=days_of_week
    )

if __name__ == '__main__':
    app.run(debug=True)
