import pandas as pd
import numpy as np
import os
from trip_data import TripData
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

selected_features = [
    'heart_rate_bpm',
    'gps_speed',
    'accelerometer_x_axis_acceleration',
    'accelerometer_y_axis_acceleration',
    'accelerometer_z_axis_acceleration',
    'gyroscope_x_axis_rotation_rate',
    'gyroscope_y_axis_rotation_rate',
    'gyroscope_z_axis_rotation_rate'
]

subjects = [
    "as",
    "av",
    "bl",
    "cv",
    "kh",
    "sg"
]

data_path = "data/"


def read_data(subjects, directory):
    data_read = []
    for subject in subjects:
        path = directory + subject
        for item in os.listdir(path):
            if os.path.isdir(os.path.join(path, item)):
                if item != "other" and not item.startswith("0") and not item.startswith("1"):
                    trip_id = item.split('_')[0]
                    data_read.append(TripData(trip_id, item, path))
    return data_read


def read_test_data(subjects, directory):
    data_read = []
    for subject in subjects:
        path = directory + subject
        for item in os.listdir(path):
            if os.path.isdir(os.path.join(path, item)):
                if item != "other" and (item.startswith("0") or item.startswith("1")):
                    trip_id = item.split('_')[0]
                    data_read.append(TripData(trip_id, item, path))
    return data_read

# Function to pad or truncate data to a consistent length
def pad_or_truncate(data, target_length):
    if len(data) > target_length:
        return data[:target_length]
    elif len(data) < target_length:
        return np.pad(data, (0, target_length - len(data)), 'constant')
    return data


data = read_data(subjects, data_path)

# Define the target length (e.g., 5 minutes at 50 Hz)
# target_length = 5 * 60 * 50
target_length = max([len(trip.get_data()["gps_speed"]) for trip in data])

# Ensure all trips have the same length of data
padded_data = []

for trip in data:
    trip_data = trip.get_data()
    padded_trip_data = {}
    for column in trip_data.columns:
        if column in selected_features:
            padded_trip_data[column] = pad_or_truncate(trip_data[column].values, target_length)
    padded_data.append(padded_trip_data)

# Convert to DataFrame
flat_data = []
for trip_data in padded_data:
    flat_trip_data = []
    for key, value in trip_data.items():
        flat_trip_data.extend(value)
    flat_data.append(flat_trip_data)

flat_data_df = pd.DataFrame(flat_data)
flat_data_df = flat_data_df.apply(pd.to_numeric, errors='coerce')

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(flat_data_df)

# Ensure the comfort scores align with the trip features
comfort_scores = [trip.get_comfort_score() for trip in data]

# Load and process the test trip data
test_data = read_test_data(subjects, data_path)
padded_test_data = []

test_comfort_scores = [trip.get_comfort_score() for trip in test_data]

for trip in test_data:
    trip_data = trip.get_data()
    padded_trip_data = {}
    for column in trip_data.columns:
        if column in selected_features:
            padded_trip_data[column] = pad_or_truncate(trip_data[column].values, target_length)
    padded_test_data.append(padded_trip_data)

# Flatten and normalize the 26th trip data
flat_test_data = []
for trip_data in padded_test_data:
    flat_trip_data = []
    for key, value in trip_data.items():
        flat_trip_data.extend(value)
    flat_test_data.append(flat_trip_data)

flat_test_data_df = pd.DataFrame(flat_test_data)
flat_test_data_df = flat_test_data_df.apply(pd.to_numeric, errors='coerce')
normalized_test_data = scaler.fit_transform(flat_test_data_df)


def evaluate_model(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validated R^2 scores for {model.__class__.__name__}: {cv_scores}")
    print(f"Mean R^2 score for {model.__class__.__name__}: {np.mean(cv_scores)}")
    model.fit(X, y)

    # Predict
    index = 0
    for test_trip in normalized_test_data:
        predicted_comfort = model.predict(test_trip.reshape(1, -1))
        print("Predicted comfort: {} actual: {}".format(predicted_comfort[0], test_comfort_scores[index]))
        index += 1


models = {
    # "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Machine": SVR()
}

if __name__ == '__main__':
    # Evaluate each model
    for name, model in models.items():
        print(f"Evaluating {name}...")
        evaluate_model(model, normalized_data, comfort_scores)
