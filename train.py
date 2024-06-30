import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from trip_data import TripData
from tcn_model import TCNModel
from driving_dataset import DrivingDataset
from driving_dataset import collate_fn

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
model_file_name = 'models/tcn_model_{}.pth'.format("".join(subjects))


def read_data(subjects, directory):
    data_read = []
    for subject in subjects:
        path = directory + subject
        for item in os.listdir(path):
            if os.path.isdir(os.path.join(path, item)):
                if item != "other" and not item.startswith("0"):
                    trip_id = item.split('_')[0]
                    data_read.append(TripData(trip_id, item, path))
    return data_read


def read_test_data(subjects, directory):
    data_read = []
    for subject in subjects:
        path = directory + subject
        for item in os.listdir(path):
            if os.path.isdir(os.path.join(path, item)):
                if item != "other":
                    trip_id = item.split('_')[0]
                    data_read.append(TripData(trip_id, item, path))
    return data_read


# Configuration
num_inputs = len(selected_features)  # Number of selected_features (speed, acceleration, gravity)
num_channels = [25, 50, 100]
output_size = 1


def train():
    data = read_data(subjects, data_path)
    dataset = DrivingDataset(data, selected_features)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

    num_epochs = 50
    model = TCNModel(num_inputs, num_channels, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch_X, batch_y, lengths in dataloader:
            batch_X = batch_X.permute(0, 2, 1)  # Change to (batch_size, num_features, seq_length)
            outputs = model(batch_X, lengths)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(model.state_dict(), model_file_name)


def test():
    test_data = read_test_data(subjects, data_path)
    test_dataset = DrivingDataset(test_data, selected_features)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    loaded_model = TCNModel(num_inputs, num_channels, output_size)

    # Load the saved state dictionary
    loaded_model.load_state_dict(torch.load(model_file_name))

    # Ensure the model is in evaluation mode if making predictions
    loaded_model.eval()

    # Now `loaded_model` can be used for inference or further training

    with torch.no_grad():
        index = 0
        predicted_scores = []
        actual_scores = []
        for batch_X, batch_y, lengths in test_dataloader:
            batch_X = batch_X.permute(0, 2, 1)  # Change to (batch_size, num_features, seq_length)
            outputs = loaded_model(batch_X, lengths)

            # Assuming your model outputs a single value for the comfort score prediction
            predicted_scores.append(outputs.item())

            actual_scores.append(test_data[index].get_comfort_score())
            index += 1
        index = 0
        for predicted in predicted_scores:
            print("Predicted comfort score: {} - actual score {}".format(predicted, actual_scores[index]))
            index += 1
        # calculate different evaluation metrics
        #MAE
        predicted = np.array(predicted_scores).astype(np.float64)
        actual = np.array(actual_scores).astype(np.float64)
        mae = np.mean(np.abs(predicted - actual))
        # MSE
        mse = np.mean((predicted - actual) ** 2)
        # RMSE
        rmse = np.sqrt(mse)
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        # MAPE
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        print("MAE: {}, MSE: {}, RMSE: {}, R-SQUARED: {}, MAPE: {}".format(mae, mse, rmse, r_squared, mape))



if __name__ == '__main__':
    train()
    test()
