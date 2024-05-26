import os
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from TripData import TripData


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

def read_data(directory):
    data_read = []
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            if item != "other" and not item.startswith("4"):
                trip_id = item.split('_')[0]
                data_read.append(TripData(trip_id, item))
    return data_read

def read_test_data(directory):
    data_read = []
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            if item != "other":
                trip_id = item.split('_')[0]
                data_read.append(TripData(trip_id, item))
    return data_read


data = read_data("data")
test_data = read_test_data("data")

lengths = torch.tensor([len(x.get_data()) for x in data])


class DrivingDataset(Dataset):
    def __init__(self, trip_data: List[TripData]):
        self.trip_data = trip_data

    def __len__(self):
        return len(self.trip_data)

    def __getitem__(self, idx):
        trip = self.trip_data[idx]
        trip_data = trip.get_data()

        # Extract features and convert to tensor
        features = trip_data[selected_features].values
        features = torch.tensor(features, dtype=torch.float)

        # Extract comfort score and convert to tensor
        comfort_score = float(trip.get_comfort_score())
        comfort_score = torch.tensor([comfort_score], dtype=torch.float)

        return features, comfort_score

def collate_fn(batch):
    features, scores = zip(*batch)
    lengths = torch.tensor([len(f) for f in features])
    padded_features = pad_sequence(features, batch_first=True)
    scores = torch.stack(scores)
    return padded_features, scores, lengths


dataset = DrivingDataset(data)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

# 2. Prepare the test DataLoader
test_dataset = DrivingDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=1)


# Define TCN Block
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.net1 = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net2 = nn.Sequential(self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net1(x)
        out = self.net2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self, num_inputs, num_channels, output_size):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x, lengths):
        y1 = self.tcn(x)
        batch_size = y1.size(0)
        out = torch.zeros(batch_size, y1.size(1)).to(y1.device)
        for i in range(batch_size):
            out[i] = y1[i, :, lengths[i] - 1]  # Use the output at the last valid time step
        return self.linear(out)


# Model parameters
num_inputs = len(selected_features)  # Number of selected_features (speed, acceleration, gravity)
num_channels = [25, 50, 100]
output_size = 1

model = TCNModel(num_inputs, num_channels, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
# if __name__ == '__main__':
#     for epoch in range(num_epochs):
#         for batch_X, batch_y, lengths in dataloader:
#             batch_X = batch_X.permute(0, 2, 1)  # Change to (batch_size, num_features, seq_length)
#             outputs = model(batch_X, lengths)
#             loss = criterion(outputs, batch_y)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#
#     # Save the trained model
#     torch.save(model.state_dict(), 'tcn_model.pth')

if __name__ == '__main__':
    loaded_model = TCNModel(num_inputs, num_channels, output_size)

    # Load the saved state dictionary
    loaded_model.load_state_dict(torch.load('tcn_model.pth'))

    # Ensure the model is in evaluation mode if making predictions
    loaded_model.eval()

    # Now `loaded_model` can be used for inference or further training

    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            batch_X = batch_X.permute(0, 2, 1)  # Change to (batch_size, num_features, seq_length)
            outputs = loaded_model(batch_X, lengths)

            # Assuming your model outputs a single value for the comfort score prediction
            predicted_score = outputs.item()

            print("Predicted comfort score:", predicted_score)