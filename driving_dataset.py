from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from trip_data import TripData
from typing import List
import torch


class DrivingDataset(Dataset):
    def __init__(self, trip_data: List[TripData], selected_features):
        self.trip_data = trip_data
        self.selected_features = selected_features

    def __len__(self):
        return len(self.trip_data)

    def __getitem__(self, idx):
        trip = self.trip_data[idx]
        trip_data = trip.get_data()

        # Extract features and convert to tensor
        features = trip_data[self.selected_features].values
        features = torch.tensor(features, dtype=torch.float)

        # Extract comfort score and convert to tensor
        comfort_score = float(trip.get_comfort_score())
        comfort_score = torch.tensor([comfort_score], dtype=torch.float)

        return features, comfort_score


def collate_fn(batch):
    features, scores = zip(*batch)
    collocate_lengths = torch.tensor([len(f) for f in features])
    padded_features = pad_sequence(features, batch_first=True)
    scores = torch.stack(scores)
    return padded_features, scores, collocate_lengths