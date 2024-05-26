import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Example data with variable lengths
data = [torch.randn(6000, 10), torch.randn(5000, 10), torch.randn(10000, 10)]  # 3 samples with different lengths
labels = torch.tensor([50, 70, 90], dtype=torch.float).view(-1, 1)

# Pad sequences to the same length
padded_data = pad_sequence(data, batch_first=True)
lengths = torch.tensor([x.size(0) for x in data])  # Actual lengths before padding


class DrivingDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    data, labels = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in data])
    padded_data = pad_sequence(data, batch_first=True)
    labels = torch.stack(labels)
    return padded_data, labels, lengths


dataset = DrivingDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)


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
        # Use lengths to handle variable-length sequences
        batch_size = y1.size(0)
        out = torch.zeros(batch_size, y1.size(1)).to(y1.device)
        for i in range(batch_size):
            out[i] = y1[i, :, lengths[i] - 1]  # Use the output at the last valid time step
        return self.linear(out)

# Define model parameters
num_inputs = 10
num_channels = [25, 50, 100]
output_size = 1

model = TCNModel(num_inputs, num_channels, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
if __name__ == '__main__':
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
    torch.save(model.state_dict(), 'tcn_model.pth')

    # Later or in a different script, load the model
    # Define the same model architecture
    loaded_model = TCNModel(num_inputs, num_channels, output_size)

    # Load the saved state dictionary
    loaded_model.load_state_dict(torch.load('tcn_model.pth'))

    # Ensure the model is in evaluation mode if making predictions
    loaded_model.eval()

    # Now `loaded_model` can be used for inference or further training


