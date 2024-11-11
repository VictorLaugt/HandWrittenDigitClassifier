import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader


class ConvolutionalClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # convolutional layer parameters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # flattening
        self.flatten = nn.Flatten(start_dim=1)

        # fully connected layer parameters
        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = F.relu(self.conv2(pool1))
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        flat = self.flatten(pool2)

        fc1 = F.relu(self.fc1(flat))
        fc2 = F.relu(self.fc2(fc1))
        fc3 = self.fc3(fc2)

        return fc3

    def predict(self, x):
        return F.softmax(self.forward(x), dim=1).argmax(dim=1)

    def train(self, device, train_data, nb_epochs, batch_size, learning_rate):
        print(f"Training on device: {device}")

        loss_values = []
        epoch_values = []

        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(self.parameters(), lr=learning_rate)

        for epoch in range(nb_epochs):
            running_loss = 0.
            for batch in data_loader:
                x = batch[0].to(device)
                y = batch[1].to(device)

                optimizer.zero_grad()

                y_hat = self.forward(x)
                loss = criterion(y_hat, y)
                loss.backward()

                optimizer.step()

                with torch.no_grad():
                    running_loss += loss.item() * x.size(0)

            epoch_loss = running_loss / len(train_data)
            epoch_values.append(epoch)
            loss_values.append(epoch_loss)
            if epoch % 10 == 0:
                print(f'[epoch {epoch}] epoch loss = {epoch_loss:.4f}')

        return epoch_values, loss_values
