import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

class FullyConnectedClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=28*28, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)


    def forward(self, x):
        fc1 = F.relu(self.fc1(x))
        fc2 = F.relu(self.fc2(fc1))
        fc3 = F.sigmoid(self.fc3(fc2))
        return fc3


def train(model, train_data, nb_epochs, batch_size, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{device = }")
    model = model.to(device)

    loss_values = []
    epoch_values = []

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)

    for epoch in range(nb_epochs):
        running_loss = 0.
        for batch in data_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)
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
