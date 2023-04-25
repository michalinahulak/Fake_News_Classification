import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna


def create_logistic_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    # define a PyTorch dataset for the data
    class MyDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # define the logistic regression model
    class LogisticRegression(nn.Module):
        def __init__(self, input_size):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_size, 1)

        def forward(self, x):
            x = self.linear(x)
            x = torch.sigmoid(x)
            return x

    # define the objective function for Optuna
    def objective(trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        epochs = trial.suggest_int('epochs', 10, 50)

        dataset_train = MyDataset(X_train, y_train)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataset_val = MyDataset(X_val, y_val)
        dataloader_val = DataLoader(dataset_val, batch_size=len(dataset_val))

        model = LogisticRegression(X_train.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            for inputs, labels in dataloader_train:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

        # evaluate the model on the validation set
        with torch.no_grad():
            for inputs, labels in dataloader_val:
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                accuracy = ((outputs >= 0.5).float() == labels.unsqueeze(1)).float().mean()

        return accuracy.item()

    # run the hyperparameter search with Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    # get the best trial and print the results
    best_trial = study.best_trial
    print('Best trial:')
    print(f'  Loss: {best_trial.value:.4f}')
    print('  Params: ')
    for key, value in best_trial.params.items():
        print(f'    {key}: {value}')

    # train the model on the full training set with the best hyperparameters
    best_lr = best_trial.params['lr']
    best_batch_size = best_trial.params['batch_size']
    best_epochs = best_trial.params['epochs']

    # create PyTorch DataLoader objects for the training, validation, and test datasets
    dataset_train = MyDataset(X_train, y_train)
    dataloader_train = DataLoader(dataset_train, batch_size=best_batch_size, shuffle=True)
    dataset_val = MyDataset(X_val, y_val)
    dataloader_val = DataLoader(dataset_val, batch_size=len(dataset_val))
    dataset_test = MyDataset(X_test, y_test)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))

    model = LogisticRegression(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=best_lr)
    criterion = nn.BCELoss()

    for epoch in range(best_epochs):
        for inputs, labels in dataloader_train:
            optimizer.zero_grad
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

    # evaluate the model on the training set
    with torch.no_grad():
        for inputs, labels in dataloader_train:
            outputs = model(inputs)
            train_loss = criterion(outputs, labels.unsqueeze(1))
            train_accuracy = ((outputs >= 0.5).float() == labels.unsqueeze(1)).float().mean()

    # evaluate the model on the validation set
    with torch.no_grad():
        for inputs, labels in dataloader_val:
            outputs = model(inputs)
            val_loss = criterion(outputs, labels.unsqueeze(1))
            val_accuracy = ((outputs >= 0.5).float() == labels.unsqueeze(1)).float().mean()

    # evaluate the model on the test set
    with torch.no_grad():
        for inputs, labels in dataloader_test:
            outputs = model(inputs)
            test_loss = criterion(outputs, labels.unsqueeze(1))
            test_accuracy = ((outputs >= 0.5).float() == labels.unsqueeze(1)).float().mean()

    # print the metrics for the three datasets
    print(f'Training set: loss={train_loss:.4f}, accuracy={train_accuracy:.4f}')
    print(f'Validation set: loss={val_loss:.4f}, accuracy={val_accuracy:.4f}')
    print(f'Test set: loss={test_loss:.4f}, accuracy={test_accuracy:.4f}')


def perform_kfold_cross_validation(X, y, X_test, y_test, n_splits=5, batch_size=16, learning_rate=1e-3, num_epochs=10):
    class MyDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class LogisticRegression(nn.Module):
        def __init__(self, input_size):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_size, 1)

        def forward(self, x):
            x = self.linear(x)
            x = torch.sigmoid(x)
            return x

    def objective(model, criterion, optimizer, dataloader):
        model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                accuracy = ((outputs >= 0.5).float() == labels.unsqueeze(1)).float().mean()

        return accuracy.item()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_accuracies = []
    val_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        dataset_train = MyDataset(X_train, y_train)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataset_val = MyDataset(X_val, y_val)
        dataloader_val = DataLoader(dataset_val, batch_size=len(dataset_val))

        model = LogisticRegression(X_train.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        for epoch in range(num_epochs):
            objective(model, criterion, optimizer, dataloader_train)

        train_accuracy = objective(model, criterion, optimizer, dataloader_train)
        val_accuracy = objective(model, criterion, optimizer, dataloader_val)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f'Fold {fold + 1}: Train accuracy = {train_accuracy:.4f}, Validation accuracy = {val_accuracy:.4f}')

    # Train final model on all data
    dataset_all = MyDataset(X, y)
    dataloader_all = DataLoader(dataset_all, batch_size=batch_size, shuffle=True)
    model = LogisticRegression(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    for epoch in range(num_epochs):
        objective(model, criterion, optimizer, dataloader_all)

    # Test model on test data
    dataset_test = MyDataset(X_test, y_test)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))
    test_accuracy = objective(model, criterion, optimizer, dataloader_test)

    print(f'Test accuracy = {test_accuracy:.4f}')
    print(f'Average train accuracy = {np.mean(train_accuracies):.4f}')
    print(f'Average validation accuracy = {np.mean(val_accuracies):.4f}')


def create_more_complex_nn(X, y, X_test, y_test, n_splits=5, batch_size=16, learning_rate=1e-3, num_epochs=10):
    class MyDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class NeuralNetwork(nn.Module):
        def __init__(self, input_size):
            super(NeuralNetwork, self).__init__()
            self.layer1 = nn.Linear(input_size, 16)
            self.layer2 = nn.Linear(16, 8)
            self.layer3 = nn.Linear(8, 4)
            self.layer4 = nn.Linear(4, 1)
            self.activation = nn.ReLU()

        def forward(self, x):
            x = self.layer1(x)
            x = self.activation(x)
            x = self.layer2(x)
            x = self.activation(x)
            x = self.layer3(x)
            x = self.activation(x)
            x = self.layer4(x)
            x = torch.sigmoid(x)
            return x

    def objective(model, criterion, optimizer, dataloader):
        model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            accuracy = 0
            count = 0
            for inputs, labels in dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                accuracy += ((outputs >= 0.5).float() == labels.unsqueeze(1)).float().mean()
                count += 1

        return accuracy.item() / count

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_accuracies = []
    val_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        dataset_train = MyDataset(X_train, y_train)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataset_val = MyDataset(X_val, y_val)
        dataloader_val = DataLoader(dataset_val, batch_size=len(dataset_val))

        model = NeuralNetwork(X_train.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        for epoch in range(num_epochs):
            objective(model, criterion, optimizer, dataloader_train)

        train_accuracy = objective(model, criterion, optimizer, dataloader_train)
        val_accuracy = objective(model, criterion, optimizer, dataloader_val)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f'Fold {fold + 1}: Train accuracy = {train_accuracy:.4f}, Validation accuracy = {val_accuracy:.4f}')

    # Train final model on all data
    dataset_all = MyDataset(X, y)
    dataloader_all = DataLoader(dataset_all, batch_size=batch_size, shuffle=True)
    model = NeuralNetwork(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    for epoch in range(num_epochs):
        objective(model, criterion, optimizer, dataloader_all)

    # Test model on test data
    dataset_test = MyDataset(X_test, y_test)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))
    test_accuracy = objective(model, criterion, optimizer, dataloader_test)

    print(f'Test accuracy = {test_accuracy:.4f}')
    print(f'Average train accuracy = {np.mean(train_accuracies):.4f}')
    print(f'Average validation accuracy = {np.mean(val_accuracies):.4f}')
