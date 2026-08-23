import pprint

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset


# Custom dataloader
class CustomDataset(Dataset):
    def __init__(self, feat: torch.Tensor, labels: torch.Tensor):
        self.feature = feat
        self.labels = labels

    def __len__(self):
        return self.feature.shape[0]

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.feature[index], self.labels[index]
    

# Defining the model
class MySimpleNN(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)  # Linear layer with input features and 1 output

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X)  # Returns raw logits; use with BCEWithLogitsLoss


# preprocessing steps
def preprocess_data(path: str, debug: bool = False) -> list[torch.Tensor]:
    # Load the dataset from Hugging Face
    ds: DatasetDict = load_dataset(path)
    df: pd.DataFrame = ds["train"].to_pandas()

    if debug:
        print("=== First 5 rows of the dataset ===")
        pprint.pprint(df.head())

    # Drop unnecessary columns
    df.drop(columns=["id", "Unnamed: 32"], inplace=True, errors="ignore")

    if debug:
        print("\n=== After dropping 'id' and 'Unnamed: 32' columns ===")
        pprint.pprint(df.head())

    # Train-test split
    split_result: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] = train_test_split(
        df.iloc[:, 1:], df.iloc[:, 0], test_size=0.2, random_state=23, stratify=df.iloc[:, 0]
    )
    X_train, X_test, y_train, y_test = split_result

    # Standardize the features
    scaler = StandardScaler()
    X_train: np.ndarray = scaler.fit_transform(X_train)
    X_test: np.ndarray = scaler.transform(X_test)

    # Label encode the target variable
    encoder = LabelEncoder()
    y_train: np.ndarray = encoder.fit_transform(y_train)
    y_test: np.ndarray = encoder.transform(y_test)

    # Using torch tensors
    X_train_tensor: torch.Tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor: torch.Tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor: torch.Tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor: torch.Tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return [X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor]


if __name__ == "__main__":
    data_path: str = "scikit-learn/breast-cancer-wisconsin"

    train_data: list[torch.Tensor] = preprocess_data(data_path, debug=True)
    X_train: torch.Tensor = train_data[0]
    X_test: torch.Tensor = train_data[1]
    y_train: torch.Tensor = train_data[2]
    y_test: torch.Tensor = train_data[3]

    # Define dataset and dataloader
    train_dataset: CustomDataset = CustomDataset(X_train, y_train)
    test_dataset: CustomDataset = CustomDataset(X_test, y_test)

    train_loader: DataLoader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader: DataLoader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Defining the parameters
    torch.manual_seed(23)
    epochs: int = 200
    learning_rate: float = 0.01
    loss_function: nn.Module = nn.BCEWithLogitsLoss()  # Combines Sigmoid + BCE for numerical stability

    # Training pipeline
    model: MySimpleNN = MySimpleNN(X_train.shape[1])

    # Define the optimizer
    optimizer: SGD = SGD(model.parameters(), lr=learning_rate)

    print("\n=== Training the model ===")
    for epoch in range(epochs):

        for batch_X, batch_y in train_loader:

            # forward pass
            y_pred: torch.Tensor = model(batch_X)

            # loss calculation
            loss: torch.Tensor = loss_function(y_pred, batch_y)

            # clear gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update parameters
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                test_pred_class: torch.Tensor = (torch.sigmoid(model(X_test)) > 0.5).float()
                test_accuracy: torch.Tensor = (test_pred_class == y_test).float().mean()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Test Accuracy: {test_accuracy.item() * 100:.2f}%")

    # Evaluate the model
    with torch.no_grad():
        correct: int = 0
        total: int = 0
        for batch_X, batch_y in test_loader:

            y_pred_test: torch.Tensor = torch.sigmoid(model.forward(batch_X))
            y_pred_test_class: torch.Tensor = (y_pred_test > 0.5).float()

            correct += (y_pred_test_class == batch_y).float().sum().item()
            total += batch_y.size(0)

        accuracy: float = correct / total
        print(f"Test Accuracy: {accuracy * 100:.2f}%")


