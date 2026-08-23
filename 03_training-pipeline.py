import torch
import pprint
import numpy as np
import pandas as pd
import torch.nn as nn

from typing import Tuple
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Defining the model
class MySimpleNN(nn.Module):
    def __init__(self, num_features: int):
        super(MySimpleNN, self).__init__()
        self.linear = nn.Linear(num_features, 1)  # Linear layer with input features and 1 output

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X)  # Returns raw logits; use with BCEWithLogitsLoss


if __name__ == "__main__":

    # Load the dataset
    df: pd.DataFrame = pd.read_csv("https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv")

    print("=== First 5 rows of the dataset ===")
    pprint.pprint(df.head())

    # Drop unnecessary columns
    df.drop(columns=["id", "Unnamed: 32"], inplace=True)
    print("\n=== After dropping 'id' and 'Unnamed: 32' columns ===")
    pprint.pprint(df.head())

    # Train-test split
    split_result: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] = train_test_split(
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

    # Defining the parameters
    torch.manual_seed(23)
    epochs: int = 200
    learning_rate: float = 0.01
    loss_function: nn.Module = nn.BCEWithLogitsLoss()  # Combines Sigmoid + BCE for numerical stability

    # Training pipeline
    model: MySimpleNN = MySimpleNN(X_train_tensor.shape[1])

    # Define the optimizer
    optimizer: SGD = SGD(model.parameters(), lr=learning_rate)

    print("\n=== Training the model ===")
    for epoch in range(epochs):
        # forward pass
        y_pred: torch.Tensor = model(X_train_tensor)

        # loss calculation
        loss: torch.Tensor = loss_function(y_pred, y_train_tensor)

        # clear gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update parameters
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                test_pred_class: torch.Tensor = (torch.sigmoid(model(X_test_tensor)) > 0.5).float()
                test_accuracy: torch.Tensor = (test_pred_class == y_test_tensor).float().mean()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Test Accuracy: {test_accuracy.item() * 100:.2f}%")

    # Evaluate the model
    with torch.no_grad():
        y_pred_test: torch.Tensor = torch.sigmoid(model.forward(X_test_tensor))
        y_pred_test_class: torch.Tensor = (y_pred_test > 0.5).float()

        accuracy: torch.Tensor = (y_pred_test_class == y_test_tensor).float().mean()
        print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")


