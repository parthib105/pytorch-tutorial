import torch
from torch import nn
from torchinfo import summary


# Defining a simple neural network model
class Model(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 3),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(3, 1),  # Linear layer
            nn.Sigmoid()  # Sigmoid activation
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        out = self.network(feat)  # Pass input through the network
        return out


if __name__ == "__main__":
    # Example usage
    features = torch.rand(10, 5)  # Random input tensor with 10 samples and 5 features
    num_features = features.shape[1]
    model = Model(num_features)

    # model.forward(features)  # Forward pass through the model
    model(features)  # Forward pass through the model

    summary(model, input_size=(10, 5)) # Print model summary