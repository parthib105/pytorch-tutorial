import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Set to True to print status messages describing what the script is doing.
DEBUG: bool = True

# Fashion-MNIST class index -> label name
CLASS_NAMES: list[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# set device
gpu_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {gpu_device}")


# Custom dataloader
class CustomDataset(Dataset):
    def __init__(self, feat: torch.Tensor, labels: torch.Tensor) -> None:
        self.feature: torch.Tensor = feat
        self.labels: torch.Tensor = labels

    def __len__(self) -> int:
        return self.feature.shape[0]

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.feature[index], self.labels[index]


# Defining the model
class MyNN(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, num_hidden_layers: int, neurons_per_hidden: int, dropout_rate: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        for i in range(num_hidden_layers):
            layers.append(nn.Linear(inp_dim, neurons_per_hidden))
            layers.append(nn.BatchNorm1d(neurons_per_hidden))  # Add batch normalization layer
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))  # Add dropout layer with suggested dropout rate

            inp_dim = neurons_per_hidden  # Update input dimension for the next layer

        # Add the output layer
        layers.append(nn.Linear(neurons_per_hidden, out_dim))

        self.model: nn.Sequential = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)


# Function to load data from hugging face or local file
def load_data(local_path: str, debug: bool = DEBUG) -> list[torch.Tensor]:
    if os.path.exists(local_path):
        if debug:
            print(f"[DEBUG] Fashion-MNIST found locally at '{local_path}'. Loading from disk...")
        # Load directly from your local folder
        ds: DatasetDict = load_from_disk(local_path)
    else:
        if debug:
            print("[DEBUG] Fashion-MNIST not found locally. Downloading from Hugging Face...")
        # Download the dataset using its standard Hugging Face identifier
        ds: DatasetDict = load_dataset("zalando-datasets/fashion_mnist")

        # Save the entire dataset locally for offline use
        if debug:
            print(f"[DEBUG] Saving dataset to '{local_path}'...")
        ds.save_to_disk(local_path)

    if debug:
        print("[DEBUG] Extracting images and labels from the train/test splits...")

    # each split has an 'image' column (PIL Image) and a 'label' column (int)
    train_images: list = ds["train"]["image"]
    train_labels: list = ds["train"]["label"]
    test_images: list = ds["test"]["image"]
    test_labels: list = ds["test"]["label"]

    if debug:
        print("[DEBUG] Flattening images and normalizing pixel values into tensors...")

    X_train: torch.Tensor = torch.tensor(
        np.stack([np.array(img) for img in train_images]).reshape(len(train_images), -1) / 255.0,
        dtype=torch.float32,
    )
    y_train: torch.Tensor = torch.tensor(train_labels, dtype=torch.long)
    X_test: torch.Tensor = torch.tensor(
        np.stack([np.array(img) for img in test_images]).reshape(len(test_images), -1) / 255.0,
        dtype=torch.float32,
    )
    y_test: torch.Tensor = torch.tensor(test_labels, dtype=torch.long)

    if debug:
        print(
            f"[DEBUG] Shapes -> X_train: {tuple(X_train.shape)}, y_train: {tuple(y_train.shape)}, "
            f"X_test: {tuple(X_test.shape)}, y_test: {tuple(y_test.shape)}"
        )

    return [X_train, y_train, X_test, y_test]


# Build the dataloaders
def build_dataloaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader]:
    """Wrap the train/test tensors in `CustomDataset` and `DataLoader` instances."""
    train_dataset: CustomDataset = CustomDataset(X_train, y_train)
    test_dataset: CustomDataset = CustomDataset(X_test, y_test)

    train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Train the model
def train_model(model: MyNN, train_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
    """Train `model` on `train_loader` for 1 epoch. Returns the average training loss."""
    model.train()

    total_epoch_loss: float = 0.0
    for X_batch, y_batch in train_loader:
        # move to gpu
        X_batch: torch.Tensor = X_batch.to(gpu_device)
        y_batch: torch.Tensor = y_batch.to(gpu_device)

        # forward pass
        y_pred: torch.Tensor = model.forward(X_batch)

        # loss calculation
        loss: torch.Tensor = loss_fn(y_pred, y_batch)

        # clear gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update parameters
        optimizer.step()

        total_epoch_loss += loss.item()

    avg_train_loss: float = total_epoch_loss / len(train_loader)
    return avg_train_loss


# Evaluate the model
def evaluate_model(model: MyNN, test_loader: DataLoader) -> tuple[float, float, list[int], list[int]]:
    """Evaluate `model` accuracy and loss on `test_loader`. Also returns the true/predicted labels."""
    model.eval()

    total: int = 0
    correct: int = 0
    total_loss: float = 0.0
    loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    y_true: list[int] = []
    y_pred_all: list[int] = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # move to gpu
            X_batch: torch.Tensor = X_batch.to(gpu_device)
            y_batch: torch.Tensor = y_batch.to(gpu_device)

            out: torch.Tensor = model(X_batch)
            y_pred: torch.Tensor
            _, y_pred = torch.max(out, 1)

            total += y_batch.shape[0]
            correct += (y_pred == y_batch).sum().item()
            total_loss += loss_fn(out, y_batch).item()

            y_true.extend(y_batch.cpu().tolist())
            y_pred_all.extend(y_pred.cpu().tolist())

    accuracy: float = correct / total
    avg_loss: float = total_loss / len(test_loader)

    return accuracy, avg_loss, y_true, y_pred_all


# optuna objective function
def objective(
    trial: optuna.Trial,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> float:
    # Dimensions
    inp_dim: int = X_train.shape[1]
    out_dim: int = len(CLASS_NAMES)

    # Hyperparameter search space
    num_hidden_layers: int = trial.suggest_int("num_hidden_layers", 2, 8)
    neurons_per_hidden: int = trial.suggest_int("neurons_per_hidden", 32, 256)
    epochs: int = trial.suggest_int("epochs", 10, 60, step=10)
    learning_rate: float = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    dropout_rate: float = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    batch_size: int = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    optimizer_name: str = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSprop", "AdamW"])

    # Instantiate the model, loss function, and optimizer
    myModel: MyNN = MyNN(inp_dim, out_dim, num_hidden_layers, neurons_per_hidden, dropout_rate).to(gpu_device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer_cls = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_cls(myModel.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Define dataset and dataloader
    train_loader: DataLoader
    test_loader: DataLoader
    train_loader, test_loader = build_dataloaders(X_train, y_train, X_test, y_test, batch_size=batch_size)

    # Training loop
    for epoch in range(epochs):
        train_loss: float = train_model(myModel, train_loader, optimizer, loss_fn)

        # Evaluate on test set
        accuracy, avg_test_loss, _, _ = evaluate_model(myModel, test_loader)

        if DEBUG and (epoch + 1) % 50 == 0:
            print(f"[Epoch] {epoch + 1}/{epochs} => Train Loss: {train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {accuracy:.4f}")

    return accuracy  # Return accuracy as the objective to maximize


# to plot first few sample images
def plot_sample_images(
    images: torch.Tensor, labels: torch.Tensor, n: int = 10, plots_dir: str = "./plots"
) -> None:
    """Plot the first `n` samples as 28x28 grayscale images with their labels and save to `plots_dir`."""
    print(f"[DEBUG] Plotting first {n} images (28x28)...")

    params: tuple[Figure, Any] = plt.subplots(1, n, figsize=(n * 1.2, 1.5))
    axes: list[Axes] = params[1]
    for i, ax in enumerate(axes):
        image_28x28: torch.Tensor = images[i].reshape(28, 28)
        ax.imshow(image_28x28, cmap="gray")
        ax.set_title(str(labels[i].item()), fontsize=8)
        ax.axis("off")

    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    save_path: str = os.path.join(plots_dir, "sample_images.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[DEBUG] Saved sample images plot to '{save_path}'.")


# Plot the train loss curve alongside the test loss curve
def plot_loss(train_losses: list[float], test_losses: list[float], plots_dir: str = "./plots") -> None:
    """Plot and save the per-epoch training and test losses."""
    print("[DEBUG] Plotting loss curve...")

    epochs_range: range = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs_range, train_losses, label="Train Loss", color="#1f77b4")
    ax.plot(epochs_range, test_losses, label="Test Loss", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over Epochs")
    ax.legend()
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    save_path: str = os.path.join(plots_dir, "loss.png")
    plt.savefig(save_path)
    plt.close(fig)

    print(f"[DEBUG] Saved loss plot to '{save_path}'.")


# Plot the confusion matrix
def plot_confusion_matrix(
    y_true: list[int], y_pred: list[int], class_names: list[str] = CLASS_NAMES, plots_dir: str = "./plots"
) -> None:
    """Plot and save a confusion matrix heatmap for the given true/predicted labels."""
    print("[DEBUG] Plotting confusion matrix...")

    num_classes: int = len(class_names)
    y_true_arr: np.ndarray = np.array(y_true)
    y_pred_arr: np.ndarray = np.array(y_pred)
    cm: np.ndarray = np.bincount(
        num_classes * y_true_arr + y_pred_arr, minlength=num_classes**2
    ).reshape(num_classes, num_classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    save_path: str = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close(fig)

    print(f"[DEBUG] Saved confusion matrix plot to '{save_path}'.")


if __name__ == "__main__":
    # set debug
    DEBUG = True

    # Get the data
    local_path: str = "./fashion_mnist"
    train_test: list[torch.Tensor] = load_data(local_path, debug=DEBUG)

    X_train: torch.Tensor = train_test[0]
    y_train: torch.Tensor = train_test[1]
    X_test: torch.Tensor = train_test[2]
    y_test: torch.Tensor = train_test[3]

    # Visualize the first 10 images
    if DEBUG:
        plot_sample_images(X_train, y_train, n=10)

    # Defining the parameters
    torch.manual_seed(23)

    # Create an Optuna study to maximize accuracy
    study: optuna.Study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            X_test,
            y_test,
        ),
        n_trials=15,
    )

    # print the best hyperparameters
    print("Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    # Plot training/evaluation metrics
    # plot_loss(train_losses, test_losses)
    # plot_confusion_matrix(y_true, y_pred)

"""
Best hyperparameters found:
    num_hidden_layers: 7
    neurons_per_hidden: 137
    epochs: 40
    learning_rate: 0.00010788556585223089
    dropout_rate: 0.1
    batch_size: 128
    optimizer: RMSprop
"""
