import torch
from sklearn.datasets import make_classification
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


if __name__ == "__main__":
    # 1. Create synthetic data
    X, y = make_classification(
        n_samples=10,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=23
    )

    print(f"Shape of X : {X.shape} and X = ")
    print(X)

    # 2. Convert the data to tensors
    X: torch.Tensor = torch.tensor(X, dtype=torch.float32)
    y: torch.Tensor = torch.tensor(y, dtype=torch.long)

    # 3. Object of Custom DataLoader
    data_set: CustomDataset = CustomDataset(X, y)

    print(f"Since, data_set is an object of CustomDataset, we can access it's length: {len(data_set)}")
    print("Accessing rows of data_set:\n")
    print(f"data_set[0]: {data_set[0]}")
    print(f"data_set[5]: {data_set[5]}")
    print(f"data_set[7]: {data_set[7]}")

    # 4. DataLoader
    data_loader: DataLoader = DataLoader(dataset=data_set, batch_size=3, shuffle=True)

    print("Extracting data using dataloader:\n")
    for batch_feat, batch_labels in data_loader:
        print(f"features: {batch_feat}, labels: {batch_labels}")
        print("-"*50)