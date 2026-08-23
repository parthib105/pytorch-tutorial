import torch


# binary cross entropy loss
def binary_cross_entropy_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute the binary cross entropy loss."""
    epsilon = 1e-8  # small value to avoid log(0)
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)  # clamp predictions
    loss = -(
        y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
    )  # compute BCE loss
    return loss.mean()  # return mean loss

# Inputs
x = torch.tensor(6.7)
y = torch.tensor(0.0)

w = torch.tensor(1.0)
b = torch.tensor(0.0)

# forward pass
z = w * x + b
y_pred = torch.sigmoid(z)

# compute loss
loss = binary_cross_entropy_loss(y_pred, y)

# back propagation
# 1. dL/dy_pred: Loss with respect to prediction (y_pred)
dL_dy_pred = (y_pred - y) / (y_pred * (1 - y_pred))

# 2. dy_pred/dz: prediction with respect to linear output (z)
dy_pred_dz = y_pred * (1 - y_pred)

# 3. dz/dw and dz.db: linear output with respect to weights (w) and bias (b)
dz_dw = x
dz_db = 1.0

dL_dw = dL_dy_pred * dy_pred_dz * dz_dw
dL_db = dL_dy_pred * dy_pred_dz * dz_db

print("Manual Backpropagation Gradients:")
print(f"dL_dw: {dL_dw}")
print(f"dL_db: {dL_db}")


# using auto grad
x = torch.tensor(6.7)
y = torch.tensor(0.0)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# forward pass
z = w * x + b
y_pred = torch.sigmoid(z)

# compute loss
loss = binary_cross_entropy_loss(y_pred, y)

loss.backward()  # compute gradients

print("\nAutograd Gradients:")
print(f"dL_dw: {w.grad}")
print(f"dL_db: {b.grad}")
