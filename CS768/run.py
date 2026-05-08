import torch
import yaml

from data.zinc import get_zinc

from core.positional_encoding import (
    laplacian_positional_encoding,
    compute_vector_field
)

from core.dgn_model import DGN

from train.train import (
    train_epoch,
    eval_epoch
)


# =========================
# Load config
# =========================
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)


# =========================
# Safe config loading
# =========================
hidden_dim = cfg.get("model", {}).get("hidden_dim", 128)

layers = cfg.get("model", {}).get("layers", 5)

dropout = cfg.get("model", {}).get("dropout", 0.1)

pe_dim = cfg.get("model", {}).get("pe_dim", 4)

lr = cfg.get("training", {}).get("lr", 0.0005)

epochs = cfg.get("training", {}).get("epochs", 50)

batch_size = cfg.get("training", {}).get("batch_size", 32)


# =========================
# Device
# =========================
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# =========================
# Graph preprocessing
# =========================
def transform(data):

    phi = laplacian_positional_encoding(
        data.edge_index,
        data.num_nodes,
        pe_dim
    )

    Fmat = compute_vector_field(
        data.edge_index,
        phi,
        data.num_nodes
    )

    data.F = Fmat

    return data


# =========================
# Load datasets
# =========================
train_loader, val_loader, test_loader = get_zinc(
    batch_size=batch_size,
    transform=transform
)


# =========================
# Build model
# =========================
model = DGN(
    hidden_dim=hidden_dim,
    num_layers=layers,
    dropout=dropout
).to(device)


# =========================
# Optimizer
# =========================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=1e-5
)


# =========================
# Scheduler
# =========================
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
)


# =========================
# Training loop
# =========================
best_val = float("inf")

for epoch in range(epochs):

    train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        device
    )

    val_loss = eval_epoch(
        model,
        val_loader,
        device
    )

    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val:

        best_val = val_loss

        torch.save(
            model.state_dict(),
            "best_model.pt"
        )

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train MAE: {train_loss:.4f} | "
        f"Val MAE: {val_loss:.4f}"
    )


# =========================
# Load best model
# =========================
print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        "best_model.pt",
        map_location=device
    )
)


# =========================
# Final test evaluation
# =========================
test_loss = eval_epoch(
    model,
    test_loader,
    device
)

print("\n=========================")
print(f"Final Test MAE: {test_loss:.4f}")
print("=========================")