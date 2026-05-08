import torch

from torch.utils.data import DataLoader
from torch_geometric.datasets import ZINC


def pad_matrix(mat, target_size):

    current_size = mat.size(0)

    if current_size == target_size:
        return mat

    padded = torch.zeros(
        (target_size, target_size),
        dtype=mat.dtype
    )

    padded[:current_size, :current_size] = mat

    return padded


def pad_features(x, target_size):

    current_size = x.size(0)

    if current_size == target_size:
        return x

    padded = torch.zeros(
        (target_size, x.size(1)),
        dtype=x.dtype
    )

    padded[:current_size] = x

    return padded


def collate_fn(batch):

    max_nodes = max(data.num_nodes for data in batch)

    xs = []
    Fs = []
    ys = []
    masks = []

    for data in batch:

        x = pad_features(data.x, max_nodes)

        F = pad_matrix(data.F, max_nodes)

        mask = torch.zeros(max_nodes)
        mask[:data.num_nodes] = 1

        xs.append(x)
        Fs.append(F)
        ys.append(data.y)
        masks.append(mask)

    return {
        "x": torch.stack(xs),
        "F": torch.stack(Fs),
        "y": torch.stack(ys),
        "mask": torch.stack(masks)
    }


def get_zinc(batch_size, transform=None):

    train_dataset = ZINC(
        root='data/ZINC',
        subset=True,
        split='train',
        transform=transform
    )

    val_dataset = ZINC(
        root='data/ZINC',
        subset=True,
        split='val',
        transform=transform
    )

    test_dataset = ZINC(
        root='data/ZINC',
        subset=True,
        split='test',
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader