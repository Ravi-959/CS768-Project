import torch
import torch.nn.functional as F

from tqdm import tqdm


def train_epoch(
    model,
    loader,
    optimizer,
    device
):

    model.train()

    total_loss = 0

    for batch in tqdm(loader):

        x = batch["x"].to(device)
        Fmat = batch["F"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        out = model(
            x,
            Fmat,
            mask
        )

        loss = F.l1_loss(
            out.squeeze(),
            y.squeeze()
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(
    model,
    loader,
    device
):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for batch in loader:

            x = batch["x"].to(device)
            Fmat = batch["F"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)

            out = model(
                x,
                Fmat,
                mask
            )

            loss = F.l1_loss(
                out.squeeze(),
                y.squeeze()
            )

            total_loss += loss.item()

    return total_loss / len(loader)