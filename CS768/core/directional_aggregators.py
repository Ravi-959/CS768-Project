import torch


def normalize(F, eps=1e-6):

    norm = torch.sum(
        torch.abs(F),
        dim=-1,
        keepdim=True
    ) + eps

    return F / norm


def bav(F):

    return normalize(torch.abs(F))


def bdx(F):

    F_hat = normalize(F)

    col_sum = torch.sum(
        F_hat,
        dim=-2
    )

    diag = torch.diag_embed(col_sum)

    return F_hat - diag