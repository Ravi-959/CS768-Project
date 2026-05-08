import torch
from torch_geometric.utils import to_dense_adj


def laplacian_positional_encoding(
    edge_index,
    num_nodes,
    pe_dim=4
):

    A = to_dense_adj(
        edge_index,
        max_num_nodes=num_nodes
    )[0]

    D = torch.diag(A.sum(dim=1))

    L = D - A

    eigvals, eigvecs = torch.linalg.eigh(L)

    pe = eigvecs[:, 1:pe_dim+1]

    return pe


def compute_vector_field(
    edge_index,
    phi,
    num_nodes
):

    F = torch.zeros(
        (num_nodes, num_nodes),
        device=phi.device
    )

    direction = phi[:, 0]

    for i, j in edge_index.t():
        F[i, j] = direction[j] - direction[i]

    return F