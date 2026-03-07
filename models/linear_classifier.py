"""Linear latent classifier with optional equivariant operators."""

import torch
import torch.nn as nn
import numpy as np
IMG_SIZE = 28
INPUT_DIM = 3 * IMG_SIZE * IMG_SIZE
LATENT_DIM = 70


def cyclic_shift_matrix(n):
    """Return the n x n permutation matrix for a +1 cyclic shift."""
    P = np.zeros((n, n), dtype=int)
    for i in range(n):
        P[i, (i + 1) % n] = 1
    return P


class LinearClassifier(nn.Module):

    def __init__(
        self,
        block_size=None,
        learnable_P=False,
        num_classes=10,
        device="cuda",
        cycle=8,
        num_stages=1,
    ):
        super().__init__()

        self.device = device
        self.block_size = block_size
        self.learnable_P = learnable_P
        self.CYCLE = cycle
        self.num_stages = num_stages

        # -------- encoders --------
        self.encoders = nn.ModuleList()

        self.encoders.append(
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(INPUT_DIM, LATENT_DIM, bias=False),
            )
        )

        for _ in range(1, num_stages):
            self.encoders.append(
                nn.Sequential(
                    nn.Linear(LATENT_DIM, LATENT_DIM, bias=False),
                )
            )

        # -------- classifier --------
        self.classifier = nn.Sequential(
            nn.Linear(LATENT_DIM, LATENT_DIM),
            nn.Sigmoid(),
            nn.Linear(LATENT_DIM, num_classes),
        )

        if block_size is None:
            return

        # -------- operators --------
        self.ops = []
        if learnable_P:

            self.P = nn.ParameterList()

            for _ in range(num_stages):
                q, _ = torch.linalg.qr(torch.randn(block_size, block_size))
                self.P.append(nn.Parameter(q))

        else:

            base_P = torch.tensor(cyclic_shift_matrix(block_size), dtype=torch.float32)
            self.P = [base_P for _ in range(num_stages)]

            for P in self.P:
                self.ops.append(self.get_operators(P))

    def get_operators(self, P):
        """Build [P^0, ..., P^(cycle-1)] expanded to full latent dimension."""

        num_blocks = LATENT_DIM // self.block_size
        I = torch.eye(num_blocks, device=P.device)

        ops = []
        for k in range(self.CYCLE):

            Pk = torch.linalg.matrix_power(P, k)
            ops.append(torch.kron(I, Pk))

        return torch.stack(ops).to(self.device)

    def apply_transform(self, z, stage, degs, skip_transform):
        """Apply stage operator either at specific indices or for all indices."""

        if skip_transform:
            return z

        if self.learnable_P:
            ops = self.get_operators(self.P[stage])
        else:
            ops = self.ops[stage]

        if degs is not None:

            op = ops[degs]
            z = torch.bmm(op, z.unsqueeze(-1)).squeeze(-1)

        else:

            z_exp = z.unsqueeze(1).unsqueeze(-1)
            ops_exp = ops.unsqueeze(0)
            z = torch.matmul(ops_exp, z_exp).squeeze(-1)

        return z

    def _normalize_degs(self, degs, skip_transform=False):
        """Normalize degrees into a per-stage list used by encode/apply_transform."""
        if skip_transform or degs is None:
            return [None] * self.num_stages

        if isinstance(degs, (list, tuple)):
            if len(degs) != self.num_stages:
                raise ValueError(
                    f"Expected {self.num_stages} stage entries in degs, got {len(degs)}"
                )
            return list(degs)

        if self.num_stages == 1:
            return [degs]

        raise ValueError(
            "For multi-stage models, pass degs as a list/tuple with one entry per stage."
        )

    def encode(self, x, degs=None, skip_transform=False):
        """
        Encode inputs stage-by-stage.
        If degs[i] is None, stage i skips transformation.
        """

        z = x

        degs = self._normalize_degs(degs, skip_transform=skip_transform)

        for i, encoder in enumerate(self.encoders):

            z = encoder(z)

            z = self.apply_transform(
                z,
                stage=i,
                degs=None if degs[i] is None else degs[i],
                skip_transform=degs[i] is None,
            )

        return z

    def decode(self, z):

        return self.classifier(z)

    def forward(self, x, degs=None, skip_transform=False):

        z = self.encode(x, degs=degs, skip_transform=skip_transform)

        return z, self.decode(z.reshape(-1, LATENT_DIM))
