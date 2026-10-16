"""Architectures with a common interface: (B, C, ny, nx) -> density logits.

Graph and Fourier implementations deliberately use standard PyTorch so users
can reproduce CPU tests without specialized CUDA scatter/FFT extensions.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(inp, out):
    return nn.Sequential(nn.Conv2d(inp, out, 3, padding=1), nn.GroupNorm(4, out),
                         nn.SiLU(), nn.Conv2d(out, out, 3, padding=1),
                         nn.GroupNorm(4, out), nn.SiLU())


class ResidualBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.net = conv_block(inp, out)
        self.skip = nn.Conv2d(inp, out, 1) if inp != out else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.net(x)


class ModernUNet(nn.Module):
    def __init__(self, in_channels=10, width=24, out_channels=1):
        super().__init__()
        self.e1 = ResidualBlock(in_channels, width)
        self.e2 = ResidualBlock(width, 2 * width)
        self.mid = ResidualBlock(2 * width, 4 * width)
        self.u2 = ResidualBlock(6 * width, 2 * width)
        self.u1 = ResidualBlock(3 * width, width)
        self.out = nn.Conv2d(width, out_channels, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        z = self.mid(F.avg_pool2d(b, 2))
        z = self.u2(torch.cat([F.interpolate(z, size=b.shape[-2:], mode="bilinear",
                                             align_corners=False), b], dim=1))
        z = self.u1(torch.cat([F.interpolate(z, size=a.shape[-2:], mode="bilinear",
                                             align_corners=False), a], dim=1))
        return self.out(z)


class SpectralConv2d(nn.Module):
    """Truncated real-to-complex Fourier kernel, independent of input grid size."""
    def __init__(self, inp, out, modes_y=6, modes_x=8):
        super().__init__()
        self.out = out
        self.modes_y = modes_y
        self.modes_x = modes_x
        scale = 1 / math.sqrt(inp * out)
        self.top = nn.Parameter(scale * torch.randn(inp, out, modes_y, modes_x, dtype=torch.cfloat))
        self.bottom = nn.Parameter(scale * torch.randn(inp, out, modes_y, modes_x, dtype=torch.cfloat))

    def forward(self, x):
        b, _, ny, nx = x.shape
        f = torch.fft.rfft2(x, norm="ortho")
        output = torch.zeros(b, self.out, ny, nx // 2 + 1, device=x.device, dtype=f.dtype)
        my = min(self.modes_y, ny // 2)
        mx = min(self.modes_x, nx // 2 + 1)
        if my and mx:
            output[:, :, :my, :mx] = torch.einsum("bixy,ioxy->boxy",
                                                  f[:, :, :my, :mx], self.top[:, :, :my, :mx])
            output[:, :, -my:, :mx] = torch.einsum("bixy,ioxy->boxy",
                                                   f[:, :, -my:, :mx], self.bottom[:, :, :my, :mx])
        return torch.fft.irfft2(output, s=(ny, nx), norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, in_channels=10, width=32, depth=4, out_channels=1):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width) for _ in range(depth)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.head = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(),
                                  nn.Conv2d(width, out_channels, 1))

    def forward(self, x):
        z = self.lift(x)
        for spectral, local in zip(self.spectral, self.local):
            z = F.gelu(spectral(z) + local(z))
        return self.head(z)


class GraphLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2 * width + 2, width), nn.SiLU(),
                                     nn.Linear(width, width))
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(),
                                    nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, state, edges, delta):
        batch, nodes, width = state.shape
        src, dst = edges
        sender, receiver = state[:, src], state[:, dst]
        msg = self.message(torch.cat([sender, receiver,
                                      delta.unsqueeze(0).expand(batch, -1, -1)], dim=-1))
        agg = torch.zeros_like(state)
        agg.index_add_(1, dst, msg)
        degree = torch.zeros(nodes, device=state.device, dtype=state.dtype)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=state.dtype))
        agg = agg / degree.clamp_min(1).view(1, -1, 1)
        return self.norm(state + self.update(torch.cat((state, agg), dim=-1)))


def grid_edges(ny, nx, device, dtype):
    ids = torch.arange(nx * ny, device=device).reshape(ny, nx)
    pairs = []
    if nx > 1:
        pairs.extend([(ids[:, :-1].reshape(-1), ids[:, 1:].reshape(-1)),
                      (ids[:, 1:].reshape(-1), ids[:, :-1].reshape(-1))])
    if ny > 1:
        pairs.extend([(ids[:-1, :].reshape(-1), ids[1:, :].reshape(-1)),
                      (ids[1:, :].reshape(-1), ids[:-1, :].reshape(-1))])
    src = torch.cat([a for a, _ in pairs])
    dst = torch.cat([b for _, b in pairs])
    yy, xx = torch.meshgrid(torch.linspace(0, 1, ny, device=device, dtype=dtype),
                            torch.linspace(0, 1, nx, device=device, dtype=dtype), indexing="ij")
    xy = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    return (src, dst), xy[dst] - xy[src]


class MeshGNN(nn.Module):
    """Message passing on the actual Q4 element adjacency graph."""
    def __init__(self, in_channels=10, width=48, depth=5, out_channels=1):
        super().__init__()
        self.embed = nn.Linear(in_channels, width)
        self.layers = nn.ModuleList([GraphLayer(width) for _ in range(depth)])
        self.readout = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                     nn.Linear(width, out_channels))

    def forward(self, x):
        b, channels, ny, nx = x.shape
        edges, delta = grid_edges(ny, nx, x.device, x.dtype)
        h = self.embed(x.permute(0, 2, 3, 1).reshape(b, ny * nx, channels))
        for layer in self.layers:
            h = layer(h, edges, delta)
        return self.readout(h).transpose(1, 2).reshape(b, -1, ny, nx)


class LearnedUpdate(nn.Module):
    """Predict a bounded density correction from early iterate + sensitivities."""
    def __init__(self, in_channels=12, width=24):
        super().__init__()
        self.net = ModernUNet(in_channels, width, 1)

    def forward(self, x):
        # x first channel is the early OC iterate, further channels include FEM information.
        correction = 0.2 * torch.tanh(self.net(x))
        return torch.clamp(x[:, :1] + correction, min=0.001, max=1)


class NeuralFEM(nn.Module):
    """Seven nodewise input channels, two node-displacement output channels."""
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(7, width, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(width, 2, 1))

    def forward(self, features):
        return self.net(features) * (1 - features[:, 5:7])


def build_model(name, in_channels=10, width=None):
    if name == "unet":
        return ModernUNet(in_channels, width or 24)
    if name == "gnn":
        return MeshGNN(in_channels, width or 48)
    if name == "fno":
        return FNO2d(in_channels, width or 32)
    if name == "learned_update":
        return LearnedUpdate(in_channels, width or 24)
    raise ValueError("Unknown architecture " + name)
