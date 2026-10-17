"""3D Scientific ML architectures for voxel FEM density warm starts.

Input [batch, channels, z, y, x], output density logits [batch, 1, z, y, x].
No 2D-to-3D extrusion: network operators and training data are volumetric.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Residual3D(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(inp, out, 3, padding=1), nn.GroupNorm(4, out), nn.GELU(),
            nn.Conv3d(out, out, 3, padding=1), nn.GroupNorm(4, out))
        self.skip = nn.Conv3d(inp, out, 1) if inp != out else nn.Identity()

    def forward(self, x):
        return F.gelu(self.conv(x) + self.skip(x))


class UNet3D(nn.Module):
    def __init__(self, in_channels=10, width=12):
        super().__init__()
        self.down1 = Residual3D(in_channels, width)
        self.down2 = Residual3D(width, width*2)
        self.core = Residual3D(width*2, width*4)
        self.up2 = Residual3D(width*6, width*2)
        self.up1 = Residual3D(width*3, width)
        self.out = nn.Conv3d(width, 1, 1)

    def forward(self, x):
        if min(x.shape[-3:]) < 4:
            raise ValueError("UNet3D requires at least four voxels on each axis")
        a = self.down1(x)
        b = self.down2(F.avg_pool3d(a, 2))
        z = self.core(F.avg_pool3d(b, 2))
        z = self.up2(torch.cat((F.interpolate(z, size=b.shape[-3:],
                           mode="trilinear", align_corners=False), b), dim=1))
        z = self.up1(torch.cat((F.interpolate(z, size=a.shape[-3:],
                           mode="trilinear", align_corners=False), a), dim=1))
        return self.out(z)


class SpectralConv3D(nn.Module):
    """Fourier operator with positive/negative z-y quadrants and rFFT x."""
    def __init__(self, inp, out, modes=(4, 4, 6)):
        super().__init__()
        self.out_channels = out
        self.modes = modes
        scale = 1 / math.sqrt(inp*out)
        self.weights = nn.ParameterList([
            nn.Parameter(scale*torch.randn(inp, out, *modes, dtype=torch.cfloat))
            for _ in range(4)
        ])

    def forward(self, x):
        b, _, nz, ny, nx = x.shape
        spectrum = torch.fft.rfftn(x, dim=(-3,-2,-1), norm="ortho")
        out = spectrum.new_zeros((b,self.out_channels,nz,ny,nx//2+1))
        mz = min(self.modes[0], max(1,nz//2))
        my = min(self.modes[1], max(1,ny//2))
        mx = min(self.modes[2], nx//2+1)
        for w, zs, ys in zip(self.weights,
                 (slice(None,mz),slice(None,mz),slice(-mz,None),slice(-mz,None)),
                 (slice(None,my),slice(-my,None),slice(None,my),slice(-my,None))):
            out[:,:,zs,ys,:mx] = torch.einsum(
                "bizyx,iozyx->bozyx",
                spectrum[:,:,zs,ys,:mx], w[:,:,:mz,:my,:mx])
        return torch.fft.irfftn(out, s=(nz,ny,nx), dim=(-3,-2,-1), norm="ortho")


class FNO3D(nn.Module):
    def __init__(self, in_channels=10, width=16, depth=3):
        super().__init__()
        self.lift = nn.Conv3d(in_channels,width,1)
        self.fourier = nn.ModuleList([SpectralConv3D(width,width) for _ in range(depth)])
        self.local = nn.ModuleList([nn.Conv3d(width,width,1) for _ in range(depth)])
        self.out = nn.Sequential(nn.Conv3d(width,width,1),nn.GELU(),nn.Conv3d(width,1,1))

    def forward(self,x):
        z=self.lift(x)
        for fourier,local in zip(self.fourier,self.local):
            z=F.gelu(fourier(z)+local(z))
        return self.out(z)


def voxel_edges(nz, ny, nx, device, dtype):
    coords = torch.arange(nz*ny*nx,device=device).reshape(nz,ny,nx)
    edges = []
    for axis in range(3):
        low = [slice(None)]*3
        high = [slice(None)]*3
        low[axis]=slice(None,-1)
        high[axis]=slice(1,None)
        p,q=coords[tuple(low)].flatten(),coords[tuple(high)].flatten()
        edges.extend(((p,q),(q,p)))
    src=torch.cat([p for p,_ in edges])
    dst=torch.cat([q for _,q in edges])
    return src,dst


class GraphConv3D(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(width*2,width),nn.SiLU(),
                                     nn.Linear(width,width))
        self.update = nn.Sequential(nn.Linear(width*2,width),nn.SiLU(),
                                    nn.Linear(width,width))
        self.norm = nn.LayerNorm(width)

    def forward(self,h,edge):
        src,dst=edge
        msg=self.message(torch.cat((h[:,src],h[:,dst]),dim=-1))
        aggregate=torch.zeros_like(h)
        aggregate.index_add_(1,dst,msg)
        count=torch.bincount(dst,minlength=h.shape[1]).to(h.dtype).clamp_min(1)
        aggregate=aggregate/count[None,:,None]
        return self.norm(h+self.update(torch.cat((h,aggregate),dim=-1)))


class MeshGNN3D(nn.Module):
    """3D voxel-element adjacency (6 neighbors); not general tetrahedral FEM."""
    def __init__(self,in_channels=10,width=24,depth=4):
        super().__init__()
        self.embed=nn.Linear(in_channels,width)
        self.blocks=nn.ModuleList([GraphConv3D(width) for _ in range(depth)])
        self.output=nn.Linear(width,1)

    def forward(self,x):
        b,c,nz,ny,nx=x.shape
        edge=voxel_edges(nz,ny,nx,x.device,x.dtype)
        h=self.embed(x.permute(0,2,3,4,1).reshape(b,nz*ny*nx,c))
        for block in self.blocks:
            h=block(h,edge)
        return self.output(h).transpose(1,2).reshape(b,1,nz,ny,nx)


def build_3d_model(name, channels=10):
    if name=="unet3d":
        return UNet3D(channels)
    if name=="fno3d":
        return FNO3D(channels)
    if name=="gnn3d":
        return MeshGNN3D(channels)
    raise ValueError("Unknown 3D architecture: "+name)
