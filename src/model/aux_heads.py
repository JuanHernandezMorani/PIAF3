import torch.nn as nn

def _make_head(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=1),
    )

class AuxHeads(nn.Module):
    """
    Heads auxiliares opcionales para predecir mapas físicos (para regularización).
    """
    def __init__(self, in_channels: int, normals=True, rough=True, spec=True, emiss=True):
        super().__init__()
        self.normals = _make_head(in_channels, 3) if normals else None
        self.rough   = _make_head(in_channels, 1) if rough   else None
        self.spec    = _make_head(in_channels, 1) if spec    else None
        self.emiss   = _make_head(in_channels, 1) if emiss   else None

    def forward(self, feat):
        out = {}
        if self.normals: out["normals"] = self.normals(feat)
        if self.rough:   out["rough"]   = self.rough(feat)
        if self.spec:    out["spec"]    = self.spec(feat)
        if self.emiss:   out["emiss"]   = self.emiss(feat)
        return out
