"""The two image models. Both take a stack of picture layers [B, C, 192, 192] and the per-fire numbers [B, K] and return one logit per cell [B, 192, 192].
unet: a U-Net with a pretrained ResNet-50 encoder (as in Karlsson et al. 2025); the per-fire numbers are painted over the map as extra layers.
firesensenet: re-implemented from the description in FireSenseNet (2026): a fuel/terrain branch (convolutions on the picture layers) and a weather branch, fused at every scale by a learned spatial gate (the cross-attentive feature interaction),
then a U-Net decoder. Our weather numbers are constant over the 19 km square, so the weather branch is an MLP whose output is broadcast over the map at each scale."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Squeeze(nn.Module):
    def forward(self, x):
        return x[:, 0]


class UNetResNet(nn.Module):
    def __init__(self, c_spatial, c_scalar, pretrained=True, encoder="resnet50"):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=c_spatial + c_scalar, classes=1)

    def forward(self, x, s):
        return self.net(torch.cat([x, s[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])], 1))[:, 0]


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Gate(nn.Module):
    """Learned spatial gate between the fuel features and the weather features at one scale: where does the weather matter, and where the ground?"""
    def __init__(self, c):
        super().__init__()
        self.g = nn.Sequential(nn.Conv2d(2 * c, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.Conv2d(c, 1, 1))

    def forward(self, f, w):
        a = torch.sigmoid(self.g(torch.cat([f, w], 1)))
        return f + a * w + (1 - a) * f * w.tanh()


class FireSenseNet(nn.Module):
    def __init__(self, c_spatial, c_scalar, ch=(32, 64, 128, 256)):
        super().__init__()
        self.enc = nn.ModuleList([block(c_spatial if i == 0 else ch[i - 1], ch[i]) for i in range(len(ch))])
        self.wx = nn.Sequential(nn.Linear(c_scalar, 128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.proj = nn.ModuleList([nn.Linear(128, c) for c in ch])
        self.gates = nn.ModuleList([Gate(c) for c in ch])
        self.dec = nn.ModuleList([block(ch[i] + ch[i - 1], ch[i - 1]) for i in range(len(ch) - 1, 0, -1)])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x, s):
        w = self.wx(s)
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            x = self.gates[i](x, self.proj[i](w)[:, :, None, None].expand_as(x))
            skips.append(x)
        for d, sk in zip(self.dec, skips[-2::-1]):
            x = d(torch.cat([F.interpolate(x, size=sk.shape[-2:], mode="bilinear", align_corners=False), sk], 1))
        return self.out(x)[:, 0]


class CNNAutoencoder(nn.Module):
    """The plain convolutional encoder-decoder of the first next-day benchmark (Huot et al. 2022): no skip connections, no pretraining. The per-fire numbers are painted over the map as extra layers."""
    def __init__(self, c_spatial, c_scalar):
        super().__init__()
        c = c_spatial + c_scalar
        self.enc = nn.Sequential(block(c, 64), nn.MaxPool2d(2), block(64, 128), nn.MaxPool2d(2), block(128, 256))
        self.dec = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), block(256, 128), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), block(128, 64), nn.Conv2d(64, 1, 1))

    def forward(self, x, s):
        return self.dec(self.enc(torch.cat([x, s[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])], 1)))[:, 0]


class SegFormerNet(nn.Module):
    """SegFormer (mix-transformer encoder, light all-MLP decoder), as compared in the published benchmarks. The ImageNet weights are used when they can be fetched, otherwise it is trained from scratch (and says so)."""
    def __init__(self, c_spatial, c_scalar, pretrained=True, encoder="mit_b2"):
        super().__init__()
        import segmentation_models_pytorch as smp
        try:
            self.net = smp.Segformer(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=c_spatial + c_scalar, classes=1)
        except Exception as e:
            print("SegFormer: pretrained weights not available, training from scratch:", repr(e)[:100], flush=True)
            self.net = smp.Segformer(encoder, encoder_weights=None, in_channels=c_spatial + c_scalar, classes=1)

    def forward(self, x, s):
        return self.net(torch.cat([x, s[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])], 1))[:, 0]


def make(name, c_spatial, c_scalar, pretrained=True):
    return {"unet": lambda: UNetResNet(c_spatial, c_scalar, pretrained), "unet18": lambda: UNetResNet(c_spatial, c_scalar, pretrained, "resnet18"), "firesensenet": lambda: FireSenseNet(c_spatial, c_scalar),
            "cnnae": lambda: CNNAutoencoder(c_spatial, c_scalar), "segformer": lambda: SegFormerNet(c_spatial, c_scalar, pretrained)}[name]()
