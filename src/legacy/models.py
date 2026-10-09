from torch import nn
import torch
from torch.nn import functional as F
from utils import rel_bandpowers, alpha_peak_hz, multiband_losses, asym_band_losses

class DWConv1d(nn.Module):
    def __init__(self, cin, cout, k=7, d=1, g=4, groups=24):
        super().__init__()
        pad = (k//2)*d
        self.dw = nn.Conv1d(cin, cin, k, padding=pad, dilation=d, groups=groups, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(g, cout), num_channels=cout)
        self.act = nn.SiLU()
    def forward(self, x): return self.act(self.gn(self.pw(self.dw(x))))

class bNormEncoder(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 3, padding=1, bias=False), nn.BatchNorm1d(ch), nn.SiLU(),
            DWConv1d(ch, ch, 7, groups=1), DWConv1d(ch, ch, 7, d=2, groups=1))
        self.mid  = nn.Sequential(DWConv1d(ch, ch, 7, d=4, groups=1), DWConv1d(ch, ch, 7, d=8, groups=1))
        self.out  = nn.Conv1d(ch, ch, 1)
    def forward(self, x):
        # x: [B,1,512]
        h = self.out(self.mid(self.stem(x)))
        return h
class Encoder(nn.Module):
    def __init__(self, ch=32, groups=24, scale_up=1):
        print('Encoder using groups:', groups)
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 3, padding=1, bias=False), nn.GroupNorm(min(4, ch), ch), nn.SiLU(),
            DWConv1d(ch, ch, 7, groups=groups), DWConv1d(ch, ch, 7, d=2, groups=max(groups//2, 1)))
        self.mid  = nn.Sequential(DWConv1d(ch, ch, 7, d=4, groups=1), DWConv1d(ch, ch, 7, d=8, groups=1))
        self.out  = nn.Conv1d(ch, ch, 1)
    def forward(self, x):
        # x: [B,1,512]
        h = self.out(self.mid(self.stem(x)))
        return h

class shortEncoder4Conv(nn.Module):
    def __init__(self, ch=32, groups=24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 3, padding=1, bias=False), nn.GroupNorm(min(4, ch), ch), nn.SiLU(),
            DWConv1d(ch, ch, 7, groups=groups))
        self.mid  = nn.Sequential(DWConv1d(ch, ch, 7, d=2, groups=max(groups//2, 1)), DWConv1d(ch, ch, 7, d=4, groups=1))
        self.out  = nn.Conv1d(ch, ch, 1)
    def forward(self, x):
        # x: [B,1,512]
        h = self.out(self.mid(self.stem(x)))
        return h

class shortEncoder3Conv(nn.Module):
    def __init__(self, ch=32, groups=24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 3, padding=1, bias=False), nn.GroupNorm(min(4, ch), ch), nn.SiLU(),
            DWConv1d(ch, ch, 7, groups=groups))
        self.mid  = DWConv1d(ch, ch, 7, d=2, groups=max(groups//2, 1))
        self.out  = nn.Conv1d(ch, ch, 1, groups=1)
    def forward(self, x):
        # x: [B,1,512]
        h = self.out(self.mid(self.stem(x)))
        return h

class deeperEncoder(nn.Module):
    def __init__(self, ch=32, groups=24, scale_up=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 3, padding=1, bias=False), nn.GroupNorm(min(4, ch), ch), nn.SiLU(),
            DWConv1d(ch, scale_up*ch, 7, groups=max(groups//2, 1)), DWConv1d(scale_up*ch, scale_up*ch, 7, d=2, groups=groups), DWConv1d(scale_up*ch, scale_up*ch, 7, d=2, groups=groups))
        self.mid  = nn.Sequential(DWConv1d(scale_up*ch, ch, 7, d=4, groups=max(groups//2, 1)), DWConv1d(ch, ch, 7, d=8, groups=1))
        self.out  = nn.Conv1d(ch, ch, 1)
    def forward(self, x):
        # x: [B,1,512]
        h = self.out(self.mid(self.stem(x)))
        return h

class shorterHead(nn.Module):
    def __init__(self, ch=32, groups=24, scale_up=1):
        super().__init__()
        self.f_z  = nn.Conv1d(ch, ch, 1)  # head-specific bottleneck (used for independence)
        self.dec  = nn.Sequential(DWConv1d(ch, scale_up*ch, 7, groups=groups), nn.Conv1d(scale_up*ch, 1, 1))

    def forward(self, h):
        z = self.f_z(h)                     # [B,ch,L]
        y = self.dec(z)                     # [B,1,L]
        z_g = torch.mean(z, dim=-1)         # [B,ch] global avg for independence
        return y, z_g

class Head(nn.Module):
    def __init__(self, ch=32, groups=24, scale_up=1):
        super().__init__()
        print(ch, scale_up*ch, groups, groups//2)
        self.f_z  = nn.Conv1d(ch, ch, 1)  # head-specific bottleneck (used for independence)
        self.dec  = nn.Sequential(DWConv1d(ch, scale_up*ch, 7, d=2, groups=max(groups//2, 1)), DWConv1d(scale_up*ch, ch, 7, groups=max(groups//2, 1)), nn.Conv1d(ch, 1, 1))
    def forward(self, h):
        z = self.f_z(h)                     # [B,ch,L]
        y = self.dec(z)                     # [B,1,L]
        z_g = torch.mean(z, dim=-1)         # [B,ch] global avg for independence
        return y, z_g

class deeperHead(nn.Module):
    def __init__(self, ch=32, groups=24, scale_up=2):
        super().__init__()
        self.f_z  = nn.Conv1d(ch, ch, 1)  # head-specific bottleneck (used for independence)
        self.dec  = nn.Sequential(DWConv1d(ch, scale_up*ch, 7, d=2, groups=max(groups//2, 1)), DWConv1d(scale_up*ch, scale_up*ch, groups=groups), DWConv1d(scale_up*ch, scale_up*ch, groups=groups), DWConv1d(scale_up*ch, ch, 7, groups=max(groups//2, 1)), DWConv1d(ch, ch, 7, groups=1), nn.Conv1d(ch, 1, 1))
    def forward(self, h):
        z = self.f_z(h)                     # [B,ch,L]
        y = self.dec(z)                     # [B,1,L]
        z_g = torch.mean(z, dim=-1)         # [B,ch] global avg for independence
        return y, z_g

class shorterSingleHeadNet(nn.Module):
    def __init__(self, ch=32, device='cuda', groups=24, scale_up=1, convLevel=4):
        super().__init__()
        if convLevel==3:
            self.enc = shortEncoder3Conv(ch, groups=groups)
        elif convLevel==4:
            self.enc = shortEncoder4Conv(ch, groups=groups)
        else:
            raise ValueError("convLevel not implemented")
        self.head_s = shorterHead(ch, groups=groups, scale_up=scale_up)   # clean
        self.device = device
    def forward(self, x):
        h = self.enc(x)                # [B,ch,L]
        y_s, z_s = self.head_s(h)
        return y_s, None, z_s, None
    def loss(self, batch, out, weights):
        return TwoHeadNet.loss(self, batch, out, weights)

class conv3SingleHeadNet(nn.Module):
    def __init__(self, ch=32, device='cuda', groups=24, scale_up=1, convLevel=4):
        super().__init__()
        self.enc = shortEncoder3Conv(ch, groups=groups)
        self.head_s = shorterHead(ch, groups=groups, scale_up=scale_up)   # clean
        self.device = device
    def forward(self, x):
        h = self.enc(x)                # [B,ch,L]
        y_s, z_s = self.head_s(h)
        return y_s, None, z_s, None
    def loss(self, batch, out, weights):
        return TwoHeadNet.loss(self, batch, out, weights)

class conv4SingleHeadNet(nn.Module):
    def __init__(self, ch=32, device='cuda', groups=24, scale_up=1, convLevel=4):
        super().__init__()
        self.enc = shortEncoder4Conv(ch, groups=groups)
        self.head_s = shorterHead(ch, groups=groups, scale_up=scale_up)   # clean
        self.device = device
    def forward(self, x):
        h = self.enc(x)                # [B,ch,L]
        y_s, z_s = self.head_s(h)
        return y_s, None, z_s, None
    def loss(self, batch, out, weights):
        return TwoHeadNet.loss(self, batch, out, weights)

class conv6SingleHeadNet(nn.Module):
    def __init__(self, ch=32, device='cuda', groups=24, scale_up=2):
        super().__init__()
        self.enc = deeperEncoder(ch, groups=groups, scale_up=scale_up)
        self.head_s = deeperHead(ch, groups=groups, scale_up=scale_up)   # clean
        self.device = device
    def forward(self, x):
        h = self.enc(x)                # [B,ch,L]
        y_s, z_s = self.head_s(h)
        return y_s, None, z_s, None
    def loss(self, batch, out, weights):
        return TwoHeadNet.loss(self, batch, out, weights)

class conv5SingleHeadNet(nn.Module):
    """Same multi-loss but single head (A1.5 in doc)"""

    def __init__(self, ch=32, device='cuda', groups=None, scale_up=1):
        super().__init__()
        if not groups:
            groups = ch
        self.enc = Encoder(ch, groups=groups, scale_up=scale_up)
        self.head_s = Head(ch, groups=groups, scale_up=scale_up)   # clean
        self.device = device
    def forward(self, x):
        h = self.enc(x)                # [B,ch,L]
        y_s, z_s = self.head_s(h)
        return y_s, None, z_s, None
    def loss(self, batch, out, weights):
        return TwoHeadNet.loss(self, batch, out, weights)

class TwoHeadNet(nn.Module):
    def __init__(self, ch=32, device='cuda'):
        super().__init__()
        self.enc = Encoder(ch)
        self.head_s = Head(ch)   # clean
        self.head_a = Head(ch)   # artifact
        self.device = device
    def forward(self, x):
        h = self.enc(x)                # [B,ch,L]
        y_s, z_s = self.head_s(h)
        y_a, z_a = self.head_a(h)
        return y_s, y_a, z_s, z_a

    def loss(self, batch, out, weights):
        # x = batch["x"].to(self.device)[:,None,:]
        s = batch["s"].to(self.device)[:,None,:]
        # a = batch["a"].to(self.device)[:,None,:]
        y_s, _, _, _ = out

        L_clean = F.mse_loss(y_s, s)
        # if y_a is not None:
        #     L_art   = F.mse_loss(y_a, a)
        #     L_sum   = F.l1_loss(x, y_s + y_a)
        #     cs = F.cosine_similarity(z_s, z_a, dim=1)
        #     L_ind = cs.abs().mean()
        # else:
        #     L_art = torch.tensor(0., device=self.device)
        #     L_sum = torch.tensor(0., device=self.device)
        #     L_ind = torch.tensor(0., device=self.device)

        # # physiology
        # rb_s  = rel_bandpowers(s).squeeze(-2)   # [B,5]
        # rb_sh = rel_bandpowers(y_s).squeeze(-2)
        # L_bands = F.l1_loss(rb_sh, rb_s)

        # f_s   = alpha_peak_hz(s)
        # f_sh  = alpha_peak_hz(y_s)
        # L_alpha = (f_sh - f_s).abs().mean()

        # # multi band
        # L_rp, L_cent, _ = multiband_losses(y_s.squeeze(1), s.squeeze(1), fs=256,
        #                            nfft=512, tau=0.2,
        #                            w_rp=1.0, w_cent=0.5)

        # L_dta, L_bg_hi, L_alpha_cent = asym_band_losses(
        #                                     y_s.squeeze(1), s.squeeze(1), fs=256, nfft=512, tau=0.2,
        #                                     margin_bg=0.02  # allow 2% RP slack in beta/gamma
        #                                 )
        # L_asym = 0.1*L_dta + L_bg_hi + 100*L_alpha_cent

        # # optional: same-clean consistency on encoder features (view x2)
        # L_cons = torch.tensor(0., device=self.device)
        # if "x2" in batch:
        #     x2 = batch["x2"].to(self.device)[:,None,:]
        #     with torch.no_grad():
        #         h = self.enc(x)
        #     h2 = self.enc(x2)
        #     z1 = torch.mean(h,  dim=-1)
        #     z2 = torch.mean(h2, dim=-1)
        #     z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
        #     L_cons = ((z1 - z2)**2).mean()

        # w = weights
        # L = (w["L_clean"]*L_clean
        #         + w["L_art"]*L_art
        #         + w["L_sum"]*L_sum
        #         + w["L_ind"]*L_ind
        #         + w["L_bands"]*L_bands
        #         + w["L_alpha"]*L_alpha
        #         + w["L_cons"]*L_cons
        #         + w["L_rp"]*L_rp
        #         + w["L_cent"]*L_cent)
        # L = (w.get("L_clean", 0)*L_clean
        #         + w.get("L_art", 0)*L_art
        #         + w.get("L_sum", 0)*L_sum
        #         + w.get("L_ind", 0)*L_ind
        #         + w.get("L_bands", 0)*L_bands
        #         + w.get("L_alpha", 0)*L_alpha
        #         + w.get("L_cons", 0)*L_cons
        #         + w.get("L_rp", 0)*L_rp
        #         + w.get("L_cent", 0)*L_cent
        #         + w.get("L_asym", 0)*L_asym

        #         )
        L = (weights.get("L_clean", 0)*L_clean)
        # return L, dict(L_clean=float(L_clean.item()),
        #                 L_art=float(L_art.item()), L_sum=float(L_sum.item()),
        #                 L_ind=float(L_ind.item()), L_bands=float(L_bands.item()),
        #                 L_alpha=float(L_alpha.item()), L_cons=float(L_cons.item()))
        return L, dict(L_clean=float(L_clean.item()))

        # without .item
        # return L, dict(L_clean=L_clean,
        #                L_art=L_art, L_sum=L_sum,
        #                L_ind=L_ind, L_bands=L_bands,
        #                L_alpha=L_alpha, L_cons=L_cons)

class bNormTwoHeadNet(TwoHeadNet):
    def __init__(self, ch=32, device='cuda'):
        super().__init__()
        self.enc = bNormEncoder(ch)
        self.head_s = Head(ch)   # clean
        self.head_a = Head(ch)   # artifact
        self.device = device

def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

class baselineModel(nn.Module):
    """Single-head, MSE only (no artifact head, no additivity, no physiology, no consistency)."""
    def __init__(self, device='cuda', ch=32):
        super().__init__()
        self.enc = Encoder(ch)
        self.head_s = Head(ch)   # clean
        self.device=device
    def forward(self, x):
        h = self.enc(x)                # [B,ch,L]
        y_s, z_s = self.head_s(h)
        return y_s, None, z_s, None

    def loss(self, batch, out, weights):
        y_s, y_a, z_s, z_a = out
        x = batch["x"].to(self.device)[:,None,:]
        s = batch["s"].to(self.device)[:,None,:]
        a = batch["a"].to(self.device)[:,None,:]

        rec_loss = F.mse_loss(y_s, s)

        total_loss = rec_loss

        return total_loss, {"L_clean":float(rec_loss.item())}
