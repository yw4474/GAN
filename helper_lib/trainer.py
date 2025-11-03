# helper_lib/trainer.py
from __future__ import annotations
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

def train_model(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str = "cpu",
    epochs: int = 2
) -> nn.Module:
    model.to(device)
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        avg = running / len(train_loader.dataset)
        print(f"[train] epoch={ep} loss={avg:.4f}")
    return model

# ==== train_gan ==============================================================
import torch, torch.nn as nn
from tqdm import tqdm

def train_gan(
    gan, data_loader, device: str = "cpu",
    epochs: int = 5, latent_dim: int = 100,
    lr_g: float = 2e-4, lr_d: float = 2e-4, beta1: float = 0.5
):
    gan.to(device)
    G, D = gan.generator, gan.discriminator
    bce = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, 0.999))

    for ep in range(1, epochs+1):
        g_loss_sum = d_loss_sum = n_sum = 0
        for real,_ in tqdm(data_loader, desc=f"GAN Epoch {ep}/{epochs}"):
            real = real.to(device)                    # (B,1,28,28) in [-1,1]
            B = real.size(0)
            ones  = torch.ones(B, device=device)
            zeros = torch.zeros(B, device=device)

            # ---- update D ----
            D.train(); G.train()
            # real
            d_real = D(real)
            loss_d_real = bce(d_real, ones)
            # fake
            z = torch.randn(B, latent_dim, device=device)
            with torch.no_grad():
                fake = G(z)
            d_fake = D(fake.detach())
            loss_d_fake = bce(d_fake, zeros)

            loss_d = loss_d_real + loss_d_fake
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # ---- update G ---- (non-saturating)
            z = torch.randn(B, latent_dim, device=device)
            fake = G(z)
            d_fake_for_g = D(fake)
            loss_g = bce(d_fake_for_g, ones)
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            g_loss_sum += loss_g.item()*B; d_loss_sum += loss_d.item()*B; n_sum += B

        print(f"[GAN] epoch={ep}  loss_G={g_loss_sum/n_sum:.4f}  loss_D={d_loss_sum/n_sum:.4f}")
    return gan
