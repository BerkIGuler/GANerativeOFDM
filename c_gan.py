import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Union
import numpy as np


class CGANTrainer:
    def __init__(
            self,
            generator: nn.Module,
            discriminator: nn.Module,
            lambda_l1: float = 100.0,
            lr: float = 0.0002,
            beta1: float = 0.5,
            beta2: float = 0.999,
            device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.lambda_l1 = lambda_l1
        self.device = device

        # Initialize optimizers
        self.optimizer_G = Adam(
            generator.parameters(),
            lr=lr,
            betas=(beta1, beta2)
        )
        self.optimizer_D = Adam(
            discriminator.parameters(),
            lr=lr,
            betas=(beta1, beta2)
        )

        # Loss functions
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train_discriminator(
            self,
            real_input: torch.Tensor,
            real_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Train discriminator for one step"""
        self.optimizer_D.zero_grad()

        # Real pairs D(x, y)
        real_concat = torch.cat([real_input, real_target], dim=1)
        pred_real = self.discriminator(real_concat)
        label_real = torch.ones_like(pred_real)
        # log(D(x, y))
        loss_D_real = self.criterion_GAN(pred_real, label_real)

        # Fake pairs D(x, G(x, z))
        fake_target = self.generator(real_input)
        fake_concat = torch.cat([real_input, fake_target.detach()], dim=1)
        pred_fake = self.discriminator(fake_concat)
        label_fake = torch.zeros_like(pred_fake)
        # log(1-D(x, G(x, z)))
        loss_D_fake = self.criterion_GAN(pred_fake, label_fake)

        # Combined D loss (divided by 2 as mentioned in the paper)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.optimizer_D.step()

        # Return losses for logging
        return loss_D, {
            'D_real': loss_D_real.item(),
            'D_fake': loss_D_fake.item(),
            'D_total': loss_D.item()
        }

    def train_generator(
            self,
            real_input: torch.Tensor,
            real_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Train generator for one step"""
        self.optimizer_G.zero_grad()

        # Generate fake target
        fake_target = self.generator(real_input)
        fake_concat = torch.cat([real_input, fake_target], dim=1)

        # GAN loss
        pred_fake = self.discriminator(fake_concat)
        label_real = torch.ones_like(pred_fake)
        loss_G_GAN = self.criterion_GAN(pred_fake, label_real)

        # L1 loss
        loss_G_L1 = self.criterion_L1(fake_target, real_target) * self.lambda_l1

        # Combined G loss
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        self.optimizer_G.step()

        return loss_G, {
            'G_GAN': loss_G_GAN.item(),
            'G_L1': loss_G_L1.item(),
            'G_total': loss_G.item()
        }

    def train_step(
            self,
            real_input: torch.Tensor,
            real_target: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one training step"""
        # Move data to device
        real_input = real_input.to(self.device)
        real_target = real_target.to(self.device)

        # Train D
        self.set_requires_grad(self.discriminator, True)
        loss_D, D_losses = self.train_discriminator(real_input, real_target)

        # Train G
        self.set_requires_grad(self.discriminator, False)
        loss_G, G_losses = self.train_generator(real_input, real_target)

        # Combine all losses for logging
        losses = {}
        losses.update(D_losses)
        losses.update(G_losses)
        return losses

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on the validation set"""
        self.generator.eval()
        self.discriminator.eval()

        val_losses = []
        for real_input, real_target, _ in dataloader:
            real_input = real_input.to(self.device)
            real_target = real_target.to(self.device)

            # Generate fake target
            fake_target = self.generator(real_input)

            # Calculate L1 loss
            loss_L1 = self.criterion_L1(fake_target, real_target) * self.lambda_l1

            # Calculate GAN loss
            fake_concat = torch.cat([real_input, fake_target], dim=1)
            pred_fake = self.discriminator(fake_concat)
            label_real = torch.ones_like(pred_fake)
            loss_G_GAN = self.criterion_GAN(pred_fake, label_real)

            # Calculate total loss
            loss_total = loss_G_GAN + loss_L1

            val_losses.append({
                'val_G_GAN': loss_G_GAN.item(),
                'val_G_L1': loss_L1.item(),
                'val_total': loss_total.item()
            })

        # Calculate average validation losses
        avg_val_losses = {}
        for key in val_losses[0].keys():
            avg_val_losses[key] = np.mean([x[key] for x in val_losses])

        self.generator.train()
        self.discriminator.train()

        return avg_val_losses

    def train_epoch(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None) -> Dict[str, float]:
        """Train for one epoch and optionally evaluate"""
        # Training
        epoch_losses = []
        for batch_data in train_dataloader:
            real_input, real_target, _ = batch_data
            batch_losses = self.train_step(real_input, real_target)
            epoch_losses.append(batch_losses)

        # Calculate average training losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([x[key] for x in epoch_losses])

        # Validation if dataloader provided
        if val_dataloader is not None:
            val_losses = self.evaluate(val_dataloader)
            avg_losses.update(val_losses)

        return avg_losses

    def save_checkpoint(self, path: str, epoch: int, val_losses: Dict[str, float] = None):
        """Save a training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }
        if val_losses is not None:
            checkpoint['val_losses'] = val_losses
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Tuple[int, Dict[str, float]]:
        """Load a training checkpoint. Returns the epoch number and validation losses if available."""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        val_losses = checkpoint.get('val_losses', None)
        return checkpoint['epoch'], val_losses