import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Tuple, Dict, Union, List
import numpy as np
import os
import json
from datetime import datetime


class CGANTrainer:
    def __init__(
            self,
            generator: nn.Module,
            discriminator: nn.Module,
            output_dir: str,
            lambda_l1: float = 100.0,
            lr_g: float = 0.0002,
            lr_d: float = 0.00002,
            beta1: float = 0.5,
            beta2: float = 0.999,
            discriminator_loss_coefficient: float = 1.0,
            device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.discriminator_loss_coefficient = discriminator_loss_coefficient
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.lambda_l1 = lambda_l1
        self.device = device
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize loss history
        self.loss_history = {
            'train': [],
            'val': []
        }

        # Initialize optimizers
        self.optimizer_G = Adam(
            generator.parameters(),
            lr=lr_g,
            betas=(beta1, beta2)
        )
        self.optimizer_D = Adam(
            discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, beta2)
        )

        # Loss functions
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()

        # Initialize loss logging files
        self._init_loss_logs()

    def _init_loss_logs(self):
        """Initialize loss logging files with headers"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create logging files
        self.train_log_path = os.path.join(self.output_dir, f'train_losses_{timestamp}.csv')
        self.val_log_path = os.path.join(self.output_dir, f'val_losses_{timestamp}.csv')

        # Write headers
        with open(self.train_log_path, 'w') as f:
            f.write('epoch,D_real,D_fake,D_total,G_GAN,G_L1,G_total\n')

        with open(self.val_log_path, 'w') as f:
            f.write('epoch,val_G_GAN,val_G_L1,val_total\n')

    def _log_losses(self, epoch: int, losses: Dict[str, float], is_train: bool = True):
        """Log losses to appropriate file"""
        log_path = self.train_log_path if is_train else self.val_log_path

        # Prepare loss values in correct order
        if is_train:
            values = [
                epoch,
                losses.get('D_real', 0),
                losses.get('D_fake', 0),
                losses.get('D_total', 0),
                losses.get('G_GAN', 0),
                losses.get('G_L1', 0),
                losses.get('G_total', 0)
            ]
        else:
            values = [
                epoch,
                losses.get('val_G_GAN', 0),
                losses.get('val_G_L1', 0),
                losses.get('val_total', 0)
            ]

        # Write to CSV
        with open(log_path, 'a') as f:
            f.write(','.join(map(str, values)) + '\n')

    def _save_loss_history(self):
        """Save complete loss history to JSON"""
        history_path = os.path.join(self.output_dir, 'loss_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.loss_history, f, indent=4)

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
        loss_D_real = self.criterion_GAN(pred_real, label_real)

        # Fake pairs D(x, G(x, z))
        fake_target = self.generator(real_input)
        fake_concat = torch.cat([real_input, fake_target.detach()], dim=1)
        pred_fake = self.discriminator(fake_concat)
        label_fake = torch.zeros_like(pred_fake)
        loss_D_fake = self.criterion_GAN(pred_fake, label_fake)

        # Combined D loss
        loss_D = (loss_D_real + loss_D_fake) * self.discriminator_loss_coefficient
        loss_D.backward()
        self.optimizer_D.step()

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
        real_input = real_input.to(self.device)
        real_target = real_target.to(self.device)

        self.set_requires_grad(self.discriminator, True)
        loss_D, D_losses = self.train_discriminator(real_input, real_target)

        self.set_requires_grad(self.discriminator, False)
        loss_G, G_losses = self.train_generator(real_input, real_target)

        losses = {}
        losses.update(D_losses)
        losses.update(G_losses)
        return losses

    @torch.no_grad()
    def evaluate(self, val_pbar, epoch: int) -> Dict[str, float]:
        """Evaluate the model on the validation set"""
        self.generator.eval()
        self.discriminator.eval()

        val_losses = []
        for real_input, real_target, _ in val_pbar:
            real_input = real_input.to(self.device)
            real_target = real_target.to(self.device)

            fake_target = self.generator(real_input)
            loss_L1 = self.criterion_L1(fake_target, real_target) * self.lambda_l1

            fake_concat = torch.cat([real_input, fake_target], dim=1)
            pred_fake = self.discriminator(fake_concat)
            label_real = torch.ones_like(pred_fake)
            loss_G_GAN = self.criterion_GAN(pred_fake, label_real)

            loss_total = loss_G_GAN + loss_L1

            batch_losses = {
                'val_G_GAN': loss_G_GAN.item(),
                'val_G_L1': loss_L1.item(),
                'val_total': loss_total.item()
            }
            val_losses.append(batch_losses)
            val_pbar.set_postfix(**{k: f'{v:.4f}' for k, v in batch_losses.items()})

        avg_val_losses = {}
        for key in val_losses[0].keys():
            avg_val_losses[key] = np.mean([x[key] for x in val_losses])

        # Log validation losses
        self._log_losses(epoch, avg_val_losses, is_train=False)
        self.loss_history['val'].append(avg_val_losses)

        self.generator.train()
        self.discriminator.train()

        return avg_val_losses

    def train_epoch(self, train_pbar, val_pbar=None, epoch: int = 0) -> Dict[str, float]:
        """Train for one epoch and optionally evaluate"""
        epoch_losses = []
        for batch_data in train_pbar:
            real_input, real_target, _ = batch_data
            batch_losses = self.train_step(real_input, real_target)
            epoch_losses.append(batch_losses)
            train_pbar.set_postfix(**{k: f'{v:.4f}' for k, v in batch_losses.items()})

        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([x[key] for x in epoch_losses])

        # Log training losses
        self._log_losses(epoch, avg_losses, is_train=True)
        self.loss_history['train'].append(avg_losses)

        if val_pbar is not None:
            val_losses = self.evaluate(val_pbar, epoch)
            avg_losses.update(val_losses)

        # Save complete loss history after each epoch
        self._save_loss_history()

        return avg_losses

    def save_checkpoint(self, path: str, epoch: int, val_losses: Dict[str, float] = None):
        """Save a training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'loss_history': self.loss_history  # Include complete loss history in checkpoint
        }
        if val_losses is not None:
            checkpoint['val_losses'] = val_losses
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Tuple[int, Dict[str, float]]:
        """Load a training checkpoint"""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.loss_history = checkpoint.get('loss_history', {'train': [], 'val': []})
        val_losses = checkpoint.get('val_losses', None)
        return checkpoint['epoch'], val_losses