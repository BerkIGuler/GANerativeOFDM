from torchgen.executorch.api.et_cpp import return_type

from c_gan import CGANTrainer
from generator import UNet
from discriminator import PatchDiscriminator
from dataloader import MatDataset
from parser import create_parser, process_args
from torch.utils.data import DataLoader
import torch
from pathlib import Path


def main():
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    args = process_args(args)

    # Set device
    device = torch.device(args.device)

    # TODO: check in_channels
    # Initialize models
    generator = UNet(p_dropout=args.dropout, debug=args.debug)
    discriminator = PatchDiscriminator(
        in_channels=4,  # 4 channels because we concatenate input and output
        base_filters=args.base_filters,
        norm_layer=torch.nn.BatchNorm2d
    )

    # Create data loaders
    train_dataloader = DataLoader(
        dataset=MatDataset(data_dir=args.train_path, pilot_dims=tuple(args.pilot_dims)),
        batch_size=args.batch_size,
        shuffle=True
    )

    val_dataloader = DataLoader(
        dataset=MatDataset(data_dir=args.val_path, pilot_dims=tuple(args.pilot_dims)),
        batch_size=args.batch_size,
        shuffle=False
    )

    # Initialize trainer
    trainer = CGANTrainer(
        generator=generator,
        discriminator=discriminator,
        lambda_l1=args.lambda_l1,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        device=device
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        start_epoch, _ = trainer.load_checkpoint(args.resume)
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.num_epochs):
        # Train and evaluate
        losses = trainer.train_epoch(train_dataloader, val_dataloader)

        # Print losses
        print(f"Epoch [{epoch + 1}/{args.num_epochs}]")
        print("Training Losses:")
        for name, value in losses.items():
            if not name.startswith('val_'):
                print(f"{name}: {value:.4f}")

        print("Validation Losses:")
        for name, value in losses.items():
            if name.startswith('val_'):
                print(f"{name}: {value:.4f}")

        # Save checkpoint at specified frequency
        if (epoch + 1) % args.save_freq == 0:
            save_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch + 1}.pth"
            trainer.save_checkpoint(
                save_path,
                epoch,
                {k: v for k, v in losses.items() if k.startswith('val_')}
            )

        # Save best model based on validation loss
        val_total_loss = losses['val_total']
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            save_path = Path(args.output_dir) / 'best_model.pth'
            trainer.save_checkpoint(
                save_path,
                epoch,
                {k: v for k, v in losses.items() if k.startswith('val_')}
            )


if __name__ == '__main__':
    main()