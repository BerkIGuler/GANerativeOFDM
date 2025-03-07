from c_gan import CGANTrainer
from generator import UNet
from discriminator import PatchDiscriminator
from dataloader import MatDataset
from parser import create_parser, process_args
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import json
from tqdm import tqdm


def save_dict_to_json(d, filepath):
    """Helper function to save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(d, f, indent=4)


def main():
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    args = process_args(args)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments to JSON
    args_dict = vars(args)
    args_json_path = output_dir / 'training_args.json'
    save_dict_to_json(args_dict, args_json_path)

    # Initialize training history
    history = {
        'train_losses': [],
        'val_losses': []
    }

    # Set device
    device = torch.device(args.device)

    # Initialize models
    generator = UNet(p_dropout=args.dropout, debug=args.debug)
    discriminator = PatchDiscriminator(
        in_channels=4,
        base_filters=args.base_filters,
        norm_layer=torch.nn.BatchNorm2d
    )

    # Create data loaders with tqdm
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
        output_dir=args.output_dir,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        beta1=args.beta1,
        beta2=args.beta2,
        device=device,
        discriminator_loss_coefficient=args.d_loss_coef
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        start_epoch, _ = trainer.load_checkpoint(args.resume)
        print(f"Resuming from epoch {start_epoch}")

    # Training loop with tqdm
    best_val_loss = float('inf')
    epoch_pbar = tqdm(range(start_epoch, args.num_epochs), desc="Training Progress")

    for epoch in epoch_pbar:
        # Train and evaluate
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs} (Train)", leave=False)
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs} (Val)", leave=False)

        losses = trainer.train_epoch(train_pbar, val_pbar)

        # Split and store losses
        train_losses = {k: v for k, v in losses.items() if not k.startswith('val_')}
        val_losses = {k: v for k, v in losses.items() if k.startswith('val_')}

        history['train_losses'].append({
            'epoch': epoch + 1,
            **train_losses
        })
        history['val_losses'].append({
            'epoch': epoch + 1,
            **val_losses
        })

        # Update progress bar description with losses
        loss_str = f"Train Loss: {train_losses.get('total', 0):.4f}, Val Loss: {val_losses.get('val_total', 0):.4f}"
        epoch_pbar.set_postfix_str(loss_str)

        # Save checkpoint at specified frequency
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_dir = output_dir / f"epoch_{epoch + 1}"
            checkpoint_dir.mkdir(exist_ok=True)

            # Save model checkpoint
            checkpoint_path = checkpoint_dir / "model_checkpoint.pth"
            trainer.save_checkpoint(checkpoint_path, epoch, val_losses)

            # Save current arguments
            save_dict_to_json(args_dict, checkpoint_dir / "args.json")

            # Save current losses
            save_dict_to_json(train_losses, checkpoint_dir / "train_losses.json")
            save_dict_to_json(val_losses, checkpoint_dir / "val_losses.json")

        # Save best model based on validation loss
        val_total_loss = losses['val_total']
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_model_dir = output_dir / 'best_model'
            best_model_dir.mkdir(exist_ok=True)

            # Save best model checkpoint
            trainer.save_checkpoint(best_model_dir / "model_checkpoint.pth", epoch, val_losses)

            # Save arguments and losses for best model
            save_dict_to_json(args_dict, best_model_dir / "args.json")
            save_dict_to_json(train_losses, best_model_dir / "train_losses.json")
            save_dict_to_json(val_losses, best_model_dir / "val_losses.json")

    # Save complete training history
    save_dict_to_json(history, output_dir / 'training_history.json')


if __name__ == '__main__':
    main()