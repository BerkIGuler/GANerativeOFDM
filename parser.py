import argparse
from pathlib import Path


def create_parser():
    parser = argparse.ArgumentParser(description='Train Conditional GAN for Signal Processing')

    # Data paths
    parser.add_argument('--train_path', type=str, default='dataset/train',
                        help='Path to training data directory')
    parser.add_argument('--val_path', type=str, default='dataset/val',
                        help='Path to validation data directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save checkpoints and logs')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Beta1 parameter for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 parameter for Adam optimizer')
    parser.add_argument('--lambda_l1', type=float, default=100.0,
                        help='Weight for L1 loss')

    # Model parameters
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate for generator')
    parser.add_argument('--base_filters', type=int, default=16,
                        help='Number of base filters in discriminator')
    parser.add_argument('--pilot_dims', nargs=2, type=int, default=[18, 2],
                        help='Dimensions of pilot signal (height width)')

    # Checkpointing
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')

    # Debug mode
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (prints model dimensions)')

    return parser


def process_args(args):
    """Process and validate command line arguments"""
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate paths
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    if not train_path.exists():
        raise ValueError(f"Training data directory {train_path} does not exist")
    if not val_path.exists():
        raise ValueError(f"Validation data directory {val_path} does not exist")

    # Validate resume checkpoint if provided
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise ValueError(f"Checkpoint file {resume_path} does not exist")

    return args
