import tensorflow as tf
import numpy as np
import argparse
import os
from models.trainer import RL_V2V_GAN_Trainer
from data.dataset_factory import DatasetFactory
from utils.evaluation import evaluate_model

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RL-V2V-GAN Training and Evaluation')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['train', 'eval', 'prepare_data'],
                      help='Mode to run: train, eval, or prepare_data')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['synthetic'],
                      help='Dataset to use (synthetic for now)')
    
    # Model parameters
    parser.add_argument('--input_shape', type=int, nargs=4, default=[16, 64, 64, 3],
                      help='Input shape (seq_length, height, width, channels)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                      help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs for training')
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                      help='Steps per epoch')
    
    # RL parameters
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor for RL')
    parser.add_argument('--tau', type=float, default=0.001,
                      help='Target network update rate')
    parser.add_argument('--lambda_v', type=float, default=1.0,
                      help='Video loss coefficient')
    parser.add_argument('--lambda_rr', type=float, default=0.1,
                      help='Recurrent loss coefficient')
    parser.add_argument('--lambda_rc', type=float, default=0.1,
                      help='Recycle loss coefficient')
    parser.add_argument('--sigma1', type=float, default=0.5,
                      help='Bernoulli parameter for replay buffer sampling')
    parser.add_argument('--sigma2', type=float, default=0.5,
                      help='Bernoulli parameter for action selection')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save/load checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default='rl_v2v_gan',
                      help='Name of the checkpoint')
    
    # Evaluation parameters
    parser.add_argument('--eval_batches', type=int, default=5,
                      help='Number of batches to evaluate')
    
    return parser.parse_args()

def prepare_data(args):
    """Data preparation mode"""
    print(f"Preparing {args.dataset} dataset...")
    DatasetFactory.prepare_dataset(args.dataset)
    print("Data preparation completed.")

def train(args):
    """Training mode"""
    print("Starting training mode...")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create trainer
    trainer = RL_V2V_GAN_Trainer(
        input_shape=tuple(args.input_shape),
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        lambda_v=args.lambda_v,
        lambda_rr=args.lambda_rr,
        lambda_rc=args.lambda_rc,
        sigma1=args.sigma1,
        sigma2=args.sigma2
    )
    
    # Create datasets
    train_dataset = DatasetFactory.get_dataset(
        args.dataset,
        batch_size=args.batch_size,
        seq_length=args.input_shape[0],
        height=args.input_shape[1],
        width=args.input_shape[2],
        channels=args.input_shape[3]
    )
    
    test_dataset = DatasetFactory.get_dataset(
        args.dataset,
        batch_size=args.batch_size,
        seq_length=args.input_shape[0],
        height=args.input_shape[1],
        width=args.input_shape[2],
        channels=args.input_shape[3]
    )
    
    # Training loop with evaluation
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Training
        trainer.train(
            dataset=train_dataset,
            epochs=1,
            steps_per_epoch=args.steps_per_epoch
        )
        
        # Evaluation
        print("\nEvaluating model...")
        metrics = evaluate_model(trainer, test_dataset, num_batches=args.eval_batches)
        
        print("\nEvaluation Metrics:")
        print(f"FID Score: {metrics['fid']:.4f}")
        print(f"PSNR Score: {metrics['psnr']:.4f}")
        print(f"SSIM Score: {metrics['ssim']:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f"{args.checkpoint_name}_epoch_{epoch+1}"
        )
        trainer.save_weights(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

def evaluate(args):
    """Evaluation mode"""
    print("Starting evaluation mode...")
    
    # Create trainer
    trainer = RL_V2V_GAN_Trainer(
        input_shape=tuple(args.input_shape),
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        lambda_v=args.lambda_v,
        lambda_rr=args.lambda_rr,
        lambda_rc=args.lambda_rc,
        sigma1=args.sigma1,
        sigma2=args.sigma2
    )
    
    # Load latest checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    if os.path.exists(checkpoint_path):
        trainer.load_weights(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    # Create test dataset
    test_dataset = DatasetFactory.get_dataset(
        args.dataset,
        batch_size=args.batch_size,
        seq_length=args.input_shape[0],
        height=args.input_shape[1],
        width=args.input_shape[2],
        channels=args.input_shape[3]
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(trainer, test_dataset, num_batches=args.eval_batches)
    
    print("\nEvaluation Metrics:")
    print(f"FID Score: {metrics['fid']:.4f}")
    print(f"PSNR Score: {metrics['psnr']:.4f}")
    print(f"SSIM Score: {metrics['ssim']:.4f}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Run selected mode
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
    else:  # prepare_data mode
        prepare_data(args)

if __name__ == "__main__":
    main() 