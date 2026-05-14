"""
Transformer Training and Demonstration Script
Demonstrates the Transformer model on a sequence copying task with noise addition.

This script:
1. Creates a simple synthetic dataset (sequence copying with noise)
2. Trains the Transformer model
3. Evaluates the model and generates metrics
4. Produces visualizations and results for reporting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from tqdm import tqdm
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import Transformer, count_parameters


class SequenceCopyDataset(Dataset):
    """
    Synthetic dataset for sequence copying task.
    The model learns to copy a sequence from input to output.
    This is a simplified task to demonstrate the Transformer's capability.
    """
    def __init__(self, num_samples=10000, seq_len=10, vocab_size=100, noise_prob=0.1):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.noise_prob = noise_prob

        # Special tokens
        self.PAD_IDX = 0
        self.SOS_IDX = 1  # Start of sequence
        self.EOS_IDX = 2  # End of sequence

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random sequence (avoid special tokens)
        seq = torch.randint(3, self.vocab_size, (self.seq_len,))

        # Add noise to some tokens
        if self.noise_prob > 0:
            noise_mask = torch.rand(self.seq_len) < self.noise_prob
            noise = torch.randint(3, self.vocab_size, (self.seq_len,))
            seq = torch.where(noise_mask, noise, seq)

        # Source: add SOS and EOS
        src = torch.cat([torch.tensor([self.SOS_IDX]), seq, torch.tensor([self.EOS_IDX])])

        # Target: add SOS at the beginning for teacher forcing
        tgt_input = torch.cat([torch.tensor([self.SOS_IDX]), seq])
        # Target output: the sequence plus EOS
        tgt_output = torch.cat([seq, torch.tensor([self.EOS_IDX])])

        return src, tgt_input, tgt_output


def collate_fn(batch):
    """Custom collate function to handle batching."""
    srcs, tgt_inputs, tgt_outputs = zip(*batch)

    srcs = torch.stack(srcs)
    tgt_inputs = torch.stack(tgt_inputs)
    tgt_outputs = torch.stack(tgt_outputs)

    return srcs, tgt_inputs, tgt_outputs


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    pbar = tqdm(dataloader, desc="Training")
    for src, tgt_input, tgt_output in pbar:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        optimizer.zero_grad()

        # Generate target mask
        tgt_seq_len = tgt_input.size(1)
        tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(device)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

        # Forward pass
        output = model(src, tgt_input, tgt_mask=tgt_mask)

        # Compute loss
        output = output.view(-1, output.size(-1))
        tgt_output = tgt_output.view(-1)
        loss = criterion(output, tgt_output)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute accuracy
        predictions = output.argmax(dim=-1)
        correct = (predictions == tgt_output).sum().item()
        total_correct += correct
        total_tokens += tgt_output.size(0)

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/tgt_output.size(0):.4f}"})

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_tokens
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for src, tgt_input, tgt_output in tqdm(dataloader, desc="Evaluating"):
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        # Generate target mask
        tgt_seq_len = tgt_input.size(1)
        tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(device)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

        # Forward pass
        output = model(src, tgt_input, tgt_mask=tgt_mask)

        # Compute loss
        output_flat = output.view(-1, output.size(-1))
        tgt_output_flat = tgt_output.view(-1)
        loss = criterion(output_flat, tgt_output_flat)

        # Compute accuracy
        predictions = output_flat.argmax(dim=-1)
        correct = (predictions == tgt_output_flat).sum().item()
        total_correct += correct
        total_tokens += tgt_output_flat.size(0)

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_tokens
    return avg_loss, avg_acc


@torch.no_grad()
def generate_examples(model, dataset, device, num_examples=5):
    """Generate some example predictions."""
    model.eval()
    examples = []

    for i in range(num_examples):
        src, tgt_input, tgt_output = dataset[i]
        src = src.unsqueeze(0).to(device)

        # Start with SOS token
        generated = torch.tensor([[dataset.SOS_IDX]]).to(device)

        # Generate sequence
        for _ in range(dataset.seq_len + 1):
            tgt_seq_len = generated.size(1)
            tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(device)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

            output = model(src, generated, tgt_mask=tgt_mask)
            next_token = output[0, -1, :].argmax().item()

            generated = torch.cat([generated, torch.tensor([[next_token]]).to(device)], dim=1)

            if next_token == dataset.EOS_IDX:
                break

        # Extract the sequence (without SOS and EOS)
        src_seq = src[0, 1:-1].cpu().tolist()
        tgt_seq = tgt_output.cpu().tolist()[:-1]
        pred_seq = generated[0, 1:-1].cpu().tolist()

        examples.append({
            "input": src_seq,
            "target": tgt_seq,
            "prediction": pred_seq,
            "match": src_seq == pred_seq
        })

    return examples


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")


def save_results(results, save_path):
    """Save results to JSON file."""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {save_path}")


def main():
    # Configuration
    config = {
        "vocab_size": 100,
        "seq_len": 10,
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "batch_size": 64,
        "num_epochs": 5,
        "learning_rate": 0.0001,
        "num_train_samples": 10000,
        "num_val_samples": 1000,
    }

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = "/home/runner/work/RefGuidedINR/RefGuidedINR/transformer_results"
    os.makedirs(output_dir, exist_ok=True)

    # Create datasets
    print("\n" + "=" * 60)
    print("Creating Datasets")
    print("=" * 60)
    train_dataset = SequenceCopyDataset(
        num_samples=config["num_train_samples"],
        seq_len=config["seq_len"],
        vocab_size=config["vocab_size"],
        noise_prob=0.0  # No noise for this task
    )
    val_dataset = SequenceCopyDataset(
        num_samples=config["num_val_samples"],
        seq_len=config["seq_len"],
        vocab_size=config["vocab_size"],
        noise_prob=0.0
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                           shuffle=False, collate_fn=collate_fn)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    print("\n" + "=" * 60)
    print("Creating Transformer Model")
    print("=" * 60)
    model = Transformer(
        src_vocab_size=config["vocab_size"],
        tgt_vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"]
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Model configuration:")
    for key, value in config.items():
        if key not in ["num_train_samples", "num_val_samples", "batch_size", "num_epochs", "learning_rate"]:
            print(f"  {key}: {value}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98), eps=1e-9)

    # Training
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0

    start_time = time.time()

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, os.path.join(output_dir, "best_model.pth"))

    training_time = time.time() - start_time

    # Generate examples
    print("\n" + "=" * 60)
    print("Generating Examples")
    print("=" * 60)
    examples = generate_examples(model, val_dataset, device, num_examples=10)

    for i, ex in enumerate(examples):
        print(f"\nExample {i + 1}:")
        print(f"  Input:      {ex['input']}")
        print(f"  Target:     {ex['target']}")
        print(f"  Prediction: {ex['prediction']}")
        print(f"  Match: {'✓' if ex['match'] else '✗'}")

    # Plot training curves
    plot_training_curves(train_losses, train_accs, val_losses, val_accs,
                        os.path.join(output_dir, "training_curves.png"))

    # Save results
    results = {
        "config": config,
        "training_time_seconds": training_time,
        "best_val_accuracy": best_val_acc,
        "final_train_loss": train_losses[-1],
        "final_train_acc": train_accs[-1],
        "final_val_loss": val_losses[-1],
        "final_val_acc": val_accs[-1],
        "model_parameters": count_parameters(model),
        "examples": examples,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs
    }

    save_results(results, os.path.join(output_dir, "results.json"))

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
