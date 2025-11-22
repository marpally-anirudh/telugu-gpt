"""
Training Script for Telugu GPT
This script trains the model on your Telugu text data
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from gpt_model import GPTModel, TELUGU_GPT_CONFIG
from tqdm import tqdm
import os

class TeluguTextDataset(Dataset):
    """
    Dataset class for Telugu text
    Converts text into sequences the model can learn from
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        # Tokenize all text
        token_ids = tokenizer.encode(txt).ids
        
        # Create overlapping sequences
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, tokenizer, batch_size=4, max_length=256, 
                      stride=128, shuffle=True):
    """
    Create DataLoader for training
    
    Parameters:
    -----------
    txt : str
        Telugu text to train on
    tokenizer : Tokenizer
        Telugu tokenizer
    batch_size : int
        Number of sequences per batch
    max_length : int
        Length of each sequence
    stride : int
        Step size for creating sequences
    
    Returns:
    --------
    dataloader : DataLoader
    """
    dataset = TeluguTextDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )
    return dataloader


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate loss for one batch
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),  # Flatten to (batch*seq_len, vocab_size)
        target_batch.flatten()  # Flatten to (batch*seq_len)
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate average loss over entire dataset
    """
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    
    return total_loss / num_batches


def train_model(model, train_loader, val_loader, optimizer, device, 
                num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """
    Main training loop
    
    Parameters:
    -----------
    model : GPTModel
        Your Telugu GPT model
    train_loader : DataLoader
        Training data
    val_loader : DataLoader
        Validation data
    optimizer : torch.optim
        Optimizer (e.g., AdamW)
    device : str
        'cuda' or 'cpu'
    num_epochs : int
        Number of training epochs
    eval_freq : int
        Evaluate every N steps
    eval_iter : int
        Number of batches to use for evaluation
    start_context : str
        Telugu text to test generation
    tokenizer : Tokenizer
        Telugu tokenizer
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    
    print(f"\n🚀 Starting training on {device}...")
    print(f"📊 Total training batches: {len(train_loader)}")
    
    for epoch in range(num_epochs):
        model.train()  # Set to training mode
        
        # Progress bar for this epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for input_batch, target_batch in pbar:
            optimizer.zero_grad()  # Reset gradients
            
            # Calculate loss
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            # Backpropagation
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
            
            # Periodic evaluation
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"\nEp {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
                # Test text generation
                print("📝 Sample generation:")
                print(generate_text(model, tokenizer, device, start_context, max_new_tokens=50))
                print()
        
        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1] if train_losses else None,
            'val_loss': val_losses[-1] if val_losses else None,
        }, f"outputs/checkpoints/model_epoch_{epoch+1}.pt")
        print(f"✅ Checkpoint saved: model_epoch_{epoch+1}.pt")
    
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate model on train and validation sets
    """
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_text(model, tokenizer, device, start_context, max_new_tokens=50):
    """
    Generate Telugu text from the model
    
    Parameters:
    -----------
    model : GPTModel
        Trained model
    tokenizer : Tokenizer
        Telugu tokenizer
    device : str
        'cuda' or 'cpu'
    start_context : str
        Telugu text to start from
    max_new_tokens : int
        How many tokens to generate
    
    Returns:
    --------
    generated_text : str
        Generated Telugu text
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    
    # Encode starting text
    encoded = tokenizer.encode(start_context).ids
    encoded = torch.tensor(encoded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = encoded[:, -context_size:]
            
            # Get predictions
            logits = model(idx_cond)
            logits = logits[:, -1, :]  # Focus on last position
            
            # Sample next token
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            
            # Append to sequence
            encoded = torch.cat((encoded, idx_next), dim=1)
    
    # Decode to text
    decoded = tokenizer.decode(encoded.squeeze(0).tolist())
    model.train()
    return decoded


# Main training script
if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("data/tokenizer/telugu_tokenizer.json")
    
    # Load Telugu text
    with open("data/processed/telugu_corpus.txt", "r", encoding="utf-8") as f:
        text_data = f.read()
    
    # Split into train/val
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_data, tokenizer,
        batch_size=4,
        max_length=256,
        stride=128
    )
    
    val_loader = create_dataloader(
        val_data, tokenizer,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=False
    )
    
    print(f"✅ Data loaded: {len(train_data):,} chars train, {len(val_data):,} chars val")
    
    # Create model
    model = GPTModel(TELUGU_GPT_CONFIG)
    model.to(device)
    print(f"✅ Model moved to {device}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    
    # Create output directories
    os.makedirs("outputs/checkpoints", exist_ok=True)
    
    # Train!
    train_losses, val_losses, tokens_seen = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=10,  # Start small for testing
        eval_freq=1,
        eval_iter=1,
        start_context="నమస్కారం",  # Telugu greeting
        tokenizer=tokenizer
    )
    
    print("\n✅ Training complete!")
    print(f"📊 Final train loss: {train_losses[-1]:.3f}")
    print(f"📊 Final val loss: {val_losses[-1]:.3f}")
