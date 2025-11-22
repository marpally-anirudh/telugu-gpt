"""
Telugu Tokenizer Builder
This script trains a tokenizer specifically for Telugu text
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

def train_telugu_tokenizer(data_files, vocab_size=32000, output_path="data/tokenizer"):
    """
    Train a Byte-Pair Encoding tokenizer for Telugu
    
    Parameters:
    -----------
    data_files : list of str
        Paths to Telugu text files (e.g., ["data/raw/wikipedia.txt"])
    vocab_size : int
        Size of vocabulary (how many unique tokens)
    output_path : str
        Where to save the trained tokenizer
    
    Returns:
    --------
    tokenizer : Tokenizer object
    """
    
    print("🚀 Starting tokenizer training...")
    
    # Step 1: Create a tokenizer with BPE model
    # BPE = Byte Pair Encoding (breaks words into subwords)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))  # <unk> = unknown token
    
    # Step 2: Set up how to split text (by whitespace)
    tokenizer.pre_tokenizer = Whitespace()
    
    # Step 3: Create trainer with special tokens
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<unk>",   # Unknown token
            "<pad>",   # Padding token
            "<bos>",   # Beginning of sequence
            "<eos>"    # End of sequence
        ],
        min_frequency=2  # Only include tokens that appear at least 2 times
    )
    
    # Step 4: Train on your Telugu files
    print(f"📚 Training on {len(data_files)} files...")
    tokenizer.train(files=data_files, trainer=trainer)
    
    # Step 5: Save the tokenizer
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save(f"{output_path}/telugu_tokenizer.json")
    print(f"✅ Tokenizer saved to {output_path}/telugu_tokenizer.json")
    
    # Step 6: Test it
    test_text = "నమస్కారం, మీరు ఎలా ఉన్నారు?"
    encoded = tokenizer.encode(test_text)
    print(f"\n🧪 Test encoding:")
    print(f"Input: {test_text}")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")
    
    return tokenizer


def load_telugu_tokenizer(path="data/tokenizer/telugu_tokenizer.json"):
    """
    Load a previously trained tokenizer
    
    Parameters:
    -----------
    path : str
        Path to saved tokenizer file
    
    Returns:
    --------
    tokenizer : Tokenizer object
    """
    tokenizer = Tokenizer.from_file(path)
    print(f"✅ Tokenizer loaded from {path}")
    return tokenizer


# Example usage:
if __name__ == "__main__":
    import glob
    
    # Automatically find all .txt files in data/raw/
    data_files = glob.glob("data/raw/*.txt")
    
    if not data_files:
        print("❌ No .txt files found in data/raw/")
        print("Please add Telugu text files to: data/raw/")
        print("Example: data/raw/wikipedia.txt")
        exit(1)
    
    print(f"Found {len(data_files)} file(s): {data_files}")
    
    # Train tokenizer
    tokenizer = train_telugu_tokenizer(
        data_files=data_files,
        vocab_size=32000,
        output_path="data/tokenizer"
    )
    
    print("\n✅ Tokenizer training complete!")
