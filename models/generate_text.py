import torch
from tokenizers import Tokenizer
from gpt_model import GPTModel, TELUGU_GPT_CONFIG

# Load saved model
checkpoint = torch.load("outputs/checkpoints/model_epoch_10.pt")

# Create model
model = GPTModel(TELUGU_GPT_CONFIG)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = Tokenizer.from_file("data/tokenizer/telugu_tokenizer.json")

# Generate text
def generate(start_text, max_tokens=50):
    encoded = tokenizer.encode(start_text).ids
    encoded = torch.tensor(encoded).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(encoded)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            encoded = torch.cat([encoded, next_token], dim=1)
    
    return tokenizer.decode(encoded[0].tolist())

# Test
print(generate("తెలుగులో"))