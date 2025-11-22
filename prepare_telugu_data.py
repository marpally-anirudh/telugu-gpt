
import os
import glob

# Read all .txt files from data/raw/
all_texts = []
data_dir = "data/raw"

print("Reading Telugu text files...")
for txt_file in glob.glob(os.path.join(data_dir, "*.txt")):
    print(f"  Reading: {txt_file}")
    with open(txt_file, 'r', encoding='utf-8') as f:
        text = f.read()
        all_texts.append(text)
        print(f"    âœ… {len(text):,} characters")

# Combine all
full_corpus = "\n\n".join(all_texts)

# Save
output_file = "data/processed/telugu_corpus.txt"
os.makedirs("data/processed", exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(full_corpus)

print(f"\nâœ… Complete! Saved to: {output_file}")
print(f"ðŸ“Š Total size: {len(full_corpus):,} characters")
