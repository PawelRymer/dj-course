from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from corpora import CORPORA_FILES

# 2. Set the pre-tokenizer (e.g., split on spaces)
pre_tokenizer = Whitespace()

# 3. Set the Trainer
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=32000,
    min_frequency=2
)

# Test texts for validation
test_texts = [
    "Litwo! Ojczyzno moja! ty jesteś jak zdrowie.",
    "Jakże mi wesoło!",
    "Jeśli wolisz mieć pełną kontrolę nad tym, które listy są łączone (a to jest bezpieczniejsze, gdy słownik może zawierać inne klucze), po prostu prześlij listę list do spłaszczenia.",
]

# Loop over all corpora and train tokenizers
for corpus_name, files in CORPORA_FILES.items():
    if not files:
        print(f"Skipping {corpus_name}: no files found")
        continue
    
    print(f"\n{'='*60}")
    print(f"Training tokenizer for: {corpus_name}")
    print(f"Number of files: {len(files)}")
    print(f"{'='*60}")
    
    # 1. Initialize the Tokenizer (BPE model)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # 2. Set the pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizer
    
    # Convert files to strings
    FILES = [str(f) for f in files]
    print(f"Files: {FILES[:3]}{'...' if len(FILES) > 3 else ''}")
    
    # 4. Train the Tokenizer
    tokenizer.train(FILES, trainer=trainer)
    
    # 5. Save the vocabulary and tokenization rules
    output_file = f"tokenizers/{corpus_name.lower()}-tokenizer.json"
    tokenizer.save(output_file)
    print(f"Saved tokenizer to: {output_file}")
    
    # 6. Test the tokenizer with sample texts
    print(f"\nTesting tokenizer on {corpus_name}:")
    for txt in test_texts:
        encoded = tokenizer.encode(txt)
        print(f"  Text: {txt[:50]}...")
        print(f"  Tokens: {encoded.tokens[:10]}{'...' if len(encoded.tokens) > 10 else ''}")
        print(f"  IDs: {encoded.ids[:10]}{'...' if len(encoded.ids) > 10 else ''}")
        print()
