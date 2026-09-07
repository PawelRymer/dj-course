from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from corpora import CORPORA_FILES

pre_tokenizer = Whitespace()

VOCAB_SIZES = [16000, 32000, 64000]

test_texts = [
    "Litwo! Ojczyzno moja! ty jesteś jak zdrowie.",
    "Jakże mi wesoło!",
    "Jeśli wolisz mieć pełną kontrolę nad tym, które listy są łączone (a to jest bezpieczniejsze, gdy słownik może zawierać inne klucze), po prostu prześlij listę list do spłaszczenia.",
]

for corpus_name, files in CORPORA_FILES.items():
    if not files:
        print(f"Skipping {corpus_name}: no files found")
        continue
    
    print(f"\n{'='*60}")
    print(f"Training tokenizer for: {corpus_name}")
    print(f"Number of files: {len(files)}")
    print(f"{'='*60}")
    
    for vocab_size in VOCAB_SIZES:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        
        tokenizer.pre_tokenizer = pre_tokenizer
        
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            vocab_size=vocab_size,
            min_frequency=2
        )
        
        FILES = [str(f) for f in files]
        print(f"\nTraining with vocab_size={vocab_size}")
        print(f"Files: {FILES[:3]}{'...' if len(FILES) > 3 else ''}")
        
        tokenizer.train(FILES, trainer=trainer)
        
        output_file = f"tokenizers/{corpus_name.lower()}-vs{vocab_size}-tokenizer.json"
        tokenizer.save(output_file)
        print(f"Saved tokenizer to: {output_file}")
        
        print(f"Testing tokenizer on {corpus_name} (vocab_size={vocab_size}):")
        for txt in test_texts:
            encoded = tokenizer.encode(txt)
            print(f"  Text: {txt[:50]}...")
            print(f"  Tokens: {encoded.tokens[:10]}{'...' if len(encoded.tokens) > 10 else ''}")
            print(f"  IDs: {encoded.ids[:10]}{'...' if len(encoded.ids) > 10 else ''}")

