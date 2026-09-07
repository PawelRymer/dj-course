import sys
from pathlib import Path
from tokenizers import Tokenizer
from corpora import get_text_files

VOCAB_SIZES = [16000, 32000, 64000]
TOKENIZER_CUSTOM_BASES = [
    "all", "nkjp", "pan_tadeusz", "wolnelektury"
]

TOKENIZER_ORIGINALS = [ "qwen-v3_8-28b", "bielik-v1", "bielik-v2", "bielik-v3" ]

TOKENIZERS = {
    f"{base}-vs{vs}": f"tokenizers/{base}-vs{vs}-tokenizer.json"
    for base in TOKENIZER_CUSTOM_BASES
    for vs in VOCAB_SIZES
}

TOKENIZERS.update({
    f"{base}": f"tokenizers/{base}-tokenizer.json"
    for base in TOKENIZER_ORIGINALS
})

TEXT_FILES = [
    ['MINIKORPUS', 'Fryderyk Chopin', 'fryderyk-chopin-wikipedia.txt'],
    ['MINIKORPUS', 'The Pickwick Papers', 'the-pickwick-papers-gutenberg.txt'],
    ['WOLNELEKTURY', 'Pan Tadeusz Ksiega 1', 'pan-tadeusz-ksiega-1.txt']
]

def load_corpus_text(corpora_file: str, glob_pattern: str) -> str:
    try:
        files = get_text_files(corpora_file, glob_pattern)
        if not files:
            raise ValueError(f"No files found for corpus '{corpora_file}' with pattern '{glob_pattern}'")
        
        text = ""
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                text += f.read()
        return text
    except ValueError as e:
        raise e
    except Exception as e:
        raise Exception(f"Error loading corpus '{corpora_file}' with pattern '{glob_pattern}': {str(e)}")

def main():
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    tokenizer_stats = {tokenizer_name: 0 for tokenizer_name in TOKENIZERS.keys()}
    corpus_results = {}
    
    for corpora_file, text_name, glob_pattern in TEXT_FILES:
        try:
            print(f"\nLoading corpus '{corpora_file}' with pattern '{glob_pattern}'...")
            source_text = load_corpus_text(corpora_file, glob_pattern)
            print(f"Loaded {len(source_text)} characters")

            fmt_text_name = text_name.replace(' ', '_').lower()
            corpus_results[fmt_text_name] = {}
            
            for tokenizer_name, tokenizer_path in TOKENIZERS.items():
                try:
                    tokenizer = Tokenizer.from_file(tokenizer_path)
                    encoded = tokenizer.encode(source_text)
                    token_count = len(encoded.ids)
                    
                    corpus_results[fmt_text_name][tokenizer_name] = token_count
                    tokenizer_stats[tokenizer_name] += token_count
                    
                    log_file = logs_dir / f"tokenized-{fmt_text_name}-{tokenizer_name}.log"
                    with open(log_file, 'w', encoding='utf-8') as f:
                        f.write(f"Corpus: {corpora_file}\n")
                        f.write(f"Pattern: {glob_pattern}\n")
                        f.write(f"Tokenizer: {tokenizer_name}\n")
                        f.write(f"Liczba tokenów: {token_count}\n")
                    
                    print(f"  {tokenizer_name}: {token_count} tokens -> {log_file}")
                    
                except Exception as e:
                    print(f"Error tokenizing with '{tokenizer_name}': {str(e)}", file=sys.stderr)
                    continue
        
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)
    
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    
    for text_name, results in corpus_results.items():
        print(f"\n{text_name}:")
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        for tokenizer_name, token_count in sorted_results:
            print(f"  {tokenizer_name}: {token_count} tokens")
    
    print("\n" + "="*60)
    print("OVERALL RANKING (total tokens across all texts):")
    print("="*60)
    sorted_stats = sorted(tokenizer_stats.items(), key=lambda x: x[1])
    for tokenizer_name, total_tokens in sorted_stats:
        print(f"{tokenizer_name}: {total_tokens} tokens")
    
    best_tokenizer, best_count = sorted_stats[0]
    print(f"\nMost efficient tokenizer: {best_tokenizer} ({best_count} tokens)")
    
    stats_file = logs_dir / "tokenizer-statistics.log"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("OVERALL RANKING (total tokens across all texts):\n")
        f.write("="*60 + "\n")
        for tokenizer_name, total_tokens in sorted_stats:
            f.write(f"{tokenizer_name}: {total_tokens} tokens\n")
        f.write(f"\nMost efficient tokenizer: {best_tokenizer} ({best_count} tokens)\n")
        f.write("\nPER-CORPUS DETAILS:\n")
        f.write("="*60 + "\n")
        for text_name, results in corpus_results.items():
            f.write(f"\n{text_name}:\n")
            sorted_results = sorted(results.items(), key=lambda x: x[1])
            for tokenizer_name, token_count in sorted_results:
                f.write(f"  {tokenizer_name}: {token_count} tokens\n")

if __name__ == "__main__":
    main()
