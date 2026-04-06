import regex as re
import json
from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(self, 
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        self.pair2rank = {pair: i for i, pair in enumerate(merges)}
        self.vocab_to_id = {v: k for k, v in vocab.items()}
        
        # Add special tokens to vocab if missing
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab_to_id:
                self.vocab[next_id] = token_bytes
                self.vocab_to_id[token_bytes] = next_id
                next_id += 1
        
        self.special_token_set = set(self.special_tokens)
        if self.special_tokens:
            # Sort by length descending to ensure longest match is tried first
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = "|".join(re.escape(t) for t in sorted_special_tokens)
            self.split_pattern = re.compile(f"({self.special_pattern})")
        else:
            self.special_pattern = None
            self.split_pattern = None
            
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
            vocab = {}
            for k, v in raw_vocab.items():
                if isinstance(v, str):
                    vocab[int(k)] = v.encode("utf-8")
                else:
                    vocab[int(k)] = bytes(v)

        with open(merges_filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()

        merges: list[tuple[bytes, bytes]] = []
        if content.startswith("["):
            raw_merges = json.loads(content)
            for pair in raw_merges:
                merges.append((bytes(pair[0]), bytes(pair[1])))
        else:
            for line in content.splitlines():
                parts = line.rstrip().split(" ")
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

        return cls(vocab, merges, special_tokens)
    
    def _merge_tokens(self, tokens: list[bytes]) -> list[bytes]:
        while True:
            # Find the lowest rank pair in the current sequence
            min_rank = float("inf")
            pair_to_merge = None
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.pair2rank.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    pair_to_merge = pair
            
            if pair_to_merge is None:
                break
            
            # Merge all occurrences of the best pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair_to_merge:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        return tokens

    def encode(self, text: str) -> list[int]:
        ids = []
        
        # 1. Split by special tokens
        if self.split_pattern:
            parts = self.split_pattern.split(text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_token_set:
                part_bytes = part.encode("utf-8")
                if part_bytes in self.vocab_to_id:
                     ids.append(self.vocab_to_id[part_bytes])
                continue
            
            if not part:
                continue

            # 2. Regex pre-tokenization
            chunks = [m.group() for m in self.pat.finditer(part)]
            
            for chunk in chunks:
                # Convert to initial byte tokens
                raw_bytes = chunk.encode("utf-8")
                initial_tokens = [bytes([b]) for b in raw_bytes]
                
                # 3. Merge
                merged_tokens = self._merge_tokens(initial_tokens)
                
                # 4. Map to IDs
                for t in merged_tokens:
                    if t in self.vocab_to_id:
                        ids.append(self.vocab_to_id[t])
                    # else: unk token? or skip?
                        
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for id in self.encode(text):
                yield id

    def decode(self, ids: list[int]) -> str:
        # Map IDs back to tokens
        tokens = [self.vocab[id] for id in ids]
        # Concatenate tokens and decode
        byte_string = b"".join(tokens)
        return byte_string.decode("utf-8", errors="ignore")