from __future__ import annotations

import heapq
import multiprocessing
import os
from collections import Counter, defaultdict

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenization(args: tuple[str, list[str]]) -> Counter[bytes]:
    chunk, special_tokens = args
    pattern = "|".join(re.escape(token) for token in special_tokens)
    chunks = re.split(pattern=pattern, string=chunk) if pattern else [chunk]
    counts: Counter[bytes] = Counter()
    for item in chunks:
        for match in re.finditer(GPT2_PAT, item):
            token = match.group()
            counts.update([token.encode("utf-8")])
    return counts


class MaxHeapEntry:
    __slots__ = ("sort_key", "pair")

    def __init__(
        self,
        count: int,
        p0_b: bytes,
        p1_b: bytes,
        pair: tuple[int, int],
    ) -> None:
        self.sort_key = (count, p0_b, p1_b)
        self.pair = pair

    def __lt__(self, other: "MaxHeapEntry") -> bool:
        return self.sort_key > other.sort_key


def _chunk_generator(
    path: str | os.PathLike,
    boundaries: list[int],
    specials: list[str],
):
    with open(path, "rb") as f:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            yield (chunk, specials)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_processes = int(kwargs.get("num_processes", 50))
    imap_chunksize = int(kwargs.get("imap_chunksize", 1))
    if num_processes <= 0:
        raise ValueError("num_processes must be positive")
    if imap_chunksize <= 0:
        raise ValueError("imap_chunksize must be positive")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    global_counts: Counter[bytes] = Counter()
    with multiprocessing.Pool(processes=num_processes) as pool:
        counts = pool.imap(
            pre_tokenization,
            _chunk_generator(input_path, boundaries, special_tokens),
            chunksize=imap_chunksize,
        )
        for item in counts:
            global_counts.update(item)

    vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    num_merges = vocab_size - len(vocab)

    all_words_bytes = list(global_counts.keys())
    ids_list = [list(b) for b in all_words_bytes]
    word_counts = list(global_counts.values())

    stats: Counter[tuple[int, int]] = Counter()
    pair_indices: dict[tuple[int, int], set[int]] = defaultdict(set)

    for word_idx, (tokens, freq) in enumerate(zip(ids_list, word_counts)):
        for pair in zip(tokens[:-1], tokens[1:]):
            stats[pair] += freq
            pair_indices[pair].add(word_idx)

    pq: list[MaxHeapEntry] = []
    for pair, count in stats.items():
        heapq.heappush(pq, MaxHeapEntry(count, vocab[pair[0]], vocab[pair[1]], pair))

    for _ in range(num_merges):
        best_pair = None
        while pq:
            entry = heapq.heappop(pq)
            if stats.get(entry.pair, -1) == entry.sort_key[0]:
                best_pair = entry.pair
                break

        if best_pair is None:
            break

        p0, p1 = best_pair
        new_id = len(vocab)
        merges.append((vocab[p0], vocab[p1]))
        vocab[new_id] = vocab[p0] + vocab[p1]

        words_to_update = list(pair_indices[best_pair])
        del stats[best_pair]
        del pair_indices[best_pair]

        touched_pairs: set[tuple[int, int]] = set()

        for word_idx in words_to_update:
            word = ids_list[word_idx]
            freq = word_counts[word_idx]

            new_word: list[int] = []
            idx = 0
            while idx < len(word):
                if idx < len(word) - 1 and word[idx] == p0 and word[idx + 1] == p1:
                    new_word.append(new_id)
                    idx += 2
                else:
                    new_word.append(word[idx])
                    idx += 1

            for p in zip(word[:-1], word[1:]):
                if p == best_pair:
                    continue
                stats[p] -= freq
                touched_pairs.add(p)
                if stats[p] == 0:
                    del stats[p]
                if p in pair_indices:
                    pair_indices[p].discard(word_idx)
                    if not pair_indices[p]:
                        del pair_indices[p]

            for p in zip(new_word[:-1], new_word[1:]):
                stats[p] += freq
                touched_pairs.add(p)
                pair_indices[p].add(word_idx)

            ids_list[word_idx] = new_word

        for p in touched_pairs:
            if p in stats:
                heapq.heappush(pq, MaxHeapEntry(stats[p], vocab[p[0]], vocab[p[1]], p))

    return vocab, merges
