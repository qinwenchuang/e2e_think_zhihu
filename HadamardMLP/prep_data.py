import csv
import json
import os
from collections import Counter

import numpy as np

DATA_CSV = "data/dataset.csv"
OUT_DIR = "data"
VOCAB_CAP = 8192
MIN_ROW_CHARS = 30
VAL_FRAC = 0.10
TEST_FRAC = 0.10
SEED = 42


def load_texts(path):
    texts = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            for field in row:
                field = field.strip()
                if len(field) >= MIN_ROW_CHARS:
                    texts.append(field)
    return texts


def main():
    texts = load_texts(DATA_CSV)
    print(f"rows kept: {len(texts)}, chars: {sum(len(t) for t in texts):,}")

    counts = Counter()
    for t in texts:
        counts.update(t)
    chars = [c for c, n in counts.most_common() if n >= 2][:VOCAB_CAP - 1]
    itos = ["<unk>"] + chars
    stoi = {c: i for i, c in enumerate(chars, start=1)}
    vocab_size = len(itos)
    print(f"vocab_size={vocab_size}, id0='{itos[0]}', id1='{itos[1]}' (freq {counts[itos[1]]})")

    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(texts))
    texts = [texts[i] for i in order]

    all_ids = []
    for t in texts:
        all_ids.append(np.array([stoi.get(c, 0) for c in t], dtype=np.uint16))
    total_tokens = sum(len(a) for a in all_ids)
    n_val = max(1, int(total_tokens * VAL_FRAC))
    n_test = max(1, int(total_tokens * TEST_FRAC))

    val_parts, test_parts, train_parts = [], [], []
    val_so_far = test_so_far = 0
    for a in all_ids:
        if val_so_far < n_val:
            take = min(len(a), n_val - val_so_far)
            val_parts.append(a[:take])
            rest = a[take:]
            val_so_far += take
            if rest.size and test_so_far < n_test:
                take2 = min(len(rest), n_test - test_so_far)
                test_parts.append(rest[:take2])
                test_so_far += take2
                if take2 < len(rest):
                    train_parts.append(rest[take2:])
            elif rest.size:
                train_parts.append(rest)
        elif test_so_far < n_test:
            take = min(len(a), n_test - test_so_far)
            test_parts.append(a[:take])
            test_so_far += take
            if take < len(a):
                train_parts.append(a[take:])
        else:
            train_parts.append(a)

    train_ids = np.concatenate(train_parts)
    val_ids = np.concatenate(val_parts)
    test_ids = np.concatenate(test_parts)
    train_ids.tofile(os.path.join(OUT_DIR, "train.bin"))
    val_ids.tofile(os.path.join(OUT_DIR, "val.bin"))
    test_ids.tofile(os.path.join(OUT_DIR, "test.bin"))

    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "test_tokens": int(len(test_ids)),
        "source": DATA_CSV,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"train_tokens={len(train_ids):,}  val_tokens={len(val_ids):,}  "
          f"test_tokens={len(test_ids):,}")
    print("wrote data/train.bin, data/val.bin, data/test.bin, data/meta.json")


if __name__ == "__main__":
    main()
