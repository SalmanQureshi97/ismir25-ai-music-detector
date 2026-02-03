"""
Train logistic regression on artifact fingerprints (fakeprints) and evaluate.

Supports:
- Random split (default)
- FMA split (Table 1): --split dataset_medium_split.npy --synth fp_dac.npy fp_encodec.npy fp_musika.npy --report table1
- SONICS split (Table 2): --split sonics/sonics_split.npy --synth fp_sonics_synth.npy --report table2

Example (Table 1):
  python train_test_regressor.py --real fp_fma_real.npy --synth fp_fma_dac.npy fp_fma_encodec.npy fp_fma_musika.npy --split dataset_medium_split.npy --report table1

Example (Table 2):
  python train_test_regressor.py --real fp_sonics_real.npy --synth fp_sonics_synth.npy --split sonics/sonics_split.npy --report table2
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--real", help="real fakeprints .npy (dict filename -> vector)", type=str, required=True)
parser.add_argument("--synth", nargs="+", help="one or more synth fakeprints .npy (Table 1: dac, encodec, musika)", type=str, default=[])
parser.add_argument("--split", help="split file .npy: FMA has train/validation/test; SONICS has train/test/valid + per-source keys", type=str, default=None)
parser.add_argument("--report", choices=["table1", "table2", "default"], help="table1 = per-codec (FMA), table2 = per-source (SONICS), default = single acc", type=str, default="default")
args = parser.parse_args()

real_db = np.load(args.real, allow_pickle=True).item()
synth_files = args.synth
if not synth_files:
    raise ValueError("Provide at least one --synth .npy file")

# Load all synth dicts; keep names for table1 (from filenames: fp_fma_dac.npy -> dac)
synth_dbs = []
synth_names = []
for p in synth_files:
    d = np.load(p, allow_pickle=True).item()
    name = os.path.splitext(os.path.basename(p))[0]
    if "fma" in name.lower():
        name = name.replace("fp_fma_", "").replace("fp_", "")
    synth_dbs.append(d)
    synth_names.append(name)


def create_train_test_split(db, split_fact=0.8, seed=123):
    np.random.seed(seed)
    keys = np.array(list(db.keys()))
    np.random.shuffle(keys)
    K = int(len(keys) * split_fact)
    return keys[:K], keys[K:]


def is_fma_split(s):
    return "train" in s and "test" in s and ("validation" in s or "valid" in s) and "suno" not in str(s.keys()).lower()


def is_sonics_split(s):
    return "train" in s and "test" in s and ("suno_v3.5" in s or "udio_v120" in s)


use_split = args.split is not None
split = None
if use_split:
    split = np.load(args.split, allow_pickle=True).item()
    if is_sonics_split(split):
        split_kind = "sonics"
    elif is_fma_split(split):
        split_kind = "fma"
    else:
        split_kind = "generic"
else:
    split_kind = None

# Build train/test keys for real and synthetic
if split_kind == "fma":
    train_ids = list(split["train"])
    test_ids = list(split["test"])
    # Restrict to ids present in real and in at least one synth
    all_synth_keys = set()
    for d in synth_dbs:
        all_synth_keys.update(d.keys())
    common_train = [i for i in train_ids if i in real_db and i in all_synth_keys]
    common_test = [i for i in test_ids if i in real_db and i in all_synth_keys]
    train_real_k = common_train
    test_real_k = common_test
    train_synth_k = {name: [i for i in common_train if i in d] for name, d in zip(synth_names, synth_dbs)}
    test_synth_k = {name: [i for i in common_test if i in d] for name, d in zip(synth_names, synth_dbs)}
    # Pool all synth for training
    train_synth_k_pooled = []
    for name, d in zip(synth_names, synth_dbs):
        for i in common_train:
            if i in d:
                train_synth_k_pooled.append((i, name))
    train_synth_keys_pooled = [i for i, _ in train_synth_k_pooled]
elif split_kind == "sonics":
    train_synth_ids = list(split["train"])
    test_synth_ids = list(split["test"])
    real_keys = list(real_db.keys())
    np.random.seed(123)
    np.random.shuffle(real_keys)
    n_train_s = len(train_synth_ids)
    n_test_s = len(test_synth_ids)
    n_real = len(real_keys)
    # Match sizes roughly: use half real for train, half for test
    train_real_k = real_keys[: min(n_real // 2, n_train_s)]
    test_real_k = real_keys[min(n_real // 2, n_train_s) :][: max(n_test_s, 1)]
    # Synth: only one db for SONICS
    d = synth_dbs[0]
    train_synth_keys_pooled = [f for f in train_synth_ids if f in d]
    # Per-source test sets (files in split["test"] that belong to each source)
    source_keys = [k for k in split.keys() if k not in ("train", "test", "valid")]
    test_synth_k = {}
    for sk in source_keys:
        ids = [f for f in split["test"] if f in split.get(sk, [])]
        if ids and synth_dbs:
            test_synth_k[sk] = [f for f in ids if f in synth_dbs[0]]
    train_real_k = train_real_k or list(real_db.keys())[: len(train_synth_keys_pooled)]
    test_real_k = test_real_k or list(real_db.keys())[len(train_real_k) :][: len(test_synth_ids)]
elif split_kind == "generic" and split is not None:
    train_ids = list(split.get("train", []))
    test_ids = list(split.get("test", []))
    all_synth_keys = set()
    for d in synth_dbs:
        all_synth_keys.update(d.keys())
    common_train = [i for i in train_ids if i in real_db and i in all_synth_keys]
    common_test = [i for i in test_ids if i in real_db and i in all_synth_keys]
    train_real_k = common_train
    test_real_k = common_test
    train_synth_k_pooled = []
    for d in synth_dbs:
        for i in common_train:
            if i in d:
                train_synth_k_pooled.append(i)
                break
    train_synth_keys_pooled = list(set(k for d in synth_dbs for k in common_train if k in d))
    test_synth_k = {name: [i for i in common_test if i in d] for name, d in zip(synth_names, synth_dbs)}
else:
    # Random split
    train_real_k, test_real_k = create_train_test_split(real_db)
    # Pool synth: one db or merge all
    if len(synth_dbs) == 1:
        train_synth_k, test_synth_k_list = create_train_test_split(synth_dbs[0])
        train_synth_keys_pooled = train_synth_k
        test_synth_k = {synth_names[0]: test_synth_k_list}
    else:
        train_synth_k = {name: create_train_test_split(d)[0] for name, d in zip(synth_names, synth_dbs)}
        test_synth_k = {name: create_train_test_split(d)[1] for name, d in zip(synth_names, synth_dbs)}
        train_synth_keys_pooled = [k for name in synth_names for k in train_synth_k[name]]

# Build feature matrices
X_r_train = np.stack([real_db[k] for k in train_real_k], 0)
X_r_test = np.stack([real_db[k] for k in test_real_k], 0)

# Pooled synth for training: for FMA use all codecs; for SONICS use single db
if split_kind == "fma":
    X_s_train_list = []
    for name, d in zip(synth_names, synth_dbs):
        for k in train_real_k:
            if k in d:
                X_s_train_list.append(d[k])
    X_s_train = np.stack(X_s_train_list, 0) if X_s_train_list else np.stack([synth_dbs[0][k] for k in train_synth_keys_pooled], 0)
elif split_kind == "sonics":
    X_s_train = np.stack([synth_dbs[0][k] for k in train_synth_keys_pooled], 0)
else:
    if len(synth_dbs) == 1:
        X_s_train = np.stack([synth_dbs[0][k] for k in train_synth_keys_pooled], 0)
    else:
        parts = [np.stack([synth_dbs[i][k] for k in train_synth_k[synth_names[i]]], 0) for i in range(len(synth_dbs))]
        X_s_train = np.concatenate(parts, 0) if parts and all(p.size > 0 for p in parts) else np.stack([synth_dbs[0][k] for k in train_synth_keys_pooled], 0)
    if X_s_train.size == 0:
        X_s_train = np.stack([synth_dbs[0][k] for k in train_synth_keys_pooled], 0)

X = np.concatenate((X_r_train, X_s_train), 0)
Y = np.concatenate((np.zeros(len(X_r_train)), np.ones(len(X_s_train))), 0)

reg = LogisticRegression(class_weight="balanced", max_iter=1000)
reg.fit(X, Y)

real_acc = reg.score(X_r_test, np.zeros(len(X_r_test)))
print("Real class test acc: {:.2f}%, false positive: {:.2f}%".format(real_acc * 100, (1 - real_acc) * 100))

if args.report == "table1":
    print("\nSynthetic (per codec):")
    for name, d in zip(synth_names, synth_dbs):
        test_k = [i for i in test_real_k if i in d]
        if test_k:
            X_s_test = np.stack([d[k] for k in test_k], 0)
            acc = reg.score(X_s_test, np.ones(len(X_s_test)))
            print("  → {}: {:.2f}%".format(name, acc * 100))
elif args.report == "table2":
    print("\nSynthetic (per source):")
    for source, test_ids_s in test_synth_k.items():
        if not test_ids_s or not synth_dbs:
            continue
        X_s_test = np.stack([synth_dbs[0][k] for k in test_ids_s], 0)
        acc = reg.score(X_s_test, np.ones(len(X_s_test)))
        print("  → {}: {:.2f}%".format(source, acc * 100))
else:
    if isinstance(test_synth_k, dict):
        all_test_synth = []
        for name, ids in test_synth_k.items():
            d = synth_dbs[synth_names.index(name)] if name in synth_names else synth_dbs[0]
            for k in ids:
                if k in d:
                    all_test_synth.append(d[k])
        if all_test_synth:
            X_s_test = np.stack(all_test_synth, 0)
            synth_acc = reg.score(X_s_test, np.ones(len(X_s_test)))
            print("Synth class test acc: {:.2f}%, false negative: {:.2f}%".format(synth_acc * 100, (1 - synth_acc) * 100))
        else:
            print("No synthetic test samples found.")
    else:
        X_s_test = np.stack([synth_dbs[0][k] for k in test_synth_k], 0)
        synth_acc = reg.score(X_s_test, np.ones(len(X_s_test)))
        print("Synth class test acc: {:.2f}%, false negative: {:.2f}%".format(synth_acc * 100, (1 - synth_acc) * 100))

# Optional: save weights for visualization (e.g. Figure 5)
# weights = {"W": reg.coef_, "B": reg.intercept_}
