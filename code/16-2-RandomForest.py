import itertools
import pickle
import random
import sys
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import os
from os.path import join
current_dir = os.path.dirname(os.path.abspath(__file__))

# Binding == 0: inhibitor
# Binding == 1: substrate
# Binding == 2: non-interacting


def prep_df(df):
    inter_X, inter_y, sub_X, sub_y = [], [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        print(f"\r{i}", "/", len(df), end="\r")
        match row["Binding"]:
            case 0:
                inter_X.append(np.concatenate((row["ESM2t30"], row["MolFormer"])))
                inter_y.append(1)
                sub_X.append(np.concatenate((row["ESM2t30"], row["MolFormer"])))
                sub_y.append(0)
            case 1:
                inter_X.append(np.concatenate((row["ESM2t30"], row["MolFormer"])))
                inter_y.append(1)
                sub_X.append(np.concatenate((row["ESM2t30"], row["MolFormer"])))
                sub_y.append(1)
            case 2:
                inter_X.append(np.concatenate((row["ESM2t30"], row["MolFormer"])))
                inter_y.append(0)
            case _:
                raise ValueError("Invalid binding type: " + str(row))
    print("Done loading")
    return (np.array(inter_X), np.array(inter_y)), (np.array(sub_X), np.array(sub_y))


def rf_split(split, seed, n_jobs=None):
    with open(f"../data/splits/train_{split}_2S.pkl", "rb") as f:
        train = pickle.load(f)
    with open(f"../data/splits/test_{split}_2S.pkl", "rb") as f:
        test = pickle.load(f)
    with open("rf_hps.pkl", "rb") as f:
        hps = pickle.load(f)

    (train_inter_X, train_inter_y), (train_sub_X, train_sub_y) = prep_df(train)
    (test_inter_X, test_inter_y), (test_sub_X, test_sub_y) = prep_df(test)
    print("Training RF on interaction data")
    inter_rf = RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **hps[(split, "inter", str(seed))])
    inter_rf.fit(train_inter_X, train_inter_y)
    inter_pred = inter_rf.predict_proba(test_inter_X)[:, 1]

    print("\nTraining RF on subclass data")
    sub_rf = RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **hps[(split, "sub", str(seed))])
    sub_rf.fit(train_sub_X, train_sub_y)
    sub_pred = sub_rf.predict_proba(test_sub_X)[:, 1]
    results_path=join(current_dir,"..","data", "training_test_results","Random_Forest")
    os.makedirs(results_path, exist_ok=True)
    np.save(join(results_path, f"interaction_y_test_pred_{split}_RS{seed}.npy"), inter_pred)
    np.save(join(results_path, f"interaction_y_test_true_{split}_RS{seed}.npy"), test_inter_y)
    np.save(join(results_path, f"subclass_y_test_pred_{split}_RS{seed}.npy"), sub_pred)
    np.save(join(results_path, f"subclass_y_test_true_{split}_RS{seed}.npy"), test_sub_y)


def tune_rf(split, seed, candidates=3, head="inter"):
    with open(f"../data/splits/train_{split}_2S.pkl", "rb") as f:
        train = pickle.load(f)
    
    df_train, df_val = train_test_split(train,
                                        test_size=0.2,
                                        random_state=42,
                                        stratify=train['Binding'])

    (train_inter_X, train_inter_y), (train_sub_X, train_sub_y) = prep_df(df_train)
    (val_inter_X, val_inter_y), (val_sub_X, val_sub_y) = prep_df(df_val)

    if head == "sub":
        train_X, train_y, val_X, val_y = train_sub_X, train_sub_y, val_sub_X, val_sub_y
    else:
        train_X, train_y, val_X, val_y = train_inter_X, train_inter_y, val_inter_X, val_inter_y

    search_space = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [4, 6, 8, 10, 12],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [4, 6, 8, 10],
        'max_samples': [0.2, 0.4, 0.6, 0.8, 1.0],
    }

    combs = list(itertools.product(
        search_space['n_estimators'],
        search_space['max_depth'],
        search_space['min_samples_split'],
        search_space['min_samples_leaf'],
        search_space['max_samples']
    ))
    random.shuffle(combs)
    print(len(combs))
    results = []
    for n_estimators, max_depth, min_samples_split, min_samples_leaf, max_samples in combs[:candidates]:
        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    max_samples=max_samples,
                                    random_state=42,
                                    n_jobs=-1)
        rf.fit(train_X, train_y)
        val_pred = rf.predict(val_X)
        auroc = roc_auc_score(val_y, val_pred)
        results.append((auroc, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_samples))
        print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, "
              f"min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, "
              f"max_samples: {max_samples} => AUROC: {auroc}")
    pd.DataFrame(results, columns=['AUROC', 'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_samples']) \
        .sort_values(by='AUROC', ascending=False) \
        .to_csv(f"rf_hyperparam_tuning_{split}_{head}_{seed}.csv", index=False)


if __name__ == "__main__":
    for split in ["R", "C1", "C1e", "C1f", "C2"]:
        for seed in [42, 123, 456, 789, 999]:
            # print(f"=== Split: {split}, Seed: {seed} ===")
            # tune_rf(split, seed, candidates=150, head="inter")
            # tune_rf(split, seed, candidates=150, head="sub")
            # continue
            rf_split(split, seed, n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else None)
            print()
