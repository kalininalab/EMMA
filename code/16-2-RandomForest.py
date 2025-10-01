import pickle
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import numpy as np

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
    
    (train_inter_X, train_inter_y), (train_sub_X, train_sub_y) = prep_df(train)
    (test_inter_X, test_inter_y), (test_sub_X, test_sub_y) = prep_df(test)

    print("Training RF on interaction data           ")
    inter_rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=n_jobs)
    inter_rf.fit(train_inter_X, train_inter_y)
    inter_pred = inter_rf.predict(test_inter_X)
    
    print("\nTraining RF on subclass data            ")
    sub_rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=n_jobs)
    sub_rf.fit(train_sub_X, train_sub_y)
    sub_pred = sub_rf.predict(test_sub_X)
    
    with open(f"rf_{split}_{seed}_preds.pkl", "wb") as f:
        pickle.dump((inter_pred, sub_pred), f)


if __name__ == "__main__":
    for split in ["R", "C1", "C1e", "C1f", "C2"]:
        for seed in [42, 123, 456, 789, 999]:
            print(f"=== Split: {split}, Seed: {seed} ===")
            rf_split(split, seed, n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else None)
            print()

