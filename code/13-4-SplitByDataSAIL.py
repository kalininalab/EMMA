import numpy as np
import pandas as pd
import os
import torch
from os.path import join
import sys
import time
import pickle
import argparse
sys.path.append("./../utilities")
from helper_functions import *
CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)


def main(args):
    start_time = time.time()
    input_path = args.input_path
    split_method = args.split_method
    output_path = args.output_path
    split_size = args.split_size
    strat = args.strat
    epsilon = args.epsilon_value
    delta = args.delta_value

    if len(split_size) not in [2, 3]:

        raise ValueError("The split-size argument must be a list of either two or three integers.")
    split_path = join(output_path, f"{split_method}_epsilon{epsilon}_delta{delta}_{len(split_size)}S")
    os.makedirs(split_path, exist_ok=True)
    log_file = os.path.join(split_path,
                            f"Report_{split_method}_ep{epsilon}del{delta}_{len(split_size)}S.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    setup_logging(log_file)
    logging.info(f"Current Directory: {CURRENT_DIR}")
    data = pd.read_pickle(join(input_path))
    logging.info(
        "*** Start running the DataSAIL***\nFor more information about DataSAIL please check it's webpage: "
        "https://datasail.readthedocs.io/en/latest/index.html")
    train_output_file = os.path.join(split_path, f"train_{split_method}_{len(split_size)}S.pkl")
    test_output_file = os.path.join(split_path,
                                    f"test_{split_method}_{len(split_size)}S.pkl")
    val_output_file = os.path.join(split_path, f"val_{split_method}_{len(split_size)}S.pkl")
    final_data_output_file = os.path.join(split_path, f"Final_Dataset_{split_method}_{len(split_size)}S.pkl")
    data['ids'] = ['ID' + str(index) for index in data.index]
    e_splits, f_splits, inter_sp = datasail_wrapper(split_method, data, split_size, stratification=strat,
                                                       epsilon=epsilon, delta=delta)
    data['split'] = np.nan

    if split_method in ["C1e", "I1e"]:
        for key in e_splits.keys():
            data['split'].fillna(data['ids'].map(e_splits[key][0]), inplace=True)
        with open(os.path.join(split_path, f"datasail_output_e_splits_{split_method}.pkl"), 'wb') as f:
            pickle.dump(e_splits, f)
    elif split_method in ["C1f", "I1f"]:
        for key in f_splits.keys():
            data['split'].fillna(data['ids'].map(f_splits[key][0]), inplace=True)
        with open(os.path.join(split_path, f"datasail_output_f_splits_{split_method}.pkl"), 'wb') as f:
            pickle.dump(f_splits, f)
    elif split_method in ["C2","I2"]:
        for key in inter_sp.keys():
            inter_dict = {k[0]: v for k, v in inter_sp[key][0].items()}
            data['split'].fillna(data['ids'].map(inter_dict), inplace=True)
        with open(os.path.join(split_path, f"datasail_output_inter_sp_{split_method}.pkl"), 'wb') as f:
            pickle.dump(inter_sp, f)
    data_filtered = data[(data['split'] == "train") | (data['split'] == "test") | (data['split'] == "val")]
    data_filtered.reset_index(drop=True, inplace=True)
    train = data_filtered[data_filtered["split"] == "train"]
    train.reset_index(drop=True, inplace=True)
    test = data_filtered[data_filtered["split"] == "test"]
    test.reset_index(drop=True, inplace=True)
    val = None
    if len(split_size) == 3:
        val = data_filtered[data_filtered["split"] == "val"]
        val.reset_index(drop=True, inplace=True)
    logging.info("DataSAIL split the data successfully")
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    train.to_pickle(train_output_file)
    test.to_pickle(test_output_file)
    if len(split_size) == 2:
        Final_dataset = pd.concat([train, test], axis=0)
        Final_dataset.reset_index(drop=True, inplace=True)
        Final_dataset.to_pickle(final_data_output_file)
    elif len(split_size) == 3:
        Final_dataset = pd.concat([train, test, val], axis=0)
        val.to_pickle(val_output_file)
        Final_dataset.reset_index(drop=True, inplace=True)
        Final_dataset.to_pickle(final_data_output_file)

    # Report
    if len(split_size) == 2:
        result, total_samples, test_ratio = two_split_report(train, test)
        logging.info(
            f"Data report after splitting data by {split_method} split method and check for NaN or null cells in the "
            f"data\n{result.to_string()}")
        logging.info(f"Total number of samples: {total_samples}")
        logging.info(f"Total number of deleted samples: {len(data) - total_samples}")
        if strat:
            logging.info(f"Delta value : {delta}")
        logging.info(f"Epsilon value : {epsilon}")
        logging.info(f"Requested Split size : {split_size}")
        logging.info(f"Ratio of test set to dataset: {test_ratio}")
        logging.info(
            f"Ratio of inh  to sub in train: {round(len(train.loc[train['Binding'] == 0]) / len(train.loc[train['Binding'] == 1]),3)}")
        logging.info(
            f"Ratio of inh  to sub in test: {round(len(test.loc[test['Binding'] == 0]) / len(test.loc[test['Binding'] == 1]),3)}")
        logging.info(
            f"Ratio of inh  to ni in train: {round(len(train.loc[train['Binding'] == 0]) / len(train.loc[train['Binding'] == 2]),3)}")
        logging.info(
            f"Ratio of inh  to ni in test: {round(len(test.loc[test['Binding'] == 0]) / len(test.loc[test['Binding'] == 2]),3)}")

        logging.info(
            f"Ratio of sub  to ni in train: {round(len(train.loc[train['Binding'] == 1]) / len(train.loc[train['Binding'] == 2]),3)}")
        logging.info(
            f"Ratio of sub  to ni in test: {round(len(test.loc[test['Binding'] == 1]) / len(test.loc[test['Binding'] == 2]),3)}")
        end_time = time.time()
        total_time = (end_time - start_time) / 60
        logging.info(f"Time taken for the process: {total_time:.2f} minutes")
    elif len(split_size) == 3:
        result, total_samples, test_ratio, val_ratio = three_split_report(train, test, val)
        logging.info(
            f"Data report after splitting data by {split_method} split method and check for NaN or null cells in the "
            f"data\n{result.to_string()}")
        logging.info(f"Total number of samples: {total_samples}")
        logging.info(f"Total number of deleted samples: {len(data) - total_samples}")
        if strat:
            logging.info(f"Delta value : {delta}")
        logging.info(f"Epsilon value : {epsilon}")
        logging.info(f"Requested Split size : {split_size}")
        logging.info(f"Ratio of test set to dataset: {test_ratio}")
        logging.info(f"Ratio of val set to dataset: {val_ratio}")
        logging.info(
            f"Ratio of inh  to sub in train: {round(len(train.loc[train['Binding'] == 0]) / len(train.loc[train['Binding'] == 1]),3)}")
        logging.info(
            f"Ratio of inh  to sub in test: {round(len(test.loc[test['Binding'] == 0]) / len(test.loc[test['Binding'] == 1]),3)}")
        logging.info(
            f"Ratio of inh  to ni in train: {round(len(train.loc[train['Binding'] == 0]) / len(train.loc[train['Binding'] == 2]),3)}")
        logging.info(
            f"Ratio of inh  to ni in test: {round(len(test.loc[test['Binding'] == 0]) / len(test.loc[test['Binding'] == 2]),3)}")
        logging.info(
            f"Ratio of sub  to ni in train: {round(len(train.loc[train['Binding'] == 1]) / len(train.loc[train['Binding'] == 2]),3)}")
        logging.info(
            f"Ratio of sub  to ni in test: {round(len(test.loc[test['Binding'] == 1]) / len(test.loc[test['Binding'] == 2]),3)}")
        logging.info(
            f"Ratio of inh  to sub in val: {round(len(val.loc[val['Binding'] == 0]) / len(val.loc[val['Binding'] == 1]),3)}")
        logging.info(
            f"Ratio of inh  to ni in val: {round(len(val.loc[val['Binding'] == 0]) / len(val.loc[val['Binding'] == 2]),3)}")
        logging.info(
            f"Ratio of sub  to ni in val: {round(len(val.loc[val['Binding'] == 1]) / len(val.loc[val['Binding'] == 2]),3)}")
        end_time = time.time()
        total_time = (end_time - start_time) / 60
        logging.info(f"Time taken for the process: {total_time:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Please check the DataSAL webpage: https://datasail.readthedocs.io/en/latest/index.html")
    parser.add_argument('--split-method', type=str, required=True,
                        help="The split method should be one of [C2, I2, C1e, C1f, I1e, I1f]")
    parser.add_argument('--split-size', type=int, nargs='+', required=True,
                        help="Ordered integers determine the size of each split, e.g., 8 2 or 7 2 1 correspond to size of train test or train test val respectively")
    parser.add_argument("--strat", type=str, choices=['True', 'False'], required=True, default=False,
                        help="If True, use stratification method to split the data.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input data (pickle file).")
    parser.add_argument("--output-path", type=str, required=True, help="Directory where the splits will be saved.")
    parser.add_argument('--epsilon-value', type=float, required=True,
                        help="A multiplicative factor by how much the limits (as defined in the -s / –splits argument defined) of the splits can be exceeded.")
    parser.add_argument('--delta-value', type=float, required=True,
                        help="A multiplicative factor by how much the limits (as defined in the -s / –splits argument defined) of the stratification can be exceeded.")
    args = parser.parse_args()
    args.strat = args.strat == 'True'
    main(args)
