# this script is meant to be ran in the root directory!!!
import sys
sys.path.append('./')
import pickle
import torch
import argparse
import pandas as pd
from helper.utils import *
from helper.distance_map import *

def get_esm_dist():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_file', type=str)
    args = parser.parse_args()
    _, ec_id_dict = get_ec_id_dict('./data/' + args.train_file + '.csv')
    # use cpu and high precision by default
    device, dtype = torch.device("cpu"), torch.float64
    esm_emb = esm_embedding(ec_id_dict, device, dtype)
    esm_dist = get_dist_map(ec_id_dict, esm_emb, device, dtype)
    esm_df = pd.DataFrame.from_dict(esm_dist)
    pickle.dump(esm_dist, open('./data/distance_map/' +
                args.train_file + '.pkl', 'wb'))
    esm_df.to_csv('./data/distance_map/' + args.train_file + '_df.csv')


if __name__ == '__main__':
    get_esm_dist()
