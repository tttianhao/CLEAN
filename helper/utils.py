import csv
import random
import argparse
import math
import pickle
import os
import torch
import numpy as np

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-e', '--epoch', type=int, default=10000)
    parser.add_argument('-n', '--model_name', type=str,
                        default='default_model')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-k', '--knn', type=int, default=10)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('-b', '--batch_size', type=int, default=2720)
    parser.add_argument('-c', '--check_point', type=str, default='no')
    parser.add_argument('-m', '--margin', type=float, default=1)
    parser.add_argument('--adaptive_rate', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--hyperbolic', type=bool, default=False)
    parser.add_argument('--high_precision', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    return args


def get_ec_id_dict(csv_name : str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter = '\t')
    id_ec = {}
    ec_id = {}

    for i,rows in enumerate(csvreader):
        if i > 0:
            id_ec[rows[0]] = rows[1].split(';')
            for ec in rows[1].split(';'):
                if ec not in ec_id.keys():
                    ec_id[ec] = set()
                    ec_id[ec].add(rows[0])
                else:
                    ec_id[ec].add(rows[0])
    return id_ec, ec_id

def mine_hard_negative(dist_map, knn = 10):
    print("The number of unique EC numbers: ", len(dist_map.keys()))
    ecs = list(dist_map.keys())
    negative = {}
    for i, target in enumerate(ecs):
        sort_orders = sorted(dist_map[target].items(), key=lambda x: x[1], reverse=False)
        if sort_orders[1][1] != 0:
            freq = [1/i[1] for i in sort_orders[1:1 + knn]]
            neg_ecs = [i[0] for i in sort_orders[1:1 + knn]]
        elif sort_orders[2][1] != 0:
            freq = [1/i[1] for i in sort_orders[2:2+knn]]
            neg_ecs = [i[0] for i in sort_orders[2:2+knn]]
        elif sort_orders[3][1] != 0:
            freq = [1/i[1] for i in sort_orders[3:3+knn]]
            neg_ecs = [i[0] for i in sort_orders[3:3+knn]]
        else:
            freq = [1/i[1] for i in sort_orders[4:4+knn]]
            neg_ecs = [i[0] for i in sort_orders[4:4+knn]]

        normalized_freq = [i/sum(freq) for i in freq]
        negative[target] = {
            'weights' : normalized_freq,
            'negative' : neg_ecs
        }
    return negative

def mutate(seq: str, position: int) -> str:
    seql = seq[ : position]
    seqr = seq[position+1 : ]
    seq = seql + '*' + seqr
    return seq

def mask_sequences(single_id: list, csv_name: str, fasta_name: str) -> None:
    csv_file = open('../data/'+ csv_name + '.csv')
    csvreader = csv.reader(csv_file, delimiter = '\t')
    output_fasta = open('../data/' + fasta_name + '.fasta','w')
    single_id = set(single_id)
    for i, rows in enumerate(csvreader):
        if rows[0] in single_id:
            for j in range(10):
                seq = rows[2].strip()
                mu, sigma = .10, .02 # mean and standard deviation
                s = np.random.normal(mu, sigma, 1)
                mut_rate = s[0]
                times = math.ceil(len(seq) * mut_rate)
                for k in range(times):
                    position = random.randint(1 , len(seq) - 1)
                    seq = mutate(seq, position)
                seq = seq.replace('*', '<mask>')
                output_fasta.write('>' + rows[0] + '_' + str(j) + '\n')
                output_fasta.write(seq + '\n')

def convert_dict():
    dir_path = '/home/tianhao/project/EC_pred/esm_data/'
    counter = 0
    for files in os.listdir(dir_path):
        counter += 1
        if files.endswith('.pt'):
            esm = torch.load(dir_path + files)
            if type(esm) == dict:
                esm = esm['mean_representations'][33]
                assert(type(esm) == torch.Tensor)
                torch.save(esm, dir_path + files)
        if counter % 100000 == 0:
            print(counter)  

if __name__ == '__main__':
    #convert_dict()
    id_ec, ec_id = get_ec_id_dict('../data/uniref10_train_split_0.csv')
    single_id = set()
    single_ec = set()
    for ec in ec_id.keys():
        if len(ec_id[ec]) == 1:
            single_ec.add(ec)
    for id in id_ec.keys():
        for ec in id_ec[id]:
            if ec in single_ec:
                single_id.add(id)
                break
    print(len(single_id), len(single_ec))
    mask_sequences(single_id, 'uniref10_train_split_0', 'to_be_embed_0')