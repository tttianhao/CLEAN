# Warning: this script is meant to be ran in the root directory!!!
import sys
sys.path.append('./')
import csv
import random
import math
import os
import torch
import numpy as np
from helper.utils import *

def mutate(seq: str, position: int) -> str:
    seql = seq[ : position]
    seqr = seq[position+1 : ]
    seq = seql + '*' + seqr
    return seq

def mask_sequences(single_id: list, csv_name: str, fasta_name: str) -> None:
    csv_file = open('./data/'+ csv_name + '.csv')
    csvreader = csv.reader(csv_file, delimiter = '\t')
    output_fasta = open('./data/fasta/' + fasta_name + '.fasta','w')
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

def main():
    #convert_dict()
    id_ec, ec_id = get_ec_id_dict('./data/uniref30/uniref30_train_split_4.csv')
    single_id = set()
    single_ec = set()
    for ec in ec_id.keys():
        if len(ec_id[ec]) == 1:
            single_ec.add(ec)
    for id in id_ec.keys():
        for ec in id_ec[id]:
            if ec in single_ec and not os.path.exists('./data/esm_data/' + id + '_2.pt'):
                single_id.add(id)
                break
    print(len(single_id), len(single_ec))
    mask_sequences(single_id, 'uniref30/uniref30_train_split_4', 'esm')

if __name__ == '__main__':
    main()
