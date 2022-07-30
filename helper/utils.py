import csv
import random
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm


def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_ec_id_dict(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            id_ec[rows[0]] = rows[1].split(';')
            for ec in rows[1].split(';'):
                if ec not in ec_id.keys():
                    ec_id[ec] = set()
                    ec_id[ec].add(rows[0])
                else:
                    ec_id[ec].add(rows[0])
    return id_ec, ec_id

def get_ec_id_dict_non_prom(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            if len(rows[1].split(';')) == 1:
                id_ec[rows[0]] = rows[1].split(';')
                for ec in rows[1].split(';'):
                    if ec not in ec_id.keys():
                        ec_id[ec] = set()
                        ec_id[ec].add(rows[0])
                    else:
                        ec_id[ec].add(rows[0])
    return id_ec, ec_id

# def get_ec_id_dict_single_ec(csv_name: str) -> dict:
#     csv_file = open(csv_name)
#     csvreader = csv.reader(csv_file, delimiter='\t')
#     id_ec = {}
#     ec_id = {}

#     for i, rows in enumerate(csvreader):
#         if i > 0:
#             if len(rows[1].split(';')) == 1:
#                 id_ec[rows[0]] = rows[1].split(';')
#                 for ec in rows[1].split(';'):
#                     if ec not in ec_id.keys():
#                         ec_id[ec] = set()
#                         ec_id[ec].add(rows[0])
#                     else:
#                         ec_id[ec].add(rows[0])

#     csv_file = open(csv_name)
#     csvreader = csv.reader(csv_file, delimiter='\t')
    
#     for i, rows in enumerate(csvreader):
#         if i > 0:
#             if len(rows[1].split(';')) > 1:
#                 id_ec[rows[0]] = rows[1].split(';')
#                 for ec in rows[1].split(';'):
#                     if ec not in ec_id.keys():
#                         ec_id[ec] = set()
#                         ec_id[ec].add(rows[0])
                     
#     return id_ec, ec_id

def format_esm(a):
    if type(a) == dict:
        a = a['mean_representations'][33]
    return a


def load_esm(lookup):
    esm = format_esm(torch.load('./data/esm_data/' + lookup + '.pt'))
    return esm.unsqueeze(0)


def esm_embedding(ec_id_dict, device, dtype):
    '''
    Loading esm embedding in the sequence of EC numbers
    prepare for calculating cluster center by EC
    '''
    esm_emb = []
    # for ec in tqdm(list(ec_id_dict.keys())):
    for ec in list(ec_id_dict.keys()):
        ids_for_query = list(ec_id_dict[ec])
        esm_to_cat = [load_esm(id) for id in ids_for_query]
        esm_emb = esm_emb + esm_to_cat
    return torch.cat(esm_emb).to(device=device, dtype=dtype)


def model_embedding_test(id_ec_test, model, device, dtype):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    then inferenced with model to get model embedding
    '''
    ids_for_query = list(id_ec_test.keys())
    esm_to_cat = [load_esm(id) for id in ids_for_query]
    esm_emb = torch.cat(esm_to_cat).to(device=device, dtype=dtype)
    model_emb = model(esm_emb)
    return model_emb

def model_embedding_test_ensemble(id_ec_test, device, dtype):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    '''
    ids_for_query = list(id_ec_test.keys())
    esm_to_cat = [load_esm(id) for id in ids_for_query]
    esm_emb = torch.cat(esm_to_cat).to(device=device, dtype=dtype)
    return esm_emb
