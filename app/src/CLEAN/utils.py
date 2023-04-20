import csv
import random
import os
import math
from re import L
import torch
import numpy as np
import subprocess
import pickle
from .distance_map import get_dist_map

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

def csv_to_fasta(csv_name, fasta_name):
    csvfile = open(csv_name, 'r')
    csvreader = csv.reader(csvfile, delimiter='\t')
    outfile = open(fasta_name, 'w')
    for i, rows in enumerate(csvreader):
        if i > 0:
            outfile.write('>' + rows[0] + '\n')
            outfile.write(rows[2] + '\n')
            
def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def retrive_esm1b_embedding(fasta_name):
    esm_script = "esm/scripts/extract.py"
    esm_out = "data/esm_data"
    esm_type = "esm1b_t33_650M_UR50S"
    fasta_name = "data/" + fasta_name + ".fasta"
    command = ["python", esm_script, esm_type, 
              fasta_name, esm_out, "--include", "mean"]
    subprocess.run(command)
 
def compute_esm_distance(train_file):
    ensure_dirs('./data/distance_map/')
    _, ec_id_dict = get_ec_id_dict('./data/' + train_file + '.csv')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    esm_emb = esm_embedding(ec_id_dict, device, dtype)
    esm_dist = get_dist_map(ec_id_dict, esm_emb, device, dtype)
    pickle.dump(esm_dist, open('./data/distance_map/' + train_file + '.pkl', 'wb'))
    pickle.dump(esm_emb, open('./data/distance_map/' + train_file + '_esm.pkl', 'wb'))
    
def prepare_infer_fasta(fasta_name):
    retrive_esm1b_embedding(fasta_name)
    csvfile = open('./data/' + fasta_name +'.csv', 'w', newline='')
    csvwriter = csv.writer(csvfile, delimiter = '\t')
    csvwriter.writerow(['Entry', 'EC number', 'Sequence'])
    fastafile = open('./data/' + fasta_name +'.fasta', 'r')
    for i in fastafile.readlines():
        if i[0] == '>':
            csvwriter.writerow([i.strip()[1:], ' ', ' '])
    
def mutate(seq: str, position: int) -> str:
    seql = seq[ : position]
    seqr = seq[position+1 : ]
    seq = seql + '*' + seqr
    return seq

def mask_sequences(single_id, csv_name, fasta_name) :
    csv_file = open('./data/'+ csv_name + '.csv')
    csvreader = csv.reader(csv_file, delimiter = '\t')
    output_fasta = open('./data/' + fasta_name + '.fasta','w')
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

def mutate_single_seq_ECs(train_file):
    id_ec, ec_id =  get_ec_id_dict('./data/' + train_file + '.csv')
    single_ec = set()
    for ec in ec_id.keys():
        if len(ec_id[ec]) == 1:
            single_ec.add(ec)
    single_id = set()
    for id in id_ec.keys():
        for ec in id_ec[id]:
            if ec in single_ec and not os.path.exists('./data/esm_data/' + id + '_1.pt'):
                single_id.add(id)
                break
    print("Number of EC numbers with only one sequences:",len(single_ec))
    print("Number of single-seq EC number sequences need to mutate: ",len(single_id))
    print("Number of single-seq EC numbers already mutated: ", len(single_ec) - len(single_id))
    mask_sequences(single_id, train_file, train_file+'_single_seq_ECs')
    fasta_name = train_file+'_single_seq_ECs'
    return fasta_name


