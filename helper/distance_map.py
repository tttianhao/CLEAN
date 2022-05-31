import csv
import pickle
import torch
import time
import argparse
import pandas as pd

def get_dist_map(device, model, out_dim, file_name):
    from helper.utils import get_ec_id_dict
    from helper.dataloader import Dataset_lookup
    
    params = {'batch_size': 30000,
    'shuffle': False,
    'num_workers': 16}

    counter = 0
    cluster_center_esm = {}
    with torch.no_grad():
        _, ec_id_dict = get_ec_id_dict('./data/' + file_name + '.csv')
        time_start = time.time()
        total_ec_n = len(ec_id_dict.keys())
        for ec in list(ec_id_dict.keys()):
            if ec.count('-') == 0:
                counter += 1
                valid_data = Dataset_lookup(list(ec_id_dict[ec]))
                valid_loader = torch.utils.data.DataLoader(valid_data, **params)
                for i, data in enumerate(valid_loader):
                    lookup = data.to(device)
                    lookup = model(lookup)
                    esm = torch.mean(lookup, dim = 0)
                    cluster_center_esm[ec] = esm.detach().cpu()
                if counter % 100 == 0:
                    elapsed = time.time() - time_start
                    time_start = time.time()
                    print(f'|Processing {counter:5d} out of {total_ec_n} |Time elapsed: {elapsed:5.2f} |')
        
        esm_lookup = torch.randn(total_ec_n, out_dim)
        ecs = list(cluster_center_esm.keys())
        for i, ec in enumerate(ecs):
            esm_lookup[i] = cluster_center_esm[ec]
        esm_lookup = esm_lookup.to(device)
        esm_dist = {}
        time_start = time.time()
        for i,ec in enumerate(ecs):
            current = esm_lookup[i]
            current = current.repeat(total_ec_n, 1)
            current = current.to(device)
            norm_esm = (current - esm_lookup).norm(dim = 1, p = 2)
            norm_esm = norm_esm.detach().cpu().numpy()
            esm_dist[ec] = {}
            for j, k in enumerate(ecs):
                esm_dist[ec][k] = norm_esm[j]
            if i % 100 == 0:
                elapsed = time.time() - time_start
                time_start = time.time()
                print(f'|Processing {i:5d} out of {total_ec_n} |Time elapsed: {elapsed:5.2f} |')
        return esm_dist

def hyperbolic_dist(u, v):
    sqdist = torch.sum((u - v) ** 2, dim=-1)
    squnorm = torch.sum(u ** 2, dim=-1)
    sqvnorm = torch.sum(v ** 2, dim=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-7
    z = torch.sqrt(x ** 2 - 1)
    return torch.log(x + z)

if __name__ == '__main__':
    from utils import get_ec_id_dict
    from dataloader import Dataset_lookup
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file',type = str)
    parser.add_argument('--hyper',type = bool, default = False)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('device used: ', device)
    torch.backends.cudnn.benchmark = True

    params = {'batch_size': 30000,
    'shuffle': False,
    'num_workers': 16}

    counter = 0
    cluster_center_esm = {}

    with torch.no_grad():
        _, ec_id_dict = get_ec_id_dict('./data/' + args.train_file +'.csv')
        time_start = time.time()
        total_ec_n = len(ec_id_dict.keys())
        for ec in ec_id_dict.keys():
            if ec.count('-') == 0:
                counter += 1
                valid_data = Dataset_lookup(list(ec_id_dict[ec]))
                valid_loader= torch.utils.data.DataLoader(valid_data, **params)
                for i, data in enumerate(valid_loader):
                    lookup = data.to(device)
                    esm = torch.mean(lookup, dim = 0)
                    cluster_center_esm[ec] = esm.detach().cpu()
                if counter % 100 == 0:
                    elapsed = time.time() - time_start
                    time_start = time.time()
                    print(f'|Processing {counter:5d} out of {total_ec_n} |Time elapsed: {elapsed:5.2f} |')
        
        esm_lookup = torch.randn(total_ec_n, 1280)
        ecs = list(cluster_center_esm.keys())
        for i, ec in enumerate(ecs):
            esm_lookup[i] = cluster_center_esm[ec]
        esm_lookup = esm_lookup.to(device)
        esm_dist = {}
        time_start = time.time()
        for i,ec in enumerate(ecs):
            current = esm_lookup[i]
            current = current.repeat(total_ec_n, 1)
            current = current.to(device)
            if args.hyper:
                norm_esm = hyperbolic_dist(current, esm_lookup)
            else:
                norm_esm = (current - esm_lookup).norm(dim = 1, p = 2)
            norm_esm = norm_esm.detach().cpu().numpy()
            esm_dist[ec] = {}
            for j, k in enumerate(ecs):
                esm_dist[ec][k] = norm_esm[j]
            if i % 100 == 0:
                elapsed = time.time() - time_start
                time_start = time.time()
                print(f'|Processing {i:5d} out of {total_ec_n} |Time elapsed: {elapsed:5.2f} |')
        esm_df = pd.DataFrame.from_dict(esm_dist)
        pickle.dump(esm_dist, open('../data/distance_map/' + args.train_file + '.pkl','wb'))
        esm_df.to_csv('./data/distance_map/' + args.train_file + '_df.csv')