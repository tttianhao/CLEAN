import torch
from .utils import * 
from .model import LayerNormNet
from .distance_map import *
from .evaluate import *
from .dataloader import *
import pandas as pd
import warnings

def get_cluster_cen(model_emb_train, model_emb_test,
                      ec_id_dict_train, id_ec_test,
                      device, dtype, dot=False):
    '''
    Get the pair-wise distance map for test queries and train EC cluster centers
    map is of size of (N_test_ids, N_EC_train)
    '''
    print("The embedding sizes for train and test:",
          model_emb_train.size(), model_emb_test.size())
    # get cluster center for all EC appeared in training set
    cluster_center_model = get_cluster_center(
        model_emb_train, ec_id_dict_train)
    return cluster_center_model

def get_dist(max_ec, train_data, report_metrics = False, 
                 pretrained=True, model_name=None, target = 300, neg_target = 2000, negative = None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')

    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet(512, 128, device, dtype)
    
    if pretrained:
        try:
            checkpoint = torch.load('./data/pretrained/'+ train_data +'.pth')
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load('./data/model/'+ model_name +'.pth')
        except FileNotFoundError as error:
            raise Exception('No model found!')
            
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt')
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt')
    else:
        emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
    
    id_ec_test = {}
    for ids in ec_id_dict_train[max_ec]:
        #if len(id_ec_train[ids]) == 1:
        id_ec_test[ids] = max_ec
    
    import random
    
    neg_dict = {}
    counter = 0
    while len(list(neg_dict.keys())) <= neg_target:
        counter += 1
        cur = random.choices(negative[max_ec]['negative'], weights= negative[max_ec]['weights'])[0]
        cur = random.choice(list(ec_id_dict_train[cur]))
        if max_ec not in id_ec_train[cur]:
            neg_dict[cur] = id_ec_train[cur]
        if counter >= 10000:
            break
    
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    ec_centers = get_cluster_cen(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    
    neg_emb_test = model_embedding_test(neg_dict, model, device, dtype)
    
    distances = []
    for i in range(len(emb_test)):
        dist = (emb_test[i] - ec_centers[max_ec].to(device)).norm(dim = 0, p = 2).detach().cpu().numpy().item()
        distances.append(dist)
        
    neg_distances = []
    for i in range(len(neg_emb_test)):
        dist = (neg_emb_test[i] - ec_centers[max_ec].to(device)).norm(dim = 0, p = 2).detach().cpu().numpy().item()
        neg_distances.append(dist)
        
    return distances, neg_distances