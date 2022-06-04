import torch
from tqdm import tqdm


def get_cluster_center(model_emb, ec_id_dict):
    cluster_center_model = {}
    id_counter = 0
    with torch.no_grad():
        for ec in tqdm(list(ec_id_dict.keys())):
            ids_for_query = list(ec_id_dict[ec])
            id_counter_prime = id_counter + len(ids_for_query)
            emb_cluster = model_emb[id_counter: id_counter_prime]
            cluster_center = emb_cluster.mean(dim=0)
            cluster_center_model[ec] = cluster_center.detach().cpu()
            id_counter = id_counter_prime
    return cluster_center_model


def get_dist_map(ec_id_dict, esm_emb, device, dtype, model=None):
    '''
    Get the distance map for training, size of (N_EC_train, N_EC_train)
    between all possible pairs of EC cluster centers
    '''
    # inference all queries at once to get model embedding
    if model is not None:
        model_emb = model(esm_emb.to(device=device, dtype=dtype))
    else:
        # the first distance map before training comes from ESM
        model_emb = esm_emb
    # calculate cluster center by averaging all embeddings in one EC
    cluster_center_model = get_cluster_center(model_emb, ec_id_dict)
    # organize cluster centers in a matrix
    total_ec_n, out_dim = len(ec_id_dict.keys()), model_emb.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device, dtype=dtype)
    # calculate pairwise distance map between total_ec_n * total_ec_n pairs
    model_dist = {}
    print(f'Calculating distance map, number of unique EC is {total_ec_n}')
    for i, ec in tqdm(enumerate(ecs)):
        current = model_lookup[i].unsqueeze(0)
        dist_norm = (current - model_lookup).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        model_dist[ec] = {}
        for j, ec_prime in enumerate(ecs):
            model_dist[ec][ec_prime] = dist_norm[j]
    return model_dist


def get_dist_map_test(model_emb_train, model_emb_test,
                      ec_id_dict_train, id_ec_test,
                      device, dtype):
    '''
    Get the pair-wise distance map for test queries and train EC cluster centers
    map is of size of (N_test_ids, N_EC_train)
    '''
    print("The embedding sizes for train and test:",
          model_emb_train.size(), model_emb_test.size())
    # get cluster center for all EC appeared in training set
    cluster_center_model = get_cluster_center(
        model_emb_train, ec_id_dict_train)
    total_ec_n, out_dim = len(ec_id_dict_train.keys()), model_emb_train.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device, dtype=dtype)
    # calculate distance map between n_query_test * total_ec_n (training) pairs
    query_dist = {}
    ids = list(id_ec_test.keys())
    print(f'Calculating eval distance map, between {len(ids)} test ids '
          f'and {total_ec_n} train EC cluster centers')
    eval_dist = {}
    for i, id in tqdm(enumerate(ids)):
        id_embedding = model_emb_test[i]
        dist_norm = (id_embedding - model_lookup).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        eval_dist[id] = {}
        for j, ec in enumerate(ecs):
            eval_dist[id][ec] = dist_norm[j]
    return eval_dist        
    
