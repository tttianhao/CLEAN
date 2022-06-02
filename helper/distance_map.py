import torch
from tqdm import tqdm


def get_dist_map(ec_id_dict, esm_emb, device, dtype, model=None):
    cluster_center_model = {}
    # inference all queries at once to get model embedding
    if model is not None:
        model_emb = model(esm_emb.to(device=device, dtype=dtype))
    else:
        # the first distance map before training comes from ESM
        model_emb = esm_emb
    # calculate cluster center by averaging all embeddings in one EC
    id_counter = 0
    with torch.no_grad():
        for ec in tqdm(list(ec_id_dict.keys())):
            ids_for_query = list(ec_id_dict[ec])
            id_counter_prime = id_counter + len(ids_for_query)
            emb_cluster = model_emb[id_counter: id_counter_prime]
            cluster_center = emb_cluster.mean(dim=0)
            cluster_center_model[ec] = cluster_center.detach().cpu()
            id_counter = id_counter_prime

    # calculate distance map between total_ec_n * total_ec_n pairs
    out_dim = model_emb.size(1)
    total_ec_n = len(ec_id_dict.keys())
    model_lookup = torch.randn(total_ec_n, out_dim)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device)
    model_dist = {}
    print(f'Calculating distance map, number of unique EC is {total_ec_n}')
    for i, ec in tqdm(enumerate(ecs)):
        current = model_lookup[i]
        current = current.repeat(total_ec_n, 1)
        current = current.to(device)
        norm_esm = (current - model_lookup).norm(dim=1, p=2)
        norm_esm = norm_esm.detach().cpu().numpy()
        model_dist[ec] = {}
        for j, k in enumerate(ecs):
            model_dist[ec][k] = norm_esm[j]
    return model_dist
 