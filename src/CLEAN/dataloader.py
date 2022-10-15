import torch
import random
from .utils import format_esm


def mine_hard_negative(dist_map, knn=10):
    #print("The number of unique EC numbers: ", len(dist_map.keys()))
    ecs = list(dist_map.keys())
    negative = {}
    for i, target in enumerate(ecs):
        sort_orders = sorted(
            dist_map[target].items(), key=lambda x: x[1], reverse=False)
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
            'weights': normalized_freq,
            'negative': neg_ecs
        }
    return negative


def mine_negative(anchor, id_ec, ec_id, mine_neg):
    anchor_ec = id_ec[anchor]
    pos_ec = random.choice(anchor_ec)
    neg_ec = mine_neg[pos_ec]['negative']
    weights = mine_neg[pos_ec]['weights']
    result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
    while result_ec in anchor_ec:
        result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
    neg_id = random.choice(ec_id[result_ec])
    return neg_id


def random_positive(id, id_ec, ec_id):
    pos_ec = random.choice(id_ec[id])
    pos = id
    if len(ec_id[pos_ec]) == 1:
        return pos + '_' + str(random.randint(0, 9))
    while pos == id:
        pos = random.choice(ec_id[pos_ec])
    return pos


class Triplet_dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.full_list = []
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        pos = random_positive(anchor, self.id_ec, self.ec_id)
        neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
        a = torch.load('./data/esm_data/' + anchor + '.pt')
        p = torch.load('./data/esm_data/' + pos + '.pt')
        n = torch.load('./data/esm_data/' + neg + '.pt')
        return format_esm(a), format_esm(p), format_esm(n)


class MultiPosNeg_dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg, n_pos, n_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.full_list = []
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        a = format_esm(torch.load('./data/esm_data/' +
                       anchor + '.pt')).unsqueeze(0)
        data = [a]
        for _ in range(self.n_pos):
            pos = random_positive(anchor, self.id_ec, self.ec_id)
            p = format_esm(torch.load('./data/esm_data/' +
                           pos + '.pt')).unsqueeze(0)
            data.append(p)
        for _ in range(self.n_neg):
            neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
            n = format_esm(torch.load('./data/esm_data/' +
                           neg + '.pt')).unsqueeze(0)
            data.append(n)
        return torch.cat(data)
