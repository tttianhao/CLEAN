import torch
import random

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
        return pos + '_' + str(random.randint(0,9))
    while pos == id:
        pos = random.choice(ec_id[pos_ec])
    return pos

def change_format(a):
    if type(a) == dict:
        a = a['mean_representations'][33]
    return a

class Dataset_lookup(torch.utils.data.Dataset):
    def __init__(self, lookup_list):
        self.lookup_list = lookup_list
    
    def __len__(self):
        return len(self.lookup_list)
    
    def __getitem__(self,index):
        lookup = self.lookup_list[index]
        a = torch.load('./data/esm_data/'+ lookup + '.pt')
        return change_format(a).double()
    
class Dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, full_list, mine_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.full_list = []
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)
    
    def __len__(self):
        return len(self.full_list)
    
    def __getitem__(self,index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        pos = random_positive(anchor, self.id_ec, self.ec_id)
        neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
        a = torch.load('./data/esm_data/' + anchor + '.pt')
        p = torch.load('./data/esm_data/' + pos + '.pt')
        n = torch.load('./data/esm_data/' + neg + '.pt')
        return change_format(a).double(), change_format(p).double(), change_format(n).double()