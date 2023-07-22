import torch
import torch.nn.functional as F
 
def SupConHardLoss(model_emb, temp, n_pos):
    '''
    return the SupCon-Hard loss
    features:  
        model output embedding, dimension [bsz, n_all, out_dim], 
        where bsz is batchsize, 
        n_all is anchor, pos, neg (n_all = 1 + n_pos + n_neg)
        and out_dim is embedding dimension
    temp:
        temperature     
    n_pos:
        number of positive examples per anchor
    '''
    # l2 normalize every embedding
    features = F.normalize(model_emb, dim=-1, p=2)
    # features_T is [bsz, outdim, n_all], for performing batch dot product
    features_T = torch.transpose(features, 1, 2)
    # anchor is the first embedding 
    anchor = features[:, 0]
    # anchor is the first embedding 
    anchor_dot_features = torch.bmm(anchor.unsqueeze(1), features_T)/temp 
    # anchor_dot_features now [bsz, n_all], contains 
    anchor_dot_features = anchor_dot_features.squeeze(1)
    # deduct by max logits, which will be 1/temp since features are L2 normalized 
    logits = anchor_dot_features - 1/temp
    # the exp(z_i dot z_a) excludes the dot product between itself
    # exp_logits is of size [bsz, n_pos+n_neg]
    exp_logits = torch.exp(logits[:, 1:])
    exp_logits_sum = n_pos * torch.log(exp_logits.sum(1)) # size [bsz], scale by n_pos
    pos_logits_sum = logits[:, 1:n_pos+1].sum(1) #sum over all (anchor dot pos)
    log_prob = (pos_logits_sum - exp_logits_sum)/n_pos
    loss = - log_prob.mean()
    return loss    
