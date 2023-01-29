import torch
from src.CLEAN.utils import * 
from src.CLEAN.model import LayerNormNet
from src.CLEAN.distance_map import *
from src.CLEAN.evaluate import *
from src.CLEAN.dataloader import mine_hard_negative
from src.CLEAN.infer import infer_maxsep
from src.CLEAN.uncertainty import get_dist
from sklearn import mixture
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

train_data = "split100"

id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')

counter = 0

dist_map = pickle.load(open('./data/distance_map/split100.pkl', 'rb'))
negative = mine_hard_negative(dist_map, 5)

for i in range(40):
    
    counter = 0
    
    all_distance = []

    for ec in random.choices(list(ec_id_dict_train.keys()), k = 500):

        distances, neg_distances = get_dist(ec, train_data,
                report_metrics=True, pretrained=True, neg_target = 100, negative = negative)
        all_distance.extend(neg_distances)
        all_distance.extend(distances)
        
        if counter % 100 == 0:
            print(counter)
            
        counter += 1
        
    dist = np.reshape(all_distance, (len(all_distance), 1))
    main_GMM = mixture.GaussianMixture(n_components=2, covariance_type='full',max_iter=1000,n_init=30,tol=1e-4)
    main_GMM.fit(dist)

    pickle.dump(main_GMM, open('./gmm_test/GMM_100_500_' + str(i) + '.pkl', 'wb'))

    plt.hist(all_distance, bins = 500, alpha = 0.05)
    plt.savefig('./gmm_test/GMM_100_500_' + str(i) + '.png')