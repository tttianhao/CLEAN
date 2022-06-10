# Warning: this script is meant to be ran in the root directory!!!
import sys
sys.path.append('./')
import csv
import numpy as np
from helper.utils import get_ec_id_dict
from sklearn.model_selection import KFold


def curate_testset():
    uniref = [100]
    splits = [0,1,2,3,4]

    for u in uniref:
        for s in splits:
            _, ec_id_dict = get_ec_id_dict('./data/' + 'uniref' + str(u) + '/uniref' + str(u) + '_train_split_' + str(s) +'.csv')
            test_file = open('./data/' + 'uniref' + str(u) + '/uniref' + str(u) + '_test_split_' + str(s) +'.csv')
            csvreader = csv.reader(test_file, delimiter = '\t')
            exclude_list = set()
            ids = []
            ecn = []
            seqs = []
            for i, rows in enumerate(csvreader):
                if i > 0:
                    ids.append(rows[0])
                    ecn.append(rows[1])
                    seqs.append(rows[2])
                    ecs = rows[1]
                    for ec in ecs.split(';'):
                        if ec not in ec_id_dict.keys():
                            exclude_list.add(rows[0])
            outfile = open('./data/' + 'uniref' + str(u) + '/uniref' + str(u) + '_test_split_' + str(s) +'_curate.csv','w')
            csvwriter = csv.writer(outfile, delimiter = '\t')
            csvwriter.writerow(['ID', 'EC','Sequences'])
            for i in range(len(ids)):
                if ids[i] not in exclude_list:
                    csvwriter.writerow([ids[i], ecn[i], seqs[i]])
            print(len(exclude_list), i)


def cv5_split(s, test_index):
    full_seq = open('./data/train_ec5238_seq227358.csv')
    csvreader = csv.reader(full_seq, delimiter = '\t')
    train_file = open('./data/uniref100/uniref100_train_split_' + str(s) + '.csv' , 'w')
    trainwriter = csv.writer(train_file, delimiter = '\t')
    test_file = open('./data/uniref100/uniref100_test_split_' + str(s) + '.csv' , 'w')
    testwriter = csv.writer(test_file, delimiter = '\t')
    for i, rows in enumerate(csvreader):
        if i > 0:
            if i-1 in test_index:
                testwriter.writerow(rows)
            else:
                trainwriter.writerow(rows)
        else:
            testwriter.writerow(rows)
            trainwriter.writerow(rows)

if __name__ == '__main__':
    curate_testset()
    # dummy = np.zeros(227362)
    # kf = KFold(n_splits=5, shuffle = True, random_state = 1234)
    # counter = 0
    # for train_index, test_index in kf.split(dummy):
    #     cv5_split(counter, set(test_index))
    #     counter += 1