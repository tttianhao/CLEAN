# Warning: this script is meant to be ran in the root directory!!!
import sys
sys.path.append('./')
import csv
from helper.utils import get_ec_id_dict

uniref = [10,30,50]
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