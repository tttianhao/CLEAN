import csv
from align_curate import get_alin_result
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score, roc_auc_score, accuracy_score,balanced_accuracy_score, f1_score
import argparse
from utils import get_ec_id_dict

#input csv file and calculate prescion recall

def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', type = str)
    parser.add_argument('-n', '--knn', type = int, default = 1)
    parser.add_argument('--predict', type = bool, default = False)
    parser.add_argument('-t', '--task', type = str)
    parser.add_argument('-l', '--true_label', type = str)
    parser.add_argument('-s', '--subset', type = str, default = None)
    args = parser.parse_args()
    return args


def get_prediction_result(file_name: str, knn: int = 1, subset = None):
    if subset is None:
        entries = []
        pred = []
        result = open(file_name+'.csv','r')
        csvreader = csv.reader(result, delimiter = '|')
        for rows in csvreader:
            entries.append(rows[0])
            cur = []
            for i in range(1,knn+1):
                try:
                    id, ecs = rows[i].split(':')
                    for ec in ecs.split(';'):
                        cur.append(ec)
                except:
                    pass
            pred.append(cur)
        return entries, pred
    else:
        print('subset')
        sub_file = open(subset + '.csv','r')
        subreader = csv.reader(sub_file, delimiter = '\t')
        entries = []
        for i, rows in enumerate(subreader):
            if i > 0:
                entries.append(rows[0])
        entries_set = set(entries)
        pred = []
        result = open(file_name+'.csv','r')
        csvreader = csv.reader(result, delimiter = '|')
        for rows in csvreader:
            if rows[0] in entries_set:
                cur = []
                for i in range(1,knn+1):
                    try:
                        id, ecs = rows[i].split(':')
                        for ec in ecs.split(';'):
                            cur.append(ec)
                    except:
                        pass
                pred.append(cur)
        return entries, pred

def get_true_labels(file_name: str, entries: list):
    all_label = set()
    true_label_dict = {}
    result = open(file_name+'.csv','r')
    csvreader = csv.reader(result, delimiter = '\t')
    entries_set = set(entries)
    for rows in csvreader:
        if rows[0] in entries_set:
            true_label_dict[rows[0]] = rows[1].split(';')
            for ec in rows[1].split(';'):
                all_label.add(ec)
    true_label = []
    for i in entries:
        true_label.append(true_label_dict[i])
    return true_label, all_label

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore') 
    args = add_parser()
    if args.predict:
        entries, pred = get_alin_result('mmseqs')
        if args.task == 'new':
            true_label, all_label = get_true_labels('./data/new_ec177_seq392', entries)
        elif args.task == 'price':
            true_label, all_label = get_true_labels('./data/price', entries)
        elif args.task == 'uniref':
            true_label, all_label = get_true_labels('./data/test_uniref30_over20', entries)

    else:
        entries, pred = get_prediction_result(args.file_name, args.knn, args.subset)
        if args.task == 'new':
            true_label, all_label = get_true_labels('./data/new_ec177_seq392', entries)
        elif args.task == 'price':
            true_label, all_label = get_true_labels('./data/price', entries)
        else:
            true_label, all_label = get_true_labels( args.true_label, entries)

    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_label)])

    pred_m = np.zeros((len(pred), len(mlb.classes_)))
    true_m = np.zeros((len(pred), len(mlb.classes_)))
    # for i,id in enumerate(entries):
    #     if pred[i] != true_label[i]:
    #         print(id, pred[i],true_label[i])
    counter = 0 
    for i,label in enumerate(pred):
        pred_m[i] = mlb.transform([label])
        true_m[i] = mlb.transform([true_label[i]])
        if all(pred_m[i] == true_m[i]):
            counter += 1
    print(counter)

    pre = precision_score(true_m, pred_m, average='macro', zero_division=0)
    rec = recall_score(true_m, pred_m, average='macro')
    f1 = f1_score(true_m, pred_m, average='macro')
    roc = roc_auc_score(true_m, pred_m, average='macro')
    a = accuracy_score(true_m, pred_m)
    #avg_a = balanced_accuracy_score(true_m, pred_m)
    print(f'total samples: {len(true_m)} | total ec: {len(all_label)} | precision: {pre:.3} | recall: {rec:.3}\n'
          f'F1: {f1:.3} | AUC: {roc:.3} | accuracy: {a:.3}')