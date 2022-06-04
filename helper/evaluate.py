import csv
from helper.utils import *
from helper.distance_map import *
from helper.evaluate import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, average_precision_score, roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score


def maximum_separation(dist_lst, first_grad, use_max_grad):
    opt = 0 if first_grad else -1
    gamma = np.append(dist_lst[1:], np.repeat(dist_lst[-1], 10))
    sep_lst = np.abs(dist_lst - np.mean(gamma))
    sep_grad = np.abs(sep_lst[:-1]-sep_lst[1:])
    if use_max_grad:
        # max separation index determined by largest grad
        max_sep_i = np.argmax(sep_grad)
    else:
        # max separation index determined by first or the last grad
        large_grads = np.where(sep_grad > np.mean(sep_grad))
        max_sep_i = large_grads[-1][opt]
    # if no large grad is found, just call first EC
    if max_sep_i >= 5:
        max_sep_i = 0
    return max_sep_i


def write_max_sep_choices(df, csv_name, first_grad=True, use_max_grad=False):
    out_file = open(csv_name + '_maxsep.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    all_test_EC = set()
    for col in df.columns:
        ec = []
        smallest_10_dist_df = df[col].nsmallest(10)
        dist_lst = list(smallest_10_dist_df)
        max_sep_i = maximum_separation(dist_lst, first_grad, use_max_grad)
        for i in range(max_sep_i+1):
            EC_i = smallest_10_dist_df.index[i]
            dist_i = smallest_10_dist_df[i]
            dist_str = "{:.4f}".format(dist_i)
            all_test_EC.add(EC_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        ec.insert(0, col)
        csvwriter.writerow(ec)
    return


def write_random_nk_choices(df, csv_name, random_nk_dist_map, p_value=0.05):
    out_file = open(csv_name + '_randnk.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    all_test_EC = set()
    nk = len(random_nk_dist_map.keys())
    threshold = p_value*nk
    for col in tqdm(df.columns):
        ec = []
        smallest_10_dist_df = df[col].nsmallest(10)
        for i in range(10):
            EC_i = smallest_10_dist_df.index[i]
            # find all the distances in the random nk w.r.t. EC_i
            # then sorted the nk distances
            rand_nk_dists = [random_nk_dist_map[rand_nk_id][EC_i]
                             for rand_nk_id in random_nk_dist_map.keys()]
            rand_nk_dists = np.sort(rand_nk_dists)
            # rank dist_i among rand_nk_dists
            dist_i = smallest_10_dist_df[i]
            rank = np.searchsorted(rand_nk_dists, dist_i)
            if (rank <= threshold) or (i == 0):
                dist_str = "{:.4f}".format(dist_i)
                all_test_EC.add(EC_i)
                ec.append('EC:' + str(EC_i) + '/' + dist_str)
            else:
                break
        ec.insert(0, col)
        csvwriter.writerow(ec)
    return


def write_top10_choices(df, csv_name):
    out_file = open(csv_name + '_top10.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    all_test_EC = set()
    for col in df.columns:
        ec = []
        smallest_10_dist_df = df[col].nsmallest(10)
        for i in range(10):
            EC_i = smallest_10_dist_df.index[i]
            dist_i = smallest_10_dist_df[i]
            dist_str = "{:.4f}".format(dist_i)
            all_test_EC.add(EC_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        ec.insert(0, col)
        csvwriter.writerow(ec)
    return all_test_EC


def random_nk_model(id_ec_train, emb_train, n=10, weighted=False):
    ids = list(id_ec_train.keys())
    nk = n * 1000
    if weighted:
        P = [len(id_ec_train[id]) for id in id_ec_train.keys()]
        P = P/np.sum(P)
        random_nk_id = np.random.choice(
            range(len(ids)), nk, replace=False, p=P)
    else:
        random_nk_id = np.random.choice(range(len(ids)), nk, replace=False)

    random_nk_id = np.sort(random_nk_id)
    chosen_ids = [ids[i] for i in random_nk_id]
    chosen_emb_train = emb_train[random_nk_id]
    return chosen_ids, chosen_emb_train


# def get_true_labels(file_name, entries):
#     result = open(file_name+'.csv', 'r')
#     csvreader = csv.reader(result, delimiter='\t')
#     all_label = set()
#     true_label_dict = {}
#     header = True
#     entries_set = set(entries)
#     for row in csvreader:
#         # don't read the header
#         if header is False:
#             true_ec_lst = row[1].split(';')
#             if row[0] in entries_set:
#                 true_label_dict[row[0]] = true_ec_lst
#             for ec in true_ec_lst:
#                 all_label.add(ec)
#         # change the header after first line
#         if header: header = False
#     true_label = [true_label_dict[i] for i in true_label_dict.keys()]
#     return true_label, all_label


# def get_pred_labels(pred_filename, pred_type="_maxsep"):
#     file_name = pred_filename+pred_type
#     result = open(file_name+'.csv', 'r')
#     csvreader = csv.reader(result, delimiter=',')
#     pred_label = []
#     entries = []
#     for row in csvreader:
#         # add id to entries list
#         entries.append(row[0])
#         preds_ec_lst = []
#         preds_with_dist = row[1:]
#         for pred_ec_dist in preds_with_dist:
#             # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
#             ec_i = pred_ec_dist.split(":")[1].split("/")[0]
#             preds_ec_lst.append(ec_i)
#         pred_label.append(preds_ec_lst)
#     return pred_label, entries


def get_true_labels(file_name):
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter='\t')
    all_label = set()
    true_label_dict = {}
    header = True
    count = 0
    for row in csvreader:
        # don't read the header
        if header is False:
            count += 1
            true_ec_lst = row[1].split(';')
            true_label_dict[row[0]] = true_ec_lst
            for ec in true_ec_lst:
                all_label.add(ec)
        if header:
            header = False
    true_label = [true_label_dict[i] for i in true_label_dict.keys()]
    return true_label, all_label


def get_pred_labels(out_filename, pred_type="_maxsep"):
    file_name = out_filename+pred_type
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_label = []
    for row in csvreader:
        preds_ec_lst = []
        preds_with_dist = row[1:]
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
            ec_i = pred_ec_dist.split(":")[1].split("/")[0]
            preds_ec_lst.append(ec_i)
        pred_label.append(preds_ec_lst)
    return pred_label


def print_eval_res(pred_label, true_label, all_label):
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_label)])
    n_test = len(pred_label)
    pred_m = np.zeros((n_test, len(mlb.classes_)))
    true_m = np.zeros((n_test, len(mlb.classes_)))
    for i in range(n_test):
        pred_m[i] = mlb.transform([pred_label[i]])
        true_m[i] = mlb.transform([true_label[i]])
    pre = precision_score(true_m, pred_m, average='macro', zero_division=0)
    rec = recall_score(true_m, pred_m, average='macro')
    f1 = f1_score(true_m, pred_m, average='macro')
    roc = roc_auc_score(true_m, pred_m, average='macro')
    acc = accuracy_score(true_m, pred_m)
    return pre, rec, f1, roc, acc
