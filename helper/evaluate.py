import csv
from helper.utils import *
from helper.distance_map import *
from helper.evaluate import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, \
    roc_auc_score, accuracy_score, f1_score, average_precision_score


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
    dists = []
    for col in df.columns:
        ec = []
        dist_lst = []
        smallest_10_dist_df = df[col].nsmallest(10)
        for i in range(10):
            EC_i = smallest_10_dist_df.index[i]
            dist_i = smallest_10_dist_df[i]
            dist_str = "{:.4f}".format(dist_i)
            dist_lst.append(dist_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        ec.insert(0, col)
        dists.append(dist_lst)
        csvwriter.writerow(ec)
    return dists


def random_nk_model(id_ec_train, ec_id_dict_train, emb_train, n=10, weighted=False):
    ids = list(id_ec_train.keys())
    nk = n * 1000
    if weighted:
        P = []
        for id in id_ec_train.keys():
            ecs_id = id_ec_train[id]
            ec_densities = [len(ec_id_dict_train[ec]) for ec in ecs_id]
            # the prob of calling this id is inversely prop to 1/max(density)
            P.append(1/np.max(ec_densities))
        P = P/np.sum(P)
        random_nk_id = np.random.choice(
            range(len(ids)), nk, replace=False, p=P)
    else:
        random_nk_id = np.random.choice(range(len(ids)), nk, replace=False)

    random_nk_id = np.sort(random_nk_id)
    chosen_ids = [ids[i] for i in random_nk_id]
    chosen_emb_train = emb_train[random_nk_id]
    return chosen_ids, chosen_emb_train


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


def get_eval_metrics(pred_label, true_label, all_label):
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_label)])
    n_test = len(pred_label)
    pred_m = np.zeros((n_test, len(mlb.classes_)))
    true_m = np.zeros((n_test, len(mlb.classes_)))
    for i in range(n_test):
        pred_m[i] = mlb.transform([pred_label[i]])
        true_m[i] = mlb.transform([true_label[i]])
    pre = precision_score(true_m, pred_m, average='weighted', zero_division=0)
    rec = recall_score(true_m, pred_m, average='weighted')
    f1 = f1_score(true_m, pred_m, average='weighted')
    roc = roc_auc_score(true_m, pred_m, average='weighted')
    acc = accuracy_score(true_m, pred_m)
    return pre, rec, f1, roc, acc


def get_MR_MAP_per_id(pred, true, dist, penalty=11):
    # get the rank of true labels in predicted labels
    ranks, = np.where(np.in1d(pred, true))
    # labels will have 1 only where a true label is present in pred labels
    labels = np.zeros(len(pred))
    labels[ranks] = 1
    # if a true label is not found in pred, add total rank with penalty
    total_penalty = penalty * (len(true) - len(ranks))
    # set ap to 0 if true label not found in prediction
    if len(ranks) == 0:
        ap_score = 0
    else:
        ap_score = average_precision_score(labels, - np.array(dist))
    # rank starts at 1
    ranks += 1
    nranks = len(true)
    ranksum = ranks.sum() + total_penalty
    return ranksum, nranks, ap_score


def get_MR_MAP(pred_label, true_label, top_dists, penalty=11):
    '''
        returns mean rank, mean avearge precision
    '''
    nranks, ranksum, map = 0, 0, 0
    for i in range(len(pred_label)):
        pred = pred_label[i]
        true = true_label[i]
        dist = top_dists[i]
        ranksum_i, nranks_i, ap_score = get_MR_MAP_per_id(pred, true, dist, penalty=penalty)
        ranksum += ranksum_i
        nranks += nranks_i
        map += ap_score
    
    return ranksum/nranks, map/len(pred_label)
