import sys
sys.path.append('./')
from helper.utils import *



def eval_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train_data', type=str)
    parser.add_argument('-te', '--test_data', type=str)
    args = parser.parse_args()
    return args


def main():
    args = eval_parse()
    id_ec_tr, ec_id_dict_tr = get_ec_id_dict(
        './data/' + args.train_data + '.csv')
    csv_name = './data/' + args.test_data + '.csv'
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    # number of subsets
    n_bins = 5
    out_writer_lst = []
    for bin in range(n_bins):
        subset_file = './data/subset/' + \
            args.test_data + '_subset' + str(bin) + '.csv'
        out_file = open(subset_file, 'w', newline='')
        csvwriter = csv.writer(out_file, delimiter='\t')
        out_writer_lst.append(csvwriter)

    for i, row in enumerate(csvreader):
        if i == 0:
            for bin in range(n_bins):
                out_writer_lst[bin].writerow(row)
        else:
            true_ec_lst = row[1].split(';')
            id_count_lst = [len(ec_id_dict_tr[ec]) for ec in true_ec_lst]
            id_count_ec = np.max(id_count_lst)

            if id_count_ec >= 2 and id_count_ec < 5:
                out_writer_lst[0].writerow(row)
            elif id_count_ec >= 5 and id_count_ec < 10:
                out_writer_lst[1].writerow(row)
            elif id_count_ec >= 10 and id_count_ec < 25:
                out_writer_lst[2].writerow(row)
            elif id_count_ec >= 25 and id_count_ec < 50:
                out_writer_lst[3].writerow(row)
            # for cases >= 50
            elif id_count_ec >= 2:
                out_writer_lst[4].writerow(row)


if __name__ == '__main__':
    main()
