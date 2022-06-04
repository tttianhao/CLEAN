import torch
from helper.model import Net
from helper.utils import *
from helper.distance_map import *
from helper.evaluate import *
import pandas as pd


def eval_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train_data', type=str)
    parser.add_argument('-te', '--test_data', type=str)
    parser.add_argument('-n', '--model_name', type=str,
                        default='default_model')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('-p', '--p_value', type=float, default=0.002)
    parser.add_argument('-N', '--nk_random', type=float, default=4)
    parser.add_argument('--high_precision', type=bool, default=False)
    parser.add_argument('--weighted_random', type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    seed_everything()
    args = eval_parse()
    # device and dtype
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float64 if args.high_precision else torch.float32
    # load id ec from tran and test
    id_ec_train, ec_id_dict_train = get_ec_id_dict(
        './data/' + args.train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict(
        './data/' + args.test_data + '.csv')
    # load model
    model = Net(args.hidden_dim, args.out_dim, device, dtype)
    checkpoint = torch.load('./model/'+args.model_name+'.pth')
    model.load_state_dict(checkpoint)
    # compute distance map
    emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(
        emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    eval_df = pd.DataFrame.from_dict(eval_dist)
    # write the top 10 closest EC to _top10.csv
    out_filename = './eval/' + args.test_data
    _ = write_top10_choices(eval_df, out_filename)
    rand_nk_ids, rand_nk_emb_train = random_nk_model(
        id_ec_train, emb_train, n=args.nk_random, weighted=args.weighted_random)
    random_nk_dist_map = get_random_nk_dist_map(
        emb_train, rand_nk_emb_train, ec_id_dict_train, rand_nk_ids, device, dtype)
    write_random_nk_choices(
        eval_df, out_filename, random_nk_dist_map, p_value=args.p_value)
    return


if __name__ == '__main__':
    main()
