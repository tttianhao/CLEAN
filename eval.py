import pickle
import torch
import pickle
from tqdm import tqdm
from model import Net
from helper.utils import *
from helper.distance_map import *
import pandas as pd


def eval_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train_data', type=str)
    parser.add_argument('-te', '--test_data', type=str)
    parser.add_argument('-n', '--model_name', type=str,
                        default='default_model')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('--high_precision', type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    args = eval_parse()
    # device and dtype
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float64 if args.high_precision else torch.float32
    # load id ec from tran and test
    id_ec_train, ec_id_dict_train = get_ec_id_dict(
        './data/' + args.train_data + '.csv')
    id_ec_test, ec_id_dict_test = get_ec_id_dict(
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
    eval_df.to_csv('./eval/distmap_' + args.test_data + '.csv')
    return


if __name__ == '__main__':
    main()
