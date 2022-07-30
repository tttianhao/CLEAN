import torch
from helper.model import *
from helper.utils import *
from helper.distance_map import *
from helper.evaluate import *
import pandas as pd
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def eval_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train_data', type=str)
    parser.add_argument('-te', '--test_data', type=str)
    parser.add_argument('-n', '--model_name', type=str,
                        default='default_model')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('-p', '--penalty', type=int, default=11)
    parser.add_argument('-t', '--top', type=int, default=10)
    parser.add_argument('--high_precision', type=bool, default=False)
    parser.add_argument('--dot', type=bool, default=False)
    parser.add_argument('-EP', '--eval_pretrained', type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    seed_everything()
    args = eval_parse()
    print('=' * 75)
    print(">>> arguments used: ", args)
    print('=' * 75)
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
    if args.eval_pretrained:
        # no model used for pretrained embedding
        model = lambda *args: args[0]
    else:
        model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype)
        checkpoint = torch.load('./model/'+args.model_name+'.pth')
        model.load_state_dict(checkpoint)
    # compute distance map
    emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(
        emb_train, emb_test, ec_id_dict_train, id_ec_test,
        device, dtype, dot=args.dot)
    eval_df = pd.DataFrame.from_dict(eval_dist)
    # write the top 10 closest EC to _top10.csv
    out_filename = './eval/' + args.test_data
    top_dists = write_top_choices(eval_df, out_filename, top=args.top)

    # get preds and true labels
    pred_label = get_pred_labels(out_filename, pred_type='_top'+str(args.top))
    true_label, all_label = get_true_labels('./data/'+args.test_data)
    mean_rank, map = get_MR_MAP(
        pred_label, true_label, top_dists, penalty=args.penalty)
    print('-' * 75)
    print(
        f'>>> mean rank: {mean_rank:.5} | mean average precision:  {map:.5} |')
    print('-' * 75)
    return


if __name__ == '__main__':
    main()
