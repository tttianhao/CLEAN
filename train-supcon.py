import torch
import time
import os
import pickle
from helper.dataloader import *
from helper.model import *
from helper.utils import *
from helper.losses import *
import torch.nn as nn
from helper.distance_map import get_dist_map


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--epoch', type=int, default=3500)
    parser.add_argument('-n', '--model_name', type=str,
                        default='default_model')
    parser.add_argument('-t', '--training_data', type=str)
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('-c', '--check_point', type=str, default='no')
    parser.add_argument('-m', '--margin', type=float, default=1)
    # ------------  SupCon-Hard specific  ------------ #
    parser.add_argument('-b', '--batch_size', type=int, default=512)
    parser.add_argument('-k', '--knn', type=int, default=100)
    parser.add_argument('-T', '--temp', type=float, default=0.1)
    parser.add_argument('--n_pos', type=int, default=9)
    parser.add_argument('--n_neg', type=int, default=30)
    # ------------------------------------------- #
    parser.add_argument('--adaptive_rate', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--high_precision', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--beta2', type=float, default=0.999)
    args = parser.parse_args()
    return args


def get_dataloader(dist_map, id_ec, ec_id, args):
    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
    }
    negative = mine_hard_negative(dist_map, args.knn)
    # only one positive for N-pair loss
    train_data = MultiPosNeg_dataset_with_mine_EC(
        id_ec, ec_id, negative, args.n_pos, args.n_neg)
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    return train_loader


def train(model, args, epoch, train_loader,
          optimizer, device, dtype, criterion):
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        model_emb = model(data.to(device=device, dtype=dtype))
        loss = criterion(model_emb, args.temp, args.n_pos)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if args.verbose and (batch % args.log_interval == 0) and (batch > 0):
            lr = args.learning_rate
            ms_per_batch = (time.time() - start_time) * \
                1000 / args.log_interval
            cur_loss = total_loss / args.log_interval
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    # record running average training loss
    return total_loss/(batch + 1)


def main():
    seed_everything(1234)
    args = parse()
    beta1 = 0.9
    beta2 = args.beta2
    torch.backends.cudnn.benchmark = True
    # get train set, test set is only used during evaluation
    if args.training_data is None:
        id_ec, ec_id_dict = get_ec_id_dict(
            './data/' + args.model_name + '.csv')
    else:
        id_ec, ec_id_dict = get_ec_id_dict(
            './data/' + args.training_data + '.csv')
    ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}
    #======================== override args ====================#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float64 if args.high_precision else torch.float32
    lr, epochs = args.learning_rate, args.epoch
    model_name = args.model_name
    # model_name = '_'.join([args.model_name, 'lr',
    #                       str(args.learning_rate), 'bs', str(args.batch_size)])
    print('==> device used:', device, '| dtype used: ',
          dtype, "\n==> args:", args)
    #======================== initialize model =================#
    model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype)
    if args.check_point != 'no':
        checkpoint = torch.load('./model/' + args.check_point+'.pth')
        model.load_state_dict(checkpoint)
        dist_map = pickle.load(
            open('./data/distance_map/uniref30_700.pkl', 'rb'))
    else:
        if args.training_data is None:
            dist_map = pickle.load(
                open('./data/distance_map/' + args.model_name + '.pkl', 'rb'))
        else:
            dist_map = pickle.load(
                open('./data/distance_map/' + args.training_data + '.pkl', 'rb'))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta1, beta2))
    criterion = SupConHardLoss
    best_loss = float('inf')
    train_loader = get_dataloader(dist_map, id_ec, ec_id, args)
    print("The number of unique EC numbers: ", len(dist_map.keys()))
    #======================== training =======-=================#
    # loading ESM embedding for dist map
    if os.path.exists('./data/distance_map/' + args.training_data + '_esm.pkl'):
        esm_emb = pickle.load(
            open('./data/distance_map/' + args.training_data + '_esm.pkl', 'rb')).to(device=device, dtype=dtype)
    else:
        esm_emb = esm_embedding(ec_id_dict, device, dtype)
        pickle.dump(esm_emb, open('./data/distance_map/' +
                    args.training_data + '_esm.pkl', 'wb'))
    # training
    for epoch in range(1, epochs + 1):
        if epoch % args.adaptive_rate == 0 and epoch != epochs + 1:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, betas=(beta1, beta2))
            # save updated model
            torch.save(model.state_dict(), './model/' +
                       model_name + '_' + str(epoch) + '.pth')
            # delete last model checkpoint
            if epoch != args.adaptive_rate:
                os.remove('./model/' + model_name + '_' +
                          str(epoch-args.adaptive_rate) + '.pth')
            # sample new distance map
            dist_map = get_dist_map(
                ec_id_dict, esm_emb, device, dtype, model=model)
            train_loader = get_dataloader(dist_map, id_ec, ec_id, args)
        # -------------------------------------------------------------------- #
        epoch_start_time = time.time()
        train_loss = train(model, args, epoch, train_loader,
                           optimizer, device, dtype, criterion)
        # only save the current best model near the end of training
        if (train_loss < best_loss and epoch > 0.8*epochs):
            torch.save(model.state_dict(), './model/' + model_name + '.pth')
            best_loss = train_loss
            print(f'Best from epoch : {epoch:3d}; loss: {train_loss:6.4f}')

        elapsed = time.time() - epoch_start_time
        print('-' * 75)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'training loss {train_loss:6.4f}')
        print('-' * 75)
    torch.save(model.state_dict(), './model/' + model_name + '_final.pth')


if __name__ == '__main__':
    main()
