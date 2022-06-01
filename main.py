import torch

import time
import pickle
from helper.dataloader import Dataset_with_mine_EC
from model import Net
from helper.utils import get_ec_id_dict, mine_hard_negative, parse
import torch.nn as nn
from helper.distance_map import get_dist_map


def train(model: nn.Module, args) -> None:
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        anchor, positive, negative = data
        anchor_out = model(anchor.to(device=device, dtype=dtype))
        positive_out = model(positive.to(device=device, dtype=dtype))
        negative_out = model(negative.to(device=device, dtype=dtype))

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if args.verbose and (batch % args.log_interval == 0) and (batch > 0):
            lr = args.learning_rate
            ms_per_batch = (time.time() - start_time) * \
                1000 / args.log_interval
            cur_loss = total_loss / args.log_interval
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:02.4f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    return total_loss/(batch + 1)


if __name__ == '__main__':
    args = parse()
    torch.backends.cudnn.benchmark = True
    # get train set, test set is only used during evaluation
    id_ec, ec_id_dict = get_ec_id_dict('./data/' + args.model_name + '.csv')
    train_set = list(id_ec.keys())
    ec_id = {}
    for key in ec_id_dict.keys():
        ec_id[key] = list(ec_id_dict[key])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if args.high_precision:
        dtype = torch.float64
    else:
        dtype = torch.float32
    print('device used: ', device, '; dtype used: ', dtype, "; args", args)

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 8
    }

    lr = args.learning_rate
    epochs = args.epoch
    model_name = args.model_name
    log_interval = args.log_interval

    model = Net(args.hidden_dim, args.out_dim, device, dtype)

    if args.check_point != 'no':
        checkpoint = torch.load('./model/' + args.check_point+'.pth')
        model.load_state_dict(checkpoint)
        dist_map = pickle.load(
            open('./data/distance_map/uniref30_700.pkl', 'rb'))
    else:
        dist_map = pickle.load(
            open('./data/distance_map/' + args.model_name + '.pkl', 'rb'))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=args.margin, reduction='mean')
    best_loss = float('inf')

    negative = mine_hard_negative(dist_map, args.knn)
    train_data = Dataset_with_mine_EC(id_ec, ec_id, negative)
    train_loader = torch.utils.data.DataLoader(train_data, **params)

    for epoch in range(1, epochs + 1):

        if epoch % args.adaptive_rate == 0 and epoch != epochs + 1:
            torch.save(model.state_dict(), './model/' +
                       model_name + '_' + str(epoch) + '.pth')
            dist_map = get_dist_map(args.out_dim, args.model_name, model=model)
            negative = mine_hard_negative(dist_map, args.knn)
            train_data = Dataset_with_mine_EC(
                id_ec, ec_id, train_set, negative)
            train_loader = torch.utils.data.DataLoader(train_data, **params)
            pickle.dump(dist_map, open('./data/distance_map/' +
                        model_name + '_' + str(epoch) + '.pkl', 'wb'))

        epoch_start_time = time.time()
        train_loss = train(model, args)

        elapsed = time.time() - epoch_start_time
        print('-' * 75)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'training loss {train_loss:5.2f}')
        print('-' * 75)
        if (train_loss < best_loss and epoch > 0.5*epochs):
            torch.save(model.state_dict(), './model/' + model_name + '.pth')
            best_loss = train_loss
            print(f'Best from epoch : {epoch:3d}; loss: {train_loss:5.2f}')

    torch.save(model.state_dict(), './model/' + model_name + '_final.pth')
