import torch
import argparse
import time
import pickle
from helper.dataloader import Dataset_with_mine_EC
from model import Net
from helper.utils import get_ec_id_dict, mine_hard_negative
import torch.nn as nn
from helper.distance_map import get_dist_map

torch.set_default_dtype(torch.float64)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type = float, default=0.01)
    parser.add_argument('-e', '--epoch', type = int, default=10000)
    parser.add_argument('-n', '--model_name', type = str, default='default_model')
    parser.add_argument('-d', '--hidden_dim', type = int, default=512)
    parser.add_argument('-k', '--knn', type = int, default=10)
    parser.add_argument('-o', '--out_dim', type = int, default=128)
    parser.add_argument('-b', '--batch_size', type = int, default=2720)
    parser.add_argument('-c', '--check_point',type = str,default = 'no')
    parser.add_argument('-m', '--margin',type = float,default = 1)
    parser.add_argument('--adaptive_rate',type = int, default = 100)
    parser.add_argument('--log_interval', type = int, default = 1)
    parser.add_argument('--hyperbolic', type = bool, default = False)
    args = parser.parse_args()
    return args

def train(model: nn.Module) -> None:
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch, data in enumerate(train_loader):
        anchor, positive, negative = data
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        loss = criterion(anchor_out, positive_out, negative_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = args.learning_rate
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:02.4f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module) -> float:
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch,data in enumerate(valid_loader):
            anchor, positive, negative = data
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            
            total_loss += criterion(anchor_out, positive_out, negative_out).item()
    return total_loss / (len(valid_loader) - 1)

def hyperbolic_dist(u, v):
    sqdist = torch.sum((u - v) ** 2, dim=-1)
    squnorm = torch.sum(u ** 2, dim=-1)
    sqvnorm = torch.sum(v ** 2, dim=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-7
    z = torch.sqrt(x ** 2 - 1)
    return torch.log(x + z)

if __name__ == '__main__':
    args = get_parser()
    id_ec, ec_id_dict = get_ec_id_dict('./data/' + args.model_name + '.csv')
    train_set = list(id_ec.keys())
    test_set = list(id_ec.keys())
    ec_id = {}
    for key in ec_id_dict.keys():
        ec_id[key] = list(ec_id_dict[key])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('device used: ', device)
    torch.backends.cudnn.benchmark = True

    print(args)

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 40
    }

    lr = args.learning_rate
    epochs = args.epoch
    model_name = args.model_name
    log_interval = args.log_interval

    model = Net(args.hidden_dim, args.out_dim)
    model = model.to(torch.float64)

    if args.check_point != 'no':
        checkpoint = torch.load('./model/' + args.check_point+'.pth')
        model.load_state_dict(checkpoint)
        dist_map = pickle.load(open('./data/distance_map/uniref30_700.pkl', 'rb'))
    else:
        dist_map = pickle.load(open('./data/distance_map/'+ args.model_name + '.pkl', 'rb'))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if args.hyperbolic:
        criterion = nn.TripletMarginWithDistanceLoss(distance_function = hyperbolic_dist)
    else:
        criterion = nn.TripletMarginLoss(margin = args.margin, reduction = 'mean')
    best_val_loss = float('inf')

    negative = mine_hard_negative(dist_map, args.knn)
    train_data = Dataset_with_mine_EC(id_ec, ec_id, train_set, negative)
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    valid_data = Dataset_with_mine_EC(id_ec, ec_id, test_set, negative)
    valid_loader = torch.utils.data.DataLoader(valid_data, **params)
    
    for epoch in range(1, epochs + 1):

        if epoch % args.adaptive_rate == 0 and epoch != epochs + 1:
            torch.save(model.state_dict(), './model/' + model_name + '_' + str(epoch) + '.pth')
            
            if args.hyperbolic:
                dist_map = get_dist_map(device, model, args.out_dim, args.model_name, 'hyper')
            else:
                dist_map = get_dist_map(device, model, args.out_dim, args.model_name)
            negative = mine_hard_negative(dist_map, args.knn)
            
            train_data = Dataset_with_mine_EC(id_ec, ec_id, train_set, negative)
            train_loader = torch.utils.data.DataLoader(train_data, **params)

            valid_data = Dataset_with_mine_EC(id_ec, ec_id, test_set, negative)
            valid_loader = torch.utils.data.DataLoader(valid_data, **params)
            
            pickle.dump(dist_map, open('./data/distance_map/' + model_name + '_' + str(epoch) + '.pkl','wb'))

        epoch_start_time = time.time()
        train(model)
        val_loss = evaluate(model)
        elapsed = time.time() - epoch_start_time
        print('-' * 75)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f}')
        print('-' * 75)
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), './model/' + model_name + '.pth')
            best_val_loss = val_loss
            print(f'Current best : {epoch:3d}')

    torch.save(model.state_dict(), './model/' + model_name + '_final.pth')