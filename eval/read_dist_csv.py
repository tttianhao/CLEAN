import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import stats
import copy
import argparse

# read df file, and write csv file

def write_csv(csv_name: str, df):
    #print('-'*15)
    out_file = open(csv_name + '_result.csv', 'w')
    csvwriter = csv.writer(out_file, delimiter = '|')
    avg_out = 0
    zero = 0
    print(len(df.columns))
    for col in df.columns:
        ec = []
        target = (copy.copy(df[col].nsmallest(5)))
        population = (copy.copy(df[col].nsmallest(10)))
        print(population)
        q3, q1 = np.percentile(population, [75 ,25])
        iqr = q3 - q1
        cutoff = q1 - 1.5*iqr
        counter = 0
        for j in population:
            if j < cutoff:
                counter += 1
        if counter == 0:
            zero += 1
        avg_out += counter
        # print(q1 - 1.5*iqr)
        # print(target)                                                                                                                                                         
        if args.task == 'ecoli' or args.task == 'strep' or args.task == 'halo' or args.task == 'query1':
            #if target[0] < 2:
            print(target)
            #print(population)
            print('-'*15)
            csvwriter.writerow([target])
        else:
            #print(target)
            #print(population)
            for i in range(10):
                cur = df[col].nsmallest(10).index[i]
                ec.append('EC:' + str(cur))
            ec.insert(0,col)
            csvwriter.writerow(ec)
    print(avg_out/len(df.columns))
    print(zero)

def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type = str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = add_parser()
    if args.task == 'new':
        embed_df = pd.read_csv('./data/new_ec177_seq392_embed_df.csv', index_col=[0]).T
        write_csv('./data/new_ec177_seq392_embed', embed_df)
    elif args.task == 'price':
        embed_df = pd.read_csv('./data/price_embed_df.csv', index_col=[0]).T
        write_csv('./data/price_embed', embed_df)
    elif args.task == 'ecoli':
        embed_df = pd.read_csv('./data/ecoli_final_embed_df.csv', index_col=[0]).T
        write_csv('./data/ecoli_final_embed', embed_df)
    elif args.task == 'query':
        embed_df = pd.read_csv('./data/query_embed_df.csv', index_col=[0]).T
        write_csv('./data/query_embed', embed_df)
    elif args.task == 'strep':
        embed_df = pd.read_csv('./data/streptomyces90_embed_df.csv', index_col=[0]).T
        write_csv('./data/streptomyces90_embed', embed_df)
    elif args.task == 'halo':
        embed_df = pd.read_csv('./data/halogenase_embed_df.csv', index_col=[0]).T
        write_csv('./data/halogenase_embed', embed_df)
    elif args.task == 'uniref':
        embed_df = pd.read_csv('./data/test_uniref30_over20_embed_df.csv', index_col=[0]).T
        write_csv('./data/test_uniref30_over20_embed', embed_df)
    elif args.task == 'query1':
        embed_df = pd.read_csv('./data/query1_embed_df.csv', index_col=[0]).T
        write_csv('./data/query1_embed', embed_df)
    else:
        embed_df = pd.read_csv('./data/' + args.task + '.csv', index_col=[0]).T
        write_csv('./data/' + args.task, embed_df)