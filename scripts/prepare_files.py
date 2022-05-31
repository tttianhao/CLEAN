import os
import csv
import random

def csv_to_fasta(csv_name, fasta_name, id_Ec):
    csvfile = open(csv_name, 'r')
    csvreader = csv.reader(csvfile, delimiter = '\t')
    csv_out = open(id_Ec, 'w')
    csvwriter = csv.writer(csv_out, delimiter = '\t')
    outfile = open(fasta_name,'w')
    for i, rows in enumerate(csvreader):
        if i>0:
            outfile.write('>' + rows[0] + '\n')
            outfile.write(rows[2]+ '\n')
            if rows[1] == '':
                csvwriter.writerow([rows[0], 0])
            else:
                csvwriter.writerow([rows[0], 1])
        else:
            csvwriter.writerow(['ID','EC0'])

def check_ec0(fasta_name, id_ec):
    fasta = open(fasta_name, 'r')
    csvfile = open(id_ec, 'r')
    csvreader = csv.reader(csvfile, delimiter = '\t')
    label = {}
    for i, rows in enumerate(csvreader):
        if i > 0:
            label[rows[0]] = int(rows[1])
    count = [0, 0]
    for lines in fasta.readlines():
        if lines[0] == '>':
            id = lines[1:].strip()
            count[label[id]] += 1
    print(count)

def make_fasta(fasta_name, out_file, csv_file):
    fasta = open(fasta_name, 'r')
    out_fasta = open(out_file, 'w')
    csvfile = open(csv_file, 'r')
    csvreader = csv.reader(csvfile, delimiter = '\t')
    to_be_embed = set()
    for lines in fasta.readlines():
        if lines[0] == '>':
            id = lines[1:].strip()
            if not os.path.exists('/home/tianhao/project/EC_pred/esm_data/' + id + '.pt'):
                to_be_embed.add(id)
    print(len(to_be_embed))
    for i, rows in enumerate(csvreader):
        if i > 0 and rows[0] in to_be_embed:
            out_fasta.write('>' + rows[0] + '\n')
            out_fasta.write(rows[2]+ '\n')

def prepare_label(fasta_name, id_Ec, out_file):
    random.seed(114514)
    fasta = open(fasta_name, 'r')
    ids = set()
    for lines in fasta.readlines():
        if lines[0] == '>':
            id = lines[1:].strip()
            ids.add(id)

    csvfile = open(id_Ec, 'r')
    csvreader = csv.reader(csvfile, delimiter = '\t')

    outfile = open(out_file,'w')
    csvwriter = csv.writer(outfile, delimiter = '\t')
    
    for i, rows in enumerate(csvreader):
        if i > 0 and rows[0] in ids:
            label = 0 if rows[1]=='' else 1
            csvwriter.writerow([rows[0], label, random.uniform(0,1), rows[2]])
        elif i == 0:
            csvwriter.writerow(['ID','label','rand_n','seq'])

def make_csv(out_file, csv_file):
    out_csv = open(out_file, 'w')
    csvwriter = csv.writer(out_csv, delimiter = '\t')
    csvfile = open(csv_file, 'r')
    csvreader = csv.reader(csvfile, delimiter = '\t')
    counter = [0, 0]
    for i, rows in enumerate(csvreader):
        if i > 0:
            if os.path.exists('/home/tianhao/project/EC_pred/esm_data/' + rows[0] + '.pt'):
                csvwriter.writerow(rows)
                counter[int(rows[1])] += 1
        else:
            csvwriter.writerow(rows)
    print(counter)

if __name__ == '__main__':
    #csv_to_fasta('./data/swissProt.tab', './data/swissProt.fasta', './data/ec0.csv')
    #check_ec0('./data/uniref30_rep_seq.fasta', './data/ec0.csv')
    #make_fasta('./data/uniref30_rep_seq.fasta', './data/to_be_embed.fasta','./data/swissProt.tab')
    #prepare_label('./data/uniref30_rep_seq.fasta', './data/swissProt.tab', './data/full_list.csv')
    make_csv('./data/full_list_embed.csv', './data/full_list.csv')