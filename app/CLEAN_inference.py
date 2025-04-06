## CLEAN inference version without needing to save esm1b files

from esm import pretrained
import pysam
from easydict import EasyDict as edict
from CLEAN.utils import *
from CLEAN.evaluate import *
from CLEAN.model import LayerNormNet
from CLEAN.distance_map import *
import pandas as pd 
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="split100")
    parser.add_argument("--inference_fasta_folder", type=str, default="data")
    parser.add_argument("--inference_fasta", type=str, default="new.fasta")
    parser.add_argument("--gpu_id", type=int, default=0)  # Set to None if you want to force CPU
    parser.add_argument("--inference_fasta_start", type=int, default=0)
    parser.add_argument("--inference_fasta_end", type=int, default=300)
    parser.add_argument("--toks_per_batch", type=int, default=2048)
    parser.add_argument("--esm_type", type=str, default="esm1b_t33_650M_UR50S")
    parser.add_argument("--truncation_seq_length", type=int, default=1022)
    parser.add_argument("--esm_batches_per_clean_inference", type=int, default=200)
    parser.add_argument("--gmm", type=str, default="./data/pretrained/gmm_ensumble.pkl")

    return parser.parse_args()


class CustomFastaBatchedDataset(object):
    def __init__(self, fasta_obj, fasta_start=0, fasta_end=None):
        """
        Initialize the dataset from a pysam.FastaFile object.

        Parameters:
            fasta_obj (pysam.FastaFile): The pysam FASTA file object.
            fasta_start (int, optional): Start index (inclusive) for slicing references. Defaults to 0.
            fasta_end (int, optional): End index (exclusive) for slicing references.
                                       If None, all sequences from fasta_start onward are used.
        """
        # Get all references from the FASTA object.
        refs = fasta_obj.references
        if fasta_end is None:
            fasta_end = len(refs)

        self.sequence_indices_labels_dict = {
            i:label for i, label in zip(
                range(len(refs)), refs)
        }
        self.sequence_labels_indices_dict = {
            label:i for i, label in zip(
                range(len(refs)), refs)
        }
        # Slice the list of references.
        subset_refs = refs[fasta_start:fasta_end]
        # Populate the sequence labels and the sequence strings.
        self.sequence_labels = list(subset_refs)
        self.sequence_strs = [fasta_obj.fetch(ref) for ref in subset_refs]
        

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        return self.sequence_labels[idx], self.sequence_strs[idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches


def get_last_layer_emb(args, model, toks, strs, repr_layers):
    if torch.cuda.is_available() and not args.gpu_id:
        toks = toks.to(device="cuda", non_blocking=True)
    with torch.no_grad():
        out = model(toks, repr_layers=repr_layers, return_contacts=False)
        representations = {
            layer: t.to(device="cpu") for layer, t in out["representations"].items()
        }
        representations = representations[repr_layers[0]] # only use the last layer
        truncate_lens = [min(args.truncation_seq_length, len(strs_i)) for strs_i in strs]
        embeddings = [emb[1 : truncate_len + 1].mean(0).clone() for emb,truncate_len 
                        in zip(representations, truncate_lens)]
    return embeddings

def get_max_sep_predictions_dict(inference_df, gmm):
    max_sep_predictions = {}
    for sequence_label in inference_df.columns:
        smallest_10_dist_df = inference_df[sequence_label].nsmallest(10)
        dist_lst = list(smallest_10_dist_df)
        max_sep_i = maximum_separation(dist_lst, True, False)
        ec = []
        for i in range(max_sep_i+1):
            EC_i = smallest_10_dist_df.index[i]
            dist_i = smallest_10_dist_df[i]
            if gmm != None:
                gmm_lst = pickle.load(open(gmm, 'rb'))
                dist_i = infer_confidence_gmm(dist_i, gmm_lst)
            dist_str = "{:.4f}".format(dist_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        max_sep_predictions[sequence_label] = ec
    return max_sep_predictions


def CLEAN_max_sep_predictions(
        args,  CLEAN_model, sequence_label_esm_emb_dict, emb_train, ec_id_dict_train, device):
    esm_emb_inference = torch.cat(
        [sequence_label_esm_emb_dict[label].unsqueeze(0) 
            for label in sequence_label_esm_emb_dict])
    id_ec_inference_dummy = {seq_label:[] for seq_label in sequence_label_esm_emb_dict}
    with torch.no_grad():
        model_emb_inference  = CLEAN_model(esm_emb_inference.to(device)).to("cpu").clone()
    inference_dist = get_dist_map_test(
        emb_train, model_emb_inference, ec_id_dict_train, id_ec_inference_dummy, "cpu", torch.float32)
    inference_df = pd.DataFrame.from_dict(inference_dist)
    max_sep_predictions_dict = get_max_sep_predictions_dict(inference_df, args.gmm)
    return max_sep_predictions_dict


def main():
    args = get_args()
    inference_fasta_path = f'{args.inference_fasta_folder}/{args.inference_fasta}'
    print("loading inference fasta")
    inference_fasta = pysam.FastaFile(inference_fasta_path)
    dataset = CustomFastaBatchedDataset(
        inference_fasta, fasta_start=args.inference_fasta_start, fasta_end=args.inference_fasta_end)
    # keep track of index label mappings in the original fasta
    #sequence_indices_labels_dict = dataset.sequence_indices_labels_dict
    sequence_labels_indices_dict = dataset.sequence_labels_indices_dict
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)

    print("loading ESM model")
    model, alphabet = pretrained.load_model_and_alphabet(args.esm_type)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in [-1]] # only use the last layer
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
    )
    if torch.cuda.is_available() and not args.gpu_id:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        model = model.cuda()

    ## load CLEAN model
    if torch.cuda.is_available() and not args.gpu_id:
        device = "cuda"
    else: 
        device = "cpu"

    print("loading CLEAN model")
    CLEAN_model = LayerNormNet(512, 128, device, torch.float32)
    checkpoint = torch.load('./data/pretrained/'+ args.train_data +'.pth', map_location=device)
    CLEAN_model.load_state_dict(checkpoint)
    CLEAN_model.eval()
    _, ec_id_dict_train = get_ec_id_dict('./data/' + args.train_data + '.csv')
    emb_train = torch.load('./data/pretrained/100.pt', map_location="cpu")


    sequence_label_esm_emb_dict = {}
    max_sep_predictions_dict = {} 

    for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader)):
        embeddings = get_last_layer_emb(
            args=args, model=model, toks=toks, strs=strs, repr_layers=repr_layers)
        for label, emb in zip(labels, embeddings):
            sequence_label_esm_emb_dict[label] = emb

        if (batch_idx + 1) % args.esm_batches_per_clean_inference == 0:
            # perform clean inference
            predictions = CLEAN_max_sep_predictions(
                args, CLEAN_model, sequence_label_esm_emb_dict, emb_train, ec_id_dict_train, device
            )
            max_sep_predictions_dict.update(predictions)
            sequence_label_esm_emb_dict = {}
            
    # process any remaining sequences that didn't complete a full batch group
    if sequence_label_esm_emb_dict:
        predictions = CLEAN_max_sep_predictions(
            args, CLEAN_model, sequence_label_esm_emb_dict, emb_train, ec_id_dict_train, device
        )
        max_sep_predictions_dict.update(predictions)


    max_sep_predictions_df = pd.DataFrame([
        {'Index': sequence_labels_indices_dict[seq_id], 'Seq_ID': seq_id, 'Prediction': '; '.join(max_sep_predictions_dict[seq_id])}
        for seq_id in max_sep_predictions_dict
    ]).sort_values('Index').reset_index(drop=True)
    fasta_name = args.inference_fasta.split(".")[0]
    prediction_file_name = f"results/inputs/{fasta_name}_{args.inference_fasta_start}_{args.inference_fasta_end}.csv"
    max_sep_predictions_df.to_csv(prediction_file_name, index=False)


if __name__ == '__main__':
    main()
