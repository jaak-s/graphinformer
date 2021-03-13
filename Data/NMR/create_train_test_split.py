import graphinformer as gi
from rdkit import Chem
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Create folding for NMR')
parser.add_argument('--seed', help='Seed for train-valid-test splitting', type=int, default=12345678)
parser.add_argument('--outfile', help="Output file name", type=str, required=True)
args = parser.parse_args()
print(args)

dataset = gi.MolDataset.from_npy("./nmr_webo.npy")

np.random.seed(args.seed)
idx = np.random.permutation(len(dataset))
idx_train = idx[0:20000]
idx_valid = idx[20000:22500]
idx_test  = idx[22500:]

folding = {"idx_train":idx_train, "idx_valid":idx_valid, "idx_test": idx_test}
np.save(args.outfile, folding)
print(f"Saved folding for {idx.shape[0]} compounds into \"{args.outfile}\".")

