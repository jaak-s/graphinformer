from rdkit import Chem
import scipy.io
import tqdm
import graphinformer as gi
import numpy as np
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Creating ChEMBL dataset.')
parser.add_argument('--weave_features', help='Use Weave features', type=int, default=1)
parser.add_argument('--bond_types', help='Add bond types', type=int, default=1)
parser.add_argument('--filename', help="Name for the .npy file.", type=str, required=True)
args = parser.parse_args()
print(args)

def setPeaks(mol, peaks, prop):
    if peaks is None:
        return
    for i, p in enumerate(peaks):
        mol.GetAtomWithIdx(i).SetDoubleProp(prop, p)

suppl   = Chem.SDMolSupplier("./nmrshiftdb2withsignals.sd", removeHs = False, sanitize = True)
dataset = gi.MolDataset(node_prop="peak", weave_features=args.weave_features, bond_types=args.bond_types)
ids   = []
mols  = []

for mol in tqdm.tqdm(suppl):
    if mol is None:
        continue
    peak = gi.getNMRPeaks(mol)
    if peak is None:
        continue
    setPeaks(mol, peak, "peak")
    Chem.RemoveHs(mol)
    if dataset.add_mol(mol):
        ids.append(mol.GetProp("nmrshiftdb2 ID"))
        mols.append(mol)

## saving to disk
df = pd.DataFrame({"ids": ids})
df.to_csv(f"{args.filename}.ids.csv")

dataset.save(args.filename)

print(f"Saved {len(dataset)} compounds and their NMR peaks into '{args.filename}'.")
print(f"Saved nmrshiftdb2 ids into '{args.filename}.ids.csv'.")

