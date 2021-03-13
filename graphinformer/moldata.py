import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from scipy.sparse import csr_matrix
import torch.nn.functional as F

class MolDataset(Dataset):
    def __init__(self, mols=[], node_prop=None, weave_features=False, bond_types=False, gasteiger = True):
        """
        node_prop   atom property that contains the node label
        """
        self.node_types    = {"pool": 1, 6: 2, 7: 3}
        self.default_type  = len(self.node_types) + 1
        self.node_features = []

        self.node_ids       = []
        self.adj            = []
        self.dists          = []
        self.route_features = []
        self.node_labels    = None

        self.node_prop = node_prop

        ## fields storing values for classif. and regres.
        self.mol_labels = None
        self.mol_values = None

        self.gast_f_count = 0

        self.weave_features = weave_features
        self.bond_types     = bond_types

        ## smart definitions for Hacceptors and Hdonors
        self.ha = Chem.MolFromSmarts("[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]")
        self.hd = Chem.MolFromSmarts("[!$([#6,H0,-,-2,-3])]")

        self.gasteiger = gasteiger
        for mol in mols:
            self.add_mol(mol)

    def save(self, filename):
        self.lists_to_arrays()
        np.save(filename,{
            "node_types":     self.node_types,
            "default_type":   self.default_type,
            "node_ids":       self.node_ids,
            "node_features":  self.node_features,
            "adj":            self.adj,
            "dists":          self.dists,
            "route_features": self.route_features,
            "node_labels":    self.node_labels,
            "node_prop":      self.node_prop,
            "mol_labels":     self.mol_labels,
            "mol_values":     self.mol_values,
            "weave_features": self.weave_features,
        })

    def lists_to_arrays(self):
        for k, v in self.__dict__.items():
            if type(v) != list:
                continue
            setattr(self, k, np.array(v, dtype=np.object))

    def check_ndarrays(self):
        sizes = [v.shape[0] for v in self.__dict__.values() if type(v) == np.ndarray]
        return (np.array(sizes) == sizes[0]).all()

    @staticmethod
    def from_npy(filename):
        d = np.load(filename, allow_pickle = True).item()
        dataset = MolDataset()
        for k,v in d.items():
            setattr(dataset, k, v)
        return dataset

    def subset(self, idx):
        """Keeps only idx compounds, drops all others."""
        dataset = MolDataset()
        if not self.check_ndarrays():
            raise ValueError("ndarrays of the dataset are not the same size.")
        for k, v in self.__dict__.items():
            if k in ["ha", "hd"]:
                ## ignoring smart fields
                continue
            if (v is None) or (type(v) in [dict, int, float, str, bool]):
                setattr(dataset, k, v)
            elif type(v) == np.ndarray:
                setattr(dataset, k, v[idx])
            else:
                raise TypeError(f"Subsetting failed for field '{k}' because its type in not allowed: {type(v)}.")
        return dataset

    @property
    def route_size(self):
        return self.route_features[0].shape[-1]

    @property
    def node_feature_size(self):
        return self.node_features[0].shape[-1]

    def add_atom_type(self, anum):
        if anum in self.node_types:
            return
        self.node_types[anum] = self.default_type
        self.default_type += 1

    def add_mol(self, mol, add_atom_types=True):
        """
        Adds molecule to the dataset.
        Returns True if successful, otherwise False.
        """
        ComputeGasteigerCharges(mol)
        offset = 0
        if self.gasteiger:
            gast = np.array([a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()])
            if np.isnan(gast).any():
                ## failed to compute partial charges
                return False
        else:
            offset = -1

        """Whether to add atom types."""
        if add_atom_types:
            for a in mol.GetAtoms():
                self.add_atom_type(a.GetAtomicNum())

        ## adding labels
        if self.node_prop is not None:
            if self.node_labels is None:
                self.node_labels = []
            labels = torch.FloatTensor([a.GetDoubleProp(self.node_prop) for a in mol.GetAtoms()])
            self.node_labels.append(labels)

        nodes = np.array([
            self.node_types.get(mol.GetAtomWithIdx(i).GetAtomicNum(), self.default_type)
            for i in range(mol.GetNumAtoms())
        ])
        nodes = np.concatenate([[self.node_types["pool"]], nodes])
        self.node_ids.append(torch.LongTensor(nodes))

        ## node features
        num_feat = 20
        if self.gasteiger:
            num_feat += 1
        if self.weave_features:
            num_feat += 4

        X = np.zeros((nodes.shape[0], num_feat), dtype=np.float32)
        
        fc = np.array([a.GetFormalCharge() for a in mol.GetAtoms()])
        X[1:, 0:3] = onehot(fc, -1, 1)

        hb = np.array([a.GetHybridization() for a in mol.GetAtoms()])
        X[1:, 3:8] = onehot(hb, 0, 4) 

        ev = np.array([a.GetExplicitValence() for a in mol.GetAtoms()])
        X[1:, 8:14] = onehot(ev, 0, 5)

        X[1:, 14] = [a.GetIsAromatic() for a in mol.GetAtoms()]

        X[1:, 15] = [a.IsInRingSize(3) for a in mol.GetAtoms()]
        X[1:, 16] = [a.IsInRingSize(4) for a in mol.GetAtoms()]
        X[1:, 17] = [a.IsInRingSize(5) for a in mol.GetAtoms()]
        X[1:, 18] = [a.IsInRingSize(6) for a in mol.GetAtoms()]

        X[1:, 19] = [a.IsInRing() for a in mol.GetAtoms()]

        if self.gasteiger:
            X[1:, 20] = gast

            assert np.isnan(X[1:,20]).any() == False, "Found NaN in GasteigerCharges"

        ## adding Hacceptors and Hdonors
        if self.weave_features:
            mha = mol.GetSubstructMatches(self.ha)
            mhd = mol.GetSubstructMatches(self.hd)
            assert all(len(x) == 1 for x in mha), "Hacceptor returned more than one atom"
            assert all(len(x) == 1 for x in mhd), "Hdonor returned more than one atom"
            ha_idx = [m[0]+1 for m in mha]
            hd_idx = [m[0]+1 for m in mhd]
            X[ha_idx, 21 + offset] = 1.0
            X[hd_idx, 22 + offset] = 1.0

            ## chiral centers
            chiral = Chem.FindMolChiralCenters(mol)
            r_idx  = [c[0]+1 for c in chiral if c[1] == "R"]
            s_idx  = [c[0]+1 for c in chiral if c[1] == "S"]
            X[r_idx, 23 + offset] = 1.0
            X[s_idx, 24 + offset] = 1.0

        self.node_features.append(torch.FloatTensor(X))

        ## route information
        A = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.float32)
        Aconj = np.zeros_like(A)
        A1 = np.zeros_like(A)
        A2 = np.zeros_like(A)
        A3 = np.zeros_like(A)
        Aarom = np.zeros_like(A)
        Aflex = np.zeros_like(A)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            A[i,j] = 1
            A[j,i] = 1
            Aflex[i,j] = 1
            Aflex[j,i] = 1
            if bond.GetIsConjugated():
                Aconj[i,j] = 1
                Aconj[j,i] = 1
            if bond.GetIsAromatic():
                Aarom[i,j] = 1
                Aarom[j,i] = 1
            if bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
                A1[i,j] = 1
                A1[j,i] = 1
                if not bond.GetIsConjugated():
                    ## is there a flexible shortest route
                    Aflex[i,j] = 0.99
                    Aflex[j,i] = 0.99
            elif bond.GetBondType() == rdkit.Chem.rdchem.BondType.DOUBLE:
                A2[i,j] = 1
                A2[j,i] = 1
            elif bond.GetBondType() == rdkit.Chem.rdchem.BondType.TRIPLE:
                A3[i,j] = 1
                A3[j,i] = 1

        
        self.adj.append(torch.FloatTensor(A))

        num_route_features = 16
        if self.bond_types:
            num_route_features += 3


        dists = comp_dists(A, dmax=13, self_loop=True)
        dists_conj = comp_dists(Aconj, dmax=Aconj.shape[0], self_loop=False)
        dists1     = comp_dists(A1, dmax=13, self_loop=False)
        dists2     = comp_dists(A2, dmax=13, self_loop=False)
        costs_flex = min_cost(Aflex, dmax=13)
        rigid      = (np.abs(np.round(costs_flex) - costs_flex) < 1e-4) & (costs_flex < 13)
        np.fill_diagonal(rigid, False)

        route = np.zeros((A.shape[0]+1, A.shape[0]+1, num_route_features), dtype=np.bool)
        route[1:,1:,0] = (dists == 0)
        route[1:,1:,1] = (dists == 1)
        route[1:,1:,2] = (dists == 2)
        route[1:,1:,3] = (dists == 3)
        route[1:,1:,4] = (dists == 4)
        route[1:,1:,5] = (5 <= dists) & (dists <= 6)
        route[1:,1:,6] = (7 <= dists) & (dists <= 8)
        route[1:,1:,7] = (9 <= dists) & (dists <= 12)
        route[1:,1:,8] = 13 <= dists

        route[1:,1:,9]  = dists_conj <= 4
        route[1:,1:,10] = (5 <= dists_conj) & (dists_conj < dists_conj.shape[0])
        route[1:,1:,11] = dists1 < 13
        route[1:,1:,12] = dists2 < 13
        route[1:,1:,13] = A3
        route[1:,1:,14] = rigid

        sssr = rdkit.Chem.rdmolops.GetSymmSSSR(mol)
        for ring in sssr:
            for a0 in ring:
                for a1 in ring:
                    if a0 == a1:
                        continue
                    route[a0+1, a1+1, 15] = 1

        if self.bond_types:
            route[1:, 1:, 16] = A1
            route[1:, 1:, 17] = A2
            route[1:, 1:, 18] = Aarom

        ## TODO: add 1+ queries
        self.route_features.append(torch.BoolTensor(route))

        ## if pool_dist is 0.0 then all heads can talk
        ## between pool and nodes
        ## TODO: try pool_dist 999.0
        pool_dist = 0.0
        D = np.zeros((A.shape[0]+1, A.shape[0]+1), dtype=np.float32) + pool_dist
        D[1:,1:] = dists
        self.dists.append(torch.FloatTensor(D))
        return True

    def __len__(self):
        return len(self.node_ids)

    def __getitem__(self, idx):
        item = {
            "node_ids":       self.node_ids[idx],
            "node_features":  self.node_features[idx],
            "adj":            self.adj[idx],
            "dists":          self.dists[idx],
            "route_features": self.route_features[idx].float(),
            "node_labels":    self.node_labels[idx] if self.node_labels is not None else None,
            "mol_labels":     self.mol_labels[idx] if self.mol_labels is not None else None,
            "mol_values":     self.mol_values[idx] if self.mol_values is not None else None
        }
        return item

def onehot(x, xmin, xmax):
    """convert x into (xmax - xmin + 1) binned one-hot encoding."""
    one = np.zeros((x.shape[0], xmax - xmin + 1), dtype=np.float32)
    x   = x.clip(xmin, xmax)
    one[range(len(x)), x - xmin] = 1.0
    return one

## 1) all-single
## 2) all-double

def comp_dists(adj, dmax=8, self_loop=True):
    dists = np.zeros_like(adj)
    nstep = np.eye(adj.shape[0])

    for i in range(dmax):
        nstep  = (nstep @ adj) > 0
        np.fill_diagonal(nstep, 0)
        update = (dists == 0) * nstep * (i+1)
        if update.sum() == 0:
            break
        dists += update

    dists[dists == 0] = dmax + 1
    if self_loop:
        np.fill_diagonal(dists, 0)
    else:
        np.fill_diagonal(dists, dmax + 1)
    return dists

def min_cost(adj, dmax):
    s     = csr_matrix(adj)
    costs = np.zeros_like(adj) + dmax + 1
    np.fill_diagonal(costs, 0)

    for i in range(dmax):
        for a in range(adj.shape[0]):
            start = s.indptr[a]
            end   = s.indptr[a+1]

            for off, b in enumerate(s.indices[start:end]):
                cost_ab  = s.data[start + off]
                costs[a] = np.minimum(costs[a], cost_ab + costs[b])
    return costs


def mol_collate(batch, add_adj=False):
    N   = len(batch)
    res = {}
    res["node_ids"]      = pad_sequence([s["node_ids"] for s in batch], batch_first=True)
    res["node_features"] = pad_sequence([s["node_features"] for s in batch], batch_first=True)
    if batch[0]["node_labels"] is not None:
        res["node_labels"] = pad_sequence([s["node_labels"] for s in batch], batch_first=True, padding_value=np.nan)

    if batch[0]["mol_labels"] is not None:
        res["mol_labels"] = torch.FloatTensor([ s["mol_labels"] for s in batch ])
    if batch[0]["mol_values"] is not None:
        res["mol_values"] = torch.FloatTensor([ s["mol_values"] for s in batch ])

    pads = np.array([s["node_ids"].shape[0] for s in batch])
    pads = pads.max() - pads
    p    = [(0, i, 0, i) for i in pads]

    res["dists"]          = torch.stack(
        [F.pad(batch[i]["dists"], p[i]) for i in range(N)]
    )
    res["route_features"] = torch.stack(
        [F.pad(batch[i]["route_features"], (0,0)+p[i]) for i in range(N)]
    )
    if add_adj:
        res["adj"] = torch.stack([F.pad(batch[i]["adj"], p[i]) for i in range(N)])

    return res

def getNMRPeaks(mol, prop_name = "Spectrum 13C 0", sanitize = True, atomic_num = 6):
    prop = mol.GetPropsAsDict()
    if prop_name in prop.keys():
        spectrum = np.zeros(mol.GetNumAtoms()) * np.nan
        for peak in prop[prop_name].split("|")[0:-1]:
            tok = peak.split(";")
            if int(tok[2]) >= spectrum.shape[0]:
                print(f"Peak for non-existing atom ({int(tok[2])}), atom count is {mol.GetNumAtoms()}.")
                return None
            spectrum[int(tok[2])] = float(tok[0])
        if sanitize:
            #Check if only the correct atom produces the peak and all correct atoms producing peak
            for i in range(mol.GetNumAtoms()):
                if not np.isnan(spectrum[i]) and mol.GetAtomWithIdx(i).GetAtomicNum() != atomic_num:
                    raise ValueError(f"Non-{atomic_num} had a peak.")
                if mol.GetAtomWithIdx(i).GetAtomicNum() == atomic_num and np.isnan(spectrum[i]):
                    return None
        return spectrum
    else:
        return None

def annotateEnvironments(mol, atomic_number, radius):
    prop_name = "env_%d_r%d"%(atomic_number,radius)
    for a in mol.GetAtoms():
        if not a.HasProp(prop_name):
            a.SetDoubleProp(prop_name, 0.0)

        if a.GetAtomicNum() == atomic_number:
            idx = a.GetIdx()
            env = Chem.rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius, idx)
            for b_idx in env:
                b = mol.GetBondWithIdx(b_idx)
                begin_atom = b.GetBeginAtom()
                end_atom = b.GetEndAtom()
                if begin_atom.GetIdx() != idx:
                    begin_atom.SetDoubleProp(prop_name, 1.0)
                if end_atom.GetIdx() != idx:
                    end_atom.SetDoubleProp(prop_name, 1.0)

class MolDatasetGGNN(Dataset):
    def __init__(self, node_features, node_mask, edges, n_edge_types, node_values, mol_labels):
        assert len(edges) == node_features.shape[0]
        assert node_mask.shape[0] == node_features.shape[0]

        self.node_features = node_features
        self.node_mask     = node_mask
        self.node_values   = node_values
        self.edges         = edges
        self.mol_labels    = mol_labels

        self.n_edge_types  = n_edge_types
        self.n_nodes       = self.node_features.shape[1]

    def __getitem__(self, index):
        out = {}
        out["node_features"] = self.node_features[index]
        out["node_mask"]     = self.node_mask[index]
        out["am"]            = self.create_adjacency_matrix(self.edges[index])
        if self.node_values is not None:
            out["node_values"] = self.node_values[index]
        if self.mol_labels is not None:
            out["mol_labels"] = self.mol_labels[index]

        return out

    def __len__(self):
        return len(self.node_features)

    def create_adjacency_matrix(self, edges):
        a = np.zeros([self.n_nodes, self.n_nodes * self.n_edge_types * 2], dtype=np.float32)
        for edge in edges:
            src_idx = edge[0]
            e_type = edge[1]
            tgt_idx = edge[2]
            a[tgt_idx][(e_type) * self.n_nodes + src_idx] =  1
            a[src_idx][(e_type + self.n_edge_types) * self.n_nodes + tgt_idx] =  1
        return a

    @staticmethod
    def from_npy(filename, all_route_features=False):
        d = np.load(filename, allow_pickle=True).item()
        n_atomtype    = max(d["node_types"].values()) + 1
        n_nodes       = max(x.shape[0] - 1 for x in d["node_features"])
        node_features = pad_sequence([x[1:] for x in d["node_features"]], batch_first=True)
        node_ids      = pad_sequence([x[1:] for x in d["node_ids"]], batch_first=True)
        assert node_ids.shape[0:2] == node_features.shape[0:2]
        assert d["route_features"][0].shape[2] >= 19

        node_ids_1hot = torch.zeros(node_ids.shape[0], node_ids.shape[1], n_atomtype)
        node_ids_1hot[np.repeat(range(node_ids.shape[0]), node_ids.shape[1]),
                      list(range(node_ids.shape[1])) * node_ids.shape[0],
                      node_ids.flatten()] = 1.0
        node_annotation = torch.cat([node_features, node_ids_1hot], dim=-1)
        edges = []
        if all_route_features:
            bond_layers = np.arange(d["route_features"][0].shape[-1])
        else:
            bond_layers = [16, 17, 13, 18]
        for x in d["route_features"]:
            nz = np.nonzero(x.numpy()[1:, 1:, bond_layers])
            edges_i = list(zip(nz[0], nz[2], nz[1]))
            edges.append(edges_i)

        if d["node_labels"] is None:
            node_values = None
        else:
            node_values = pad_sequence(d["node_labels"], batch_first=True, padding_value=np.nan)
            
        dataset = MolDatasetGGNN(
            node_features = node_annotation,
            node_mask     = (node_ids >= 1).float(),
            edges         = edges,
            n_edge_types  = len(bond_layers),
            node_values   = node_values,
            mol_labels    = d["mol_labels"],
        )
        return dataset

    def subset(self, idx):
        """Keeps only idx compounds, drops all others."""
        dataset = MolDatasetGGNN(
            node_features = self.node_features[idx],
            node_mask     = self.node_mask[idx],
            edges         = [self.edges[i] for i in idx],
            n_edge_types  = self.n_edge_types,
            node_values   = subset_array(self.node_values, idx),
            mol_labels    = subset_array(self.mol_labels, idx),
        )
        return dataset

def subset_array(x, idx):
    if x is None:
        return None
    return x[idx]

def to_1hot(node_ids, n_atomtype):
    node_ids_1hot = torch.zeros(node_ids.shape[0], node_ids.shape[1], n_atomtype)
    node_ids_1hot[np.repeat(range(node_ids.shape[0]), node_ids.shape[1]),
                  list(range(node_ids.shape[1])) * node_ids.shape[0],
                  node_ids.flatten()] = 1.0
    return node_ids_1hot

if __name__ == "__main__":
    m = Chem.MolFromSmiles("CC(C)(Cc1c[nH]c2ccccc12)NCC(O)COc3ccccc3C#N")
    m2 = Chem.MolFromSmiles("CC2(C)CCCC(\C)=C2\C=C\C(\C)=C\C=C\C(\C)=C\C=C\C=C(/C)\C=C\C=C(/C)\C=C\C1=C(/C)CCCC1(C)C")
    dataset = MolDataset([m, m2])

