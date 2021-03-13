import graphinformer as gi
import unittest
import numpy as np
from rdkit import Chem

class TestMolData(unittest.TestCase):
    def test_sssr(self):
        m  = Chem.MolFromSmiles("CC(C)(Cc1c[nH]c2ccccc12)NCC(O)COc3ccccc3C#N")
        m2 = Chem.MolFromSmiles("CC2(C)CCCC(\C)=C2\C=C\C(\C)=C\C=C\C(\C)=C\C=C\C=C(/C)\C=C\C=C(/C)\C=C\C1=C(/C)CCCC1(C)C")
        dataset = gi.MolDataset([m, m2])
        sssr = dataset.route_features[0][1:,1:,15].numpy()
        self.assertTrue(np.diag(sssr == 0).all())

        correct = [[5, 4, 12, 7, 6], [8, 9, 10, 11, 12, 7], [20, 21, 22, 23, 24, 19]]
        for c in correct:
            a = sssr[np.ix_(c, c)].copy()
            np.fill_diagonal(a, 1.0)
            self.assertTrue((a == 1.0).all(), f"incorrect SSSR for ring {c}")
        for c in correct:
            sssr[np.ix_(c, c)] = 0.0
        self.assertTrue((sssr == 0.0).all(), f"Some non-ring pairs marked as SSSR.")

    def test_adjacency(self):
        m = Chem.MolFromSmiles("CC1CCC23C(C)C(CO)=CC2(C)CCC13")
        dataset = gi.MolDataset(weave_features=True, bond_types=True)
        dataset.add_mol(m)

        degree = np.array([len(a.GetBonds()) for a in m.GetAtoms()])
        self.assertTrue( (dataset.adj[0].sum(0).numpy() == degree).all() )

        A1 = dataset.route_features[0][1:,1:,16]
        self.assertTrue( (A1.numpy().sum(0) <= 4).all() )
        

if __name__ == "__main__":
    unittest.main()

