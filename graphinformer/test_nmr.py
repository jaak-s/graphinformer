import moldata
from rdkit import Chem
import numpy as np

sup = Chem.SDMolSupplier('../Data/NMR/nmrshiftdb2withsignals.sd', removeHs = False, sanitize = False)
Nmol = 0
Nspec = 0
NonlyC = 0
NallC = 0
Nsym = 0
NsymallC = 0
for mol in sup:
    Nmol+=1
    spectrum = moldata.getNMRPeaks(mol, sanitize = False)
    if spectrum is not None:
        Nspec+=1
        onlyC = True
        allC = True
        for i in range(mol.GetNumAtoms()):
            if not np.isnan(spectrum[i]) and mol.GetAtomWithIdx(i).GetAtomicNum() != 6:
                onlyC = False
            if mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and np.isnan(spectrum[i]):
                allC = False
        if onlyC:
            NonlyC+=1
        if allC:
            NallC+=1
        matches = mol.GetSubstructMatches(mol,uniquify =False)
        if len(matches) > 1:
            Nsym+=1
            if allC:
                NsymallC+=1


print("Number of molecules in the SD file: %d" % Nmol)
print("Number of spectra loaded: %d" % Nspec)
print("Number of correct spectra (only C have peak): %d" % NonlyC)
print("Number of complete spectra (all C have peak): %d" % NallC)

print("Number of symmetrical molecules w. spectrum: %d" % Nsym)
print("Out of wich complete: %d" % NsymallC)


