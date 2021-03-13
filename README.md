## Installation

First make sure you have `torch` and `rdkit` installed.

Then run
```
pip install -e .
```
for development mode installation.

## Training to predict 13C shift in NMR data
First we download preprocessed data:
```
cd Data/NMR/
wget https://homes.esat.kuleuven.be/~jsimm/nmr_data/folding0.npy
wget https://homes.esat.kuleuven.be/~jsimm/nmr_data/folding1.npy
wget https://homes.esat.kuleuven.be/~jsimm/nmr_data/folding2.npy
wget https://homes.esat.kuleuven.be/~jsimm/nmr_data/nmr_webo.npy
```

Then train a GraphInformer model on the NMR shift data for 100 epochs using 3-distance attention heads:
```
python train.py --data_file nmr_webo.npy --folding_file folding0.npy \
  --head_radius 3 3 3 3 3 3 \
  --layers 2 \
  --hidden_size 256 \
  --hidden_dropout 0.1 \
  --num_epochs 100 \
  --lr_steps 50 80
```
