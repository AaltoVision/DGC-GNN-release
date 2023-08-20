# DGC-GNN: Descriptor-free Geometric-Color Graph Neural Network for 2D-3D Matching

Authors: [Shuzhe Wang](https://ffrivera0.github.io), [Juho Kannala](https://users.aalto.fi/~kannalj1/), [Daniel Barath](https://scholar.google.com/citations?hl=da&user=U9-D8DYAAAAJ&view_op=list_works&sortby=pubdate)

## Environment Setup
```
conda env create -f environment.yml
conda activate dgc-gnn
```

We need to install the corresponding ```torch_scatter=2.0.8```

```
wget https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_scatter-2.0.8-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.8-cp37-cp37m-linux_x86_64.whl
```

Now install DGC-GNN

```
pip install . --find-links https://data.pyg.org/whl/torch-1.8.0+cu11.1.html
```
## Data Preparation
Coming soon
## Evaluation

#### Pretained model

```
# Eval on MegaDepth
sh eval.sh
```
## Training
Coming soon
