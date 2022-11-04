# Influencer Detection with Dynamic Graph Neural Networks 

This repository contain the code used in the paper E. Tiukhova, E. Penaloza, M. Óskarsdóttir, H. Garcia, A. Correa Bahnsen, B. Baesens, M. Snoeck, C. Bravo. Influencer Detection with Dynamic Graph Neural Networks. Accepted at Temporal Graph Learning workshop, NeurIPS, 2022

Link to the paper: TBD

Link to the poster: TBD


## Project structure: 

The project repo holds the following structure
```
 |-models
 | |-GNNs.py
 | |-RNNs.py
 | |-decoder.py
 | |-models.py
 |-reqs
 | |-DYNAMIC_GRAPHS_3.8.10.txt
 |-utils
 | |-utils.py
 |-make_data.py
 |-train.py
 

```
### models

This folder contains the .py files used to make combinations of encoder and decoder in dynamic GNN models as well as create baseline models.

### reqs

This folder contains the files that lists all of a project's dependencies.

### utils

This folder contains a .py file that provides functions for several files.

### make_data.py

The script to generate the network data and preprocess it. 

### train.py

The script to run the experiments. 
