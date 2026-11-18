# Influencer Detection with Dynamic Graph Neural Networks

Code for the paper:

> E. Tiukhova, E. Penaloza, H. Garcia, A. Correa Bahnsen, M. Óskarsdóttir, B. Baesens, M. Snoeck, C. Bravo (2022). Influencer Detection with Dynamic Graph Neural Networks. *NeurIPS 2022 Temporal Graph Learning Workshop*. <https://doi.org/10.48550/arXiv.2211.09664>

Link to the paper: https://arxiv.org/abs/2211.09664

Link to the poster: https://neurips.cc/media/PosterPDFs/NeurIPS%202022/56519.png?t=1668072924.9758837

OpenReview: https://openreview.net/forum?id=LJeEO2lC_6

The paper treats influencer detection as node classification on a dynamic network of customers, built
from monthly snapshots. `make_data.py` turns an edge list and monthly node features into one PyTorch
Geometric graph per month. `train.py` trains an encoder-decoder model on windows of consecutive months:
a GNN encoder per snapshot (GCN or GAT in the paper; GIN and GraphSAGE are also implemented), an RNN
across snapshots (LSTM or GRU), and a feed-forward decoder. The paper's baselines are PageRank+RNN,
Features+RNN and static GNNs. The script reports AUC and AUPRC on nodes seen and unseen in training.

### Repository layout

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
 |-data
 | |-README.md
 |-make_data.py
 |-train.py
```

- `models/`: the encoders and decoders that make up the dynamic GNN models, and the baseline models.
- `reqs/`: the file that lists the project's dependencies.
- `utils/utils.py`: functions shared by the scripts (metrics, bootstrap, oversampling, logging).
- `make_data.py`: the script that generates the network data and preprocesses it.
- `train.py`: the script that runs the experiments.

## Requirements

The environment file is `reqs/DYNAMIC_GRAPHS_3.8.10.txt`. Its name gives Python 3.8.10. Many pins
carry the `+computecanada` suffix of the Digital Research Alliance of Canada wheelhouse, so the file
installs as is only on an Alliance cluster; elsewhere, remove the suffixes first.

```bash
pip install -r reqs/DYNAMIC_GRAPHS_3.8.10.txt
```

The file does not list `torch-geometric`, `networkx`, `tqdm` or `wandb`, which the code imports
(`train.py` imports `wandb`, but every call to it is commented out). Their versions were not
recorded. `models/models.py` imports the standard-library `turtle` module, which needs a Python
built with Tk.

Table A.3 of the paper lists the resources used: 2 CPU cores per task, 8 GB of memory per CPU, and
Intel E5-2683 v4 (Broadwell, 2.1 GHz) processors. Table 1 of the paper gives the total time of each
model.

Two environment variables set the paths. Their defaults, and the `data/` folder where `make_data.py`
writes and `train.py` reads the snapshots, are relative to the working directory, so run the scripts
from the repository root.

| Variable | Default | Used by |
|---|---|---|
| `DATA_DIR` | `data` | `make_data.py`: folder that holds the input CSV files |
| `OUTPUT_DIR` | `.` | `utils/utils.py` (`write_log`): folder that holds `logs/` and `model_files/` |

## Data

The paper uses data from one city of a Super-App company in Latin America that runs a delivery app
and issues credit cards. Nodes are customers, with credit-card usage features in each monthly
snapshot. Edges are colored by type: shared credit card, referral, geographical proximity (Geohash 7)
and phone contacts. Customers who referred others at least once are labelled influencers (paper,
Section 4 and Table A.1). The data are not in this repository, and neither the repository nor the
paper states how to obtain them. `data/README.md` describes the files the code expects.

## Training

Build the monthly snapshots. `make_data.py` reads `$DATA_DIR/<name>.csv` and `$DATA_DIR/TGN_fin.csv`
and writes `data/<name>_GAT.pt` (edge features as vectors, for GAT, GIN and GraphSAGE) or
`data/<name>_GCN.pt` (edge features summed into one weight, for GCN, the Features+RNN models and the
static GCN).

```bash
python make_data.py --data <name> --GNN GAT
python make_data.py --data <name> --GNN GCN
```

`--val_start` and `--test_start` (defaults 13 and 15) mark the months whose node features are scaled
with the minimum and maximum of the earlier, training months. `train.py` sets the training, validation
and test windows itself, from the last months in the snapshots.

Train one model. `write_log` appends to `$OUTPUT_DIR/logs/` and saves to `$OUTPUT_DIR/model_files/`,
and does not create these folders.

```bash
mkdir -p logs model_files
python train.py --data_name <name> --GNN GAT --RNN GRU --run_name <run-name>
```

The flags below select each model of the paper in `train.py`. They were read from the code; the
commands the authors ran were not recorded.

| Model in the paper | Flags |
|---|---|
| GCN+LSTM, GCN+GRU | `--GNN GCN --RNN LSTM`, `--GNN GCN --RNN GRU` |
| GAT+LSTM, GAT+GRU | `--GNN GAT --RNN LSTM`, `--GNN GAT --RNN GRU` |
| Features+LSTM, Features+GRU | `--RNN LSTM`, `--RNN GRU` |
| PageRank+LSTM, PageRank+GRU | `--page_rank --RNN LSTM`, `--page_rank --RNN GRU` (see the known issues) |
| Static GAT, static GCN | `--GNN GAT --full_windows`, `--GNN GCN --full_windows` (the static model accepts single months only) |

Table A.2 of the paper gives the best hyperparameters found by grid search (500 epochs, learning rate
0.0001, Adam). The grid-search scripts are not in the repository. `train.py` exposes the
hyperparameters as flags: `--embedding_dim`, `--gnn_output_dim`, `--RNN_hidden_dim`, `--GNN_layers`,
`--RNN_layers`, `--heads`, `--dropout_rate`, `--upsample_rate` (oversampling of the minority class),
`--epochs` and `--lr`. `train.py` keeps the weights of the epoch with the lowest validation loss; it
has no early stopping.

Known issues, found by reading the code (it was not run):

- `make_data.py --page_rank` saves `data/<name>_PR.pt`, but `train.py --page_rank` loads
  `data/<name>_page_rank.pt`.
- In `make_data.py`, line 180 discards the PageRank features computed at line 176, so the saved
  snapshots carry no PageRank feature.
- In `train.py`, the training loop passes a NumPy array to `torch.sigmoid`, which expects a tensor.
- `evauluate` in `utils/utils.py` calls the model with `train = False`, which only the GNN+RNN model
  accepts; the Features+RNN and static GNN models do not take this argument.

## Evaluation

`train.py` evaluates the model at the end of training. It reloads the weights with the lowest
validation loss and computes AUC and AUPRC on the validation and test windows, over all nodes, the
nodes seen in the last training window, and the unseen nodes. It adds bootstrap statistics of AUC and
AUPRC on seen and unseen nodes (`--boot_sample` resamples, default 10000). It prints all results,
appends the point estimates to `$OUTPUT_DIR/logs/<log_file>` (`--log_file`, default `logs.csv`), and
saves the weights to `$OUTPUT_DIR/model_files/<run_name>`.

## Pretrained models

Not released. The repository holds no trained weights, and it does not state a reason.

## Results

| Table or figure in the paper | Command | Expected result |
|---|---|---|
| Table 1 (test AUC on seen and unseen nodes) | `train.py`, one run per model (see Training) | Table 1 of the paper. How the runs were combined into the table: Not documented |
| Table 1 (total time) | Not documented | Table 1 of the paper |
| Table A.2 (best hyperparameters, validation AUC) | Not documented (grid search not in the repository) | Table A.2 of the paper |
| Figure A.2 (network growth) | Not documented | Figure A.2 of the paper |

## Citation

```bibtex
@inproceedings{tiukhova2022influencer,
  title         = {Influencer Detection with Dynamic Graph Neural Networks},
  author        = {Tiukhova, Elena and Penaloza, Emiliano and Garcia, Hernan and Correa Bahnsen, Alejandro and Óskarsdóttir, María and Baesens, Bart and Snoeck, Monique and Bravo, Cristián},
  booktitle     = {NeurIPS 2022 Temporal Graph Learning Workshop},
  year          = {2022},
  eprint        = {2211.09664},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2211.09664},
  url           = {https://arxiv.org/abs/2211.09664}
}
```

`CITATION.cff` holds the same reference for GitHub's "Cite this repository" button.

## Licence

Licence: pending.

## Acknowledgements

The paper acknowledges the ING Chair on Applying Deep Learning on Metadata as a Competitive
Accelerator, the Icelandic Research Fund (IRF, grant 228511-051), the Natural Sciences and
Engineering Research Council of Canada (NSERC, Discovery Grant RGPIN-2020-07114), the Canada Research
Chairs program, and Compute Ontario and the Digital Research Alliance of Canada (FT #2070).
