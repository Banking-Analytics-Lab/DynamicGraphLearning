# Data

## Source

The paper uses data from one city of a Super-App company in Latin America that runs a delivery app
and issues credit cards (paper, Section 4). The network has monthly snapshots. Nodes are customers,
with credit-card usage features. Edges have four types (paper, Table A.1): users who used the same
credit card in the app, referrals, users who ordered more than 4 times in close geographical
proximity (Geohash 7), and users who have each other in their phone contacts. Customers who referred
another customer at least once are labelled influencers.

The data are not in this repository. Neither the repository nor the paper states their licence or
how to obtain them.

## Files the code expects

`make_data.py` reads two CSV files from `DATA_DIR` (an environment variable, default `data`, relative
to the working directory):

- `<name>.csv`, where `<name>` is the value of `--data`: one edge per row, after a header row that is
  skipped. The columns are read by position: source user id, destination user id, month (a number
  used as an integer index), label (1 marks the source user as an influencer in that month), and
  then numeric edge features, for example one-hot edge types. The script stops if the features of an
  edge sum to zero.
- `TGN_fin.csv`: node features, one row per user and month, with the columns `APPLICATION_USER_ID`
  and `START_DATE` (the month, as an integer) and numeric feature columns.

`make_data.py` writes the snapshots to `data/<name>_GAT.pt`, `data/<name>_GCN.pt` or, with
`--page_rank`, `data/<name>_PR.pt`, and `train.py` reads them from `data/`.

## Keep data out of git

`.gitignore` excludes `*.csv`, `*.tsv`, `*.pt`, `*.npy` and `*.json` files, `logs/` and
`model_files/`. Do not commit data or trained weights.
