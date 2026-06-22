# Real-world datasets — sources & provenance

The raw edge lists below are the three real-world benchmarks used in the paper
(Section "Results on real graphs"). They are tiny, undirected, unweighted edge
lists (`u v` per line). `repro/prepare_data.py` turns them into the train/val/test
parquet pairs the dataset configs consume.

All three were obtained from the FUGAL release
(<https://github.com/idea-iitd/Fugal/tree/main/data/real>), which itself sources
them from the Network Repository / MultiMAGNA++. Please respect the original
dataset licenses and cite the original sources below.

| Directory | File(s) | Nodes / Edges | Original source & citation |
|-----------|---------|---------------|----------------------------|
| `ca-netscience/` | `ca-netscience.txt` | 379 / 914 | Coauthorship network of network scientists; Newman 2006 (`PhysRevE.74.036104`). Network Repository. |
| `inf-euroroad/` | `inf-euroroad.txt` | 1,174 / 1,417 | European road network; Šubelj & Bajec 2011 (`vsubelj2011robust`). Network Repository. |
| `MultiMagna/` | `yeast{0,5,10,15,20,25}_Y2H1.txt` | 1,004 / 8,323 (base) | Yeast PPI network with low-confidence noisy variants; Vijayan & Milenković 2017 (`10.1109/TCBB.2017.2740381`), MultiMAGNA++. |

## Noise models (see `repro/prepare_data.py`)

- **MultiMAGNA (edge-addition).** `yeast0_Y2H1.txt` is the trusted base graph
  (8,323 interactions). `yeast{5..25}_Y2H1.txt` add `q%` low-confidence edges on
  the **same node set**, so the base is an induced subgraph of every variant, the
  true node correspondence is the identity, and the maximum number of common edges
  is 8,323. Pairs are `(yeast0, yeast_q)`.
- **Edge add/remove (the harder benchmark, tab:realworld-noisy).** Erdős–Rényi
  edge-addition-removal noise (`loaders.generators.noise_erdos_renyi`) at the
  graph's own average degree. Faithful to the FUGAL `*_dataset.ipynb` notebooks:
  - `ca-netscience` / `inf-euroroad`: `graph_A = G`, `graph_B = noised(G)`;
    `noise1` = 0.1, `noise2` = 0.2.
  - `yeast25LC` (`yeast0_25_noise*`): `graph_A = yeast0` (trusted base, 8,323
    edges), `graph_B = noised(yeast25)` (the q=25% low-confidence variant);
    `noise005` = 0.05, `noise01` = 0.1.

## Reproducibility note

The published numbers were produced with **no fixed seed** and a **single noise
realization** per cell. `prepare_data.py` therefore defaults `--seed` to `None`
(matching the paper); pass an integer for a deterministic rebuild.
