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
- **Edge add/remove (the harder benchmark).** For `ca-netscience`, `inf-euroroad`
  and the harder yeast benchmark (`yeast25LC` in the paper), we corrupt the graph
  with the same Erdős–Rényi edge-addition-removal noise model used for the
  synthetic graphs, at the graph's own average degree
  (`loaders.generators.noise_erdos_renyi`).

## Reproducibility note

The published numbers were produced with **no fixed seed** and a **single noise
realization** per cell. `prepare_data.py` therefore defaults `--seed` to `None`
(matching the paper); pass an integer for a deterministic rebuild.
