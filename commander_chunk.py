"""Resumable chunked chain training for short (qos_dev) jobs.

Trains up to ``pipeline.chunk_size`` new chain links per invocation, resuming
from the checkpoints already in ``path_models`` (see Chaining.train_chunk).
Re-run until all ``pipeline.L`` links exist — each call replays the trained
links' feedback, warm-starts the next link from the last checkpoint, and trains
the chunk. Same config surface as commander.py.
"""
from pathlib import Path
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from models.pipeline import Chaining


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    ROOT_DIR = Path.home() if cfg.root_dir is None else Path(os.path.abspath(cfg.root_dir))
    PB_DIR = os.path.join(ROOT_DIR, "experiments-gnn-gap/")
    DATA_PB_DIR = os.path.join(PB_DIR, "data/")
    path_models = os.path.join(PB_DIR, cfg.pipeline.path_models)
    chunk_size = int(OmegaConf.select(cfg, "pipeline.chunk_size", default=2))

    chain = Chaining(path_models, cfg.pipeline.L)
    completed = chain.train_chunk(cfg, DATA_PB_DIR, chunk_size)
    done = completed >= cfg.pipeline.L
    print(f"CHUNK_RESULT completed={completed} L={cfg.pipeline.L} done={done}")


if __name__ == "__main__":
    main()
