from omegaconf import DictConfig, OmegaConf
import hydra
import os
from pathlib import Path

from models.pipeline import Chaining


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.root_dir is None:
        ROOT_DIR = Path.home()
    else:
        ROOT_DIR = os.path.abspath(cfg.root_dir)
    PB_DIR = os.path.join(ROOT_DIR,'experiments-gnn-gap/')
    DATA_PB_DIR = os.path.join(PB_DIR,'data/')
    path_models = os.path.join(PB_DIR, cfg.pipeline.path_models)
    chain = Chaining(cfg.pipeline.L, path_models)
    chain.train(cfg, DATA_PB_DIR)
    chain.loop(cfg.dataset, DATA_PB_DIR)

if __name__ == "__main__":
    main()

    
    
