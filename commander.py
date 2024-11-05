from omegaconf import DictConfig, OmegaConf
import hydra
import os
from pathlib import Path

#import torch

from models.pipeline import Chaining



global ROOT_DIR 
ROOT_DIR = Path.home()
global PB_DIR
PB_DIR = os.path.join(ROOT_DIR,'experiments-gnn-match/')
global DATA_PB_DIR 
DATA_PB_DIR = os.path.join(PB_DIR,'data/') 


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    chain = Chaining(cfg.pipeline.L, cfg.pipeline.path_models)
    #chain.train(cfg, DATA_PB_DIR)
    chain.loop(cfg.dataset, DATA_PB_DIR)

if __name__ == "__main__":
    main()

    
    
