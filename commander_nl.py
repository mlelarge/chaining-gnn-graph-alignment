from omegaconf import DictConfig, OmegaConf
import hydra
import os
from pathlib import Path
from models.pipeline import Chaining


@hydra.main(version_base=None, config_path="conf", config_name="config_nl")
def main(cfg: DictConfig):
    if cfg.root_dir is None:
        ROOT_DIR = Path.home()
    else:
        ROOT_DIR = os.path.abspath(cfg.root_dir)
    PB_DIR = os.path.join(ROOT_DIR, "experiments-gnn-gap/")
    DATA_PB_DIR = os.path.join(PB_DIR, "data/")
    path_models = os.path.join(PB_DIR, cfg.pipeline.path_models)
    negate_B = getattr(cfg.pipeline, "negate_B", False)
    chain = Chaining(path_models, cfg.pipeline.L, use_labels=False, negate_B=negate_B)
    chain.train(cfg, DATA_PB_DIR)

    # Use FAQ warm-start loop for QAPlib instances, NCE-based loop otherwise
    use_faq = bool(getattr(cfg.dataset, "instance", None))
    chain.loop(cfg.dataset, DATA_PB_DIR, use_faq_warmstart=use_faq)


if __name__ == "__main__":
    main()
