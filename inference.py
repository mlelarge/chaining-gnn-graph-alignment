from omegaconf import DictConfig, OmegaConf
import hydra
import os
from pathlib import Path
import numpy as np
from models.pipeline import Chaining
from loaders import siamese_loader
from toolbox.metrics import all_qap_chain
from toolbox.utils import check_dir

@hydra.main(version_base=None, config_path="conf", config_name="config_test")
def main(cfg: DictConfig):
    if cfg.root_dir is None:
        ROOT_DIR = Path.home()
    else:
        ROOT_DIR = os.path.abspath(cfg.root_dir)
    PB_DIR = os.path.join(ROOT_DIR,'experiments-gnn-gap/')
    DATA_PB_DIR = os.path.join(PB_DIR,'data/')
    path_models = os.path.join(PB_DIR, cfg.path_models)
    path_logs = os.path.join(PB_DIR, cfg.path_logs)
    check_dir(path_logs)
    noise = cfg.dataset.noise
    name_file = os.path.join(path_logs, f"results.npy")
    chain = Chaining(path_models)
    
    list_noises = [0, 0.05, 0.1, 0.15, 0.2, 0.25 , 0.3, 0.35]
    l = len(list_noises)
    n_ex = cfg.dataset.test.num_examples
    ALL_qap = np.zeros((l,n_ex))
    ALL_acc = np.zeros((l,n_ex))
    ALL_qap_p = np.zeros((l,n_ex))
    ALL_acc_p = np.zeros((l,n_ex))
    for (i, noise) in enumerate(list_noises):
        cfg.dataset.noise = noise
        best_model, best_data = chain.loop(cfg.dataset, DATA_PB_DIR, L = cfg.L, N_max=cfg.N_max)
        test_loader = siamese_loader(best_data, batch_size=1, shuffle=False)
        all_planted, all_qap, all_d, all_acc, all_accd = all_qap_chain(test_loader, best_model, best_model.device)
        ALL_qap[i,:] = all_qap
        ALL_acc[i,:] = all_acc
        ALL_qap_p[i,:] = all_d
        ALL_acc_p[i,:] = all_accd
        print(f"Results for noise {noise}: acc_qap={all_acc.mean()}, acc_proj={all_accd.mean()}")
    with open(name_file, 'wb') as f:
        np.save(f, list_noises)
        np.save(f, ALL_acc)
        np.save(f, ALL_qap)
        np.save(f, ALL_acc_p)
        np.save(f, ALL_qap_p)

if __name__ == "__main__":
    main()
