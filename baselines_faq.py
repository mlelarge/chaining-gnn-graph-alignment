from omegaconf import DictConfig, OmegaConf
import hydra
import os
from pathlib import Path
import numpy as np

from loaders import siamese_loader, get_data_test
from toolbox.baselines import all_qap_scipy, baseline
from toolbox.utils import check_dir


def make_all_baseline(list_n, n_ex, cfg_data, path_dataset):
    l = len(list_n)
    ALL_B = np.zeros((l,n_ex))
    ALL_U = np.zeros((l,n_ex))
    ALL_A = np.zeros((l,n_ex))
    ALL_P = np.zeros((l,n_ex))
    for (i,n) in enumerate(list_n):
        cfg_data.noise = n
        print(f"computing baselines for noise: {n}")
        data_test = get_data_test(cfg_data, path_dataset)
        test_loader = siamese_loader(data_test, batch_size=1, shuffle=False)
        all_b, all_u, all_acc, all_p = baseline(test_loader)
        ALL_B[i,:] = all_b
        ALL_U[i,:] = all_u
        ALL_A[i,:] = all_acc
        ALL_P[i,:] = all_p
    return ALL_B, ALL_U, ALL_A, ALL_P


@hydra.main(version_base=None, config_path="conf", config_name="config_baselines")
def main(cfg: DictConfig):
    if cfg.root_dir is None:
        ROOT_DIR = Path.home()
    else:
        ROOT_DIR = os.path.abspath(cfg.root_dir)
    PB_DIR = os.path.join(ROOT_DIR,'experiments-gnn-gap/')
    DATA_PB_DIR = os.path.join(PB_DIR,'data/')
    path_logs = os.path.join(PB_DIR, cfg.path_logs)
    check_dir(path_logs)
    baseline_file = os.path.join(path_logs, f"baselines.npy")

    list_noises = [0, 0.05, 0.1, 0.15, 0.2]#, 0.25 , 0.3, 0.35]
    l = len(list_noises)
    n_ex = cfg.dataset.test.num_examples

    ALL_B, ALL_U, ALL_A, ALL_P = make_all_baseline(list_noises, n_ex, cfg.dataset, DATA_PB_DIR)
    with open(baseline_file, 'wb') as f:
        np.save(f, list_noises)
        np.save(f, ALL_B)
        np.save(f, ALL_U)
        np.save(f, ALL_A)
        np.save(f, ALL_P)
    print(f"Results for baselines computed and saved in {baseline_file}")


if __name__ == "__main__":
    main()