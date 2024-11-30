from omegaconf import DictConfig, OmegaConf
import hydra
import os
from pathlib import Path
import numpy as np

from loaders import siamese_loader, get_data_test
from toolbox.baselines import all_qap_scipy
from toolbox.utils import check_dir


def make_chunk_faq(noise, n_chunk, cfg_data, path_dataset):
    chunk_size = 10
    cfg_data.noise = noise
    print(f"computing FAQ for noise: {noise}, chunk: {n_chunk}")
    data_test = get_data_test(cfg_data, path_dataset)
    data_test.data = data_test.data[n_chunk:n_chunk+chunk_size]
    test_loader = siamese_loader(data_test, batch_size=1, shuffle=False)

    all_planted, all_qap, all_d, all_acc, all_accd, all_fd, all_fproj, all_fqap, all_fplanted, all_conv_nit, all_nit = all_qap_scipy(test_loader, max_iter=5000, maxiter_faq=200, seeds=0, verbose=True)
    print(f"results FAQ for noise: {noise}, chunk: {n_chunk}: acc_faq {all_acc.mean()}, acc_d {all_accd.mean()}, conv_nit {all_conv_nit.mean()}, nit {all_nit.mean()}")
    return all_planted, all_qap, all_d, all_acc, all_accd, all_fd, all_fproj, all_fqap, all_fplanted, all_conv_nit, all_nit

@hydra.main(version_base=None, config_path="conf", config_name="config_faq")
def main(cfg: DictConfig):
    if cfg.root_dir is None:
        ROOT_DIR = Path.home()
    else:
        ROOT_DIR = os.path.abspath(cfg.root_dir)
    PB_DIR = os.path.join(ROOT_DIR,'experiments-gnn-gap/')
    DATA_PB_DIR = os.path.join(PB_DIR,'data/')
    path_logs = os.path.join(PB_DIR, cfg.path_logs)
    check_dir(path_logs)
    faq_file = os.path.join(path_logs, f"5000n_faq_noise{cfg.noise}_chunk{cfg.chunk}.npy")

    all_planted, all_qap, all_d, all_acc, all_accd, all_fd, all_fproj, all_fqap, all_fplanted, all_conv_nit, all_nit = make_chunk_faq(cfg.noise, cfg.chunk, cfg.dataset, DATA_PB_DIR)
    np.save(faq_file, [all_planted, all_qap, all_d, all_acc, all_accd, all_fd, all_fproj, all_fqap, all_fplanted, all_conv_nit, all_nit])
    print(f"Results for faq computed and saved in {faq_file}")

if __name__ == "__main__":
    main()
