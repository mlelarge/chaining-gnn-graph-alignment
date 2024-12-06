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
    name_file = os.path.join(path_logs, f"ind_results_{cfg.dataset.noise}.npy")
    chain = Chaining(path_models)
    
    #list_noises = [0, 0.05, 0.1, 0.15, 0.2, 0.25 , 0.3, 0.35]
    #list_noises = [0, 0.05, 0.1, 0.15, 0.2]
    #l = len(list_noises)
    l = cfg.L + 2*cfg.N_max+3
    n_vertices = cfg.dataset.n_vertices
    n_ex = cfg.dataset.test.num_examples
    ALL_qap = np.zeros(n_ex)
    ALL_acc = np.zeros(n_ex)
    ALL_qap_p = np.zeros(n_ex)
    ALL_acc_p = np.zeros(n_ex)
    ALL_acc_max = np.zeros(n_ex)
    ALL_nit = np.zeros(n_ex)
    ALL_nloop = np.zeros(n_ex)
    ALL_ind = np.zeros((l,n_ex,2,n_vertices))

    all_ind, best_model, first_best_data, first_loop = chain.loop(cfg.dataset, DATA_PB_DIR, L = cfg.L, N_max=cfg.N_max, verbose = True)
    size = all_ind.shape[0]
    ALL_ind[:size,:] = all_ind

    for ind in range(n_ex):
        print(f"starting with sample {ind}")
        all_ind, best_model, best_data, best_nloop = chain.loop_siamese(first_best_data, best_model, N_max=cfg.N_max, verbose = True, ind = ind)
        #print(all_ind.shape)
        #print(first_loop)
        #print(best_nloop)
        new_size = all_ind.shape[0]
        ALL_ind[size:size+new_size,ind,:,:] = all_ind[:,0,:,:]
        test_loader = siamese_loader(best_data, batch_size=1, shuffle=False)
        all_planted, all_qap, all_d, all_acc, all_accd, all_accmax, all_nit = all_qap_chain(test_loader, best_model, best_model.device, verbose=True)
        ALL_qap[ind] = all_qap[0]
        ALL_acc[ind] = all_acc[0]
        ALL_qap_p[ind] = all_d[0]
        ALL_acc_p[ind] = all_accd[0]
        ALL_acc_max[ind] = all_accmax[0]
        ALL_nit[ind] = all_nit[0]
        ALL_nloop[ind] = best_nloop
        print(f"Results for noise {cfg.dataset.noise}: acc_qap={all_acc.mean()}, acc_proj={all_accd.mean()}")
    print(f"saving results in {name_file}")
    with open(name_file, 'wb') as f:
        np.save(f, cfg.dataset.noise)
        np.save(f, ALL_acc)
        np.save(f, ALL_qap)
        np.save(f, ALL_acc_p)
        np.save(f, ALL_qap_p)
        np.save(f, ALL_acc_max)
        np.save(f, ALL_nit)
        np.save(f, ALL_nloop)
        np.save(f, ALL_ind)

if __name__ == "__main__":
    main()
