from abc import ABC, abstractmethod
from omegaconf import  OmegaConf, DictConfig
import torch
import os
import wandb
from typing import Optional

from models import get_model, get_siamese, get_siamese_name, train_siamese
from models.pl_model import Siamese_Node
from loaders.data_generator import GAP_Generator
from loaders import siamese_loader, get_data, get_data_test
import loaders.data_generator as dg
from toolbox.utils import save_json, load_json
import numpy as np
from toolbox.metrics import all_qap_chain

class Pipeline(ABC):
    def __init__(self, path_models: str, num_models: int | None = None):
        
        self.path_models = path_models
        if num_models:
            self.num_models = num_models
            self.list_models = []
        else:
            self.list_models = sorted([file for file in os.listdir(self.path_models) if file.endswith('.ckpt')])
            self.num_models = len(self.list_models)
        
        self.set_device()

    def set_device(self):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Device: {self.device}")

    @abstractmethod
    def train(self, cfg: DictConfig) -> None:
        pass

    #@abstractmethod
    #def loop(self, cfg_data: DictConfig, path_dataset: str) -> None:
    #    pass

class Chaining(Pipeline):
    def __init__(self, path_models: str, num_models: int | None = None):
        super().__init__(path_models, num_models)

    def build_ind(self, data, siamese, verbose=False, compute_nce=False):
        loader = siamese_loader(data, batch_size=self.batch_size, shuffle=False)
        ind_data, nce = dg.all_ind(loader, siamese, self.device, compute_nce)
        new = dg.make_data_from_ind_label(data, ind_data)
        if verbose:
            return new, ind_data, nce
        else:
            return new, None, nce

    def train_data(self, data_train, data_val, siamese, L):
        train_loader = siamese_loader(data_train, batch_size=self.batch_size, shuffle=True)
        val_loader = siamese_loader(data_val, batch_size=self.batch_size, shuffle=False)
        
        train_siamese(train_loader, val_loader, siamese, self.device, self.path_models, self.cfg.training.epochs, self.cfg.training.log_freq, L, self.cfg.training.lr_stop, self.cfg.training.wandb)
        
        new_train, _, _ = self.build_ind(data_train, siamese)
        new_val, _ , _ = self.build_ind(data_val, siamese)
        if self.cfg.training.wandb:
            wandb.finish()
            #os.system(f"rm -rf {self.path_models}/wandb")
        return new_train, new_val

    def train(self, cfg: DictConfig, path_dataset: str) -> None:
        self.path_dataset = path_dataset
        self.cfg = cfg
        self.batch_size = self.cfg.training.batch_size
        self.saving = True
        node_embedder = get_model(self.cfg.model)
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        save_json(os.path.join(self.path_models, 'config.json'), config_dict)

        siamese = get_siamese(node_embedder)
        siamese.set_training_mode(lr=self.cfg.training.lr, scheduler_decay=self.cfg.training.scheduler_decay, scheduler_step=self.cfg.training.scheduler_step, lr_stop=self.cfg.training.lr_stop)

        data_train, data_val = get_data(self.cfg.dataset, self.path_dataset, self.saving)
        new_train, new_val = self.train_data(data_train, data_val, siamese, L=0)
        
        siamese = get_siamese(node_embedder)
        siamese.set_training_mode(lr=self.cfg.training.lr, scheduler_decay=self.cfg.training.scheduler_decay, scheduler_step=self.cfg.training.scheduler_step, lr_stop=self.cfg.training.lr_stop)
        for i in range(1, self.num_models):
            new_train, new_val = self.train_data(new_train, new_val, siamese, L=i)

    def loop(self, cfg_data: DictConfig, 
                path_dataset: str, 
                L :  int | None = None, 
                N_max : int | None = None,
                patience : int = 10, #4
                verbose : bool = False,
                eps : float = 0.001,
                batch_size : int | None = None,
                ind : int | None = None) -> Optional[tuple]:
        config = load_json(os.path.join(self.path_models, 'config.json'))
        data_test = get_data_test(cfg_data, path_dataset)
        if ind is None:
            if batch_size:
                self.batch_size = batch_size
            else:
                self.batch_size = config['training']['batch_size']
        else:
            data_test.data = data_test.data[ind]
            self.batch_size = 1
                    
        test_loader = siamese_loader(data_test, batch_size=self.batch_size, shuffle=False)
        
        if L:
            L = min(L, self.num_models)
        else:
            L = self.num_models

        if verbose:
            all_ind_data = []

        current_max_nce = eps
        best_nloop = 0
        stop = patience
        eps = eps
        for (nloop, model_name) in enumerate(self.list_models[:L]):
            siamese = get_siamese_name(os.path.join(self.path_models,model_name), config['model'])
            new_data_test, current_ind, all_nce = self.build_ind(data_test, siamese, verbose, compute_nce=True)
            test_nce = all_nce.mean()
            print(f"Model {model_name} has test nce: {test_nce}")
            delta = test_nce - current_max_nce
            if delta > 0:
                current_max_nce = test_nce
                best_model = siamese
                best_data = data_test
                best_nloop = nloop
            if delta/current_max_nce > eps:
                stop = patience
            else:
                stop -= 1
                if stop == 0:
                    break
            data_test = new_data_test
            
            if verbose:
                all_ind_data.append(current_ind)
            test_loader = siamese_loader(data_test, batch_size=self.batch_size, shuffle=False)
        
        if N_max and stop >0:
            for i in range(N_max):
                new_data_test, current_ind, all_nce = self.build_ind(data_test, siamese, verbose, compute_nce=True)
                test_nce= all_nce.mean()
                print(f"Model {model_name}-{i} has test nce: {test_nce}")
                delta = test_nce - current_max_nce
                if delta > 0:
                    current_max_nce = test_nce
                    best_model = siamese
                    best_data = data_test
                    best_nloop += 1
                if delta/current_max_nce > eps:
                    stop = patience
                else:
                    stop -= 1
                    if stop == 0:
                        break
                if i == N_max-1:
                    break
                data_test = new_data_test
                if verbose:
                    all_ind_data.append(current_ind)
                test_loader = siamese_loader(data_test, batch_size=self.batch_size, shuffle=False)

        if verbose:
            return np.array(all_ind_data), best_model, best_data, best_nloop
        else:
            return best_model, best_data, best_nloop

    def loop_siamese(self, dataset: list, 
                siamese: Siamese_Node, 
                N_max : int | None = None,
                patience : int = 10, #10 #4
                verbose : bool = False,
                eps : float = 0.001,
                ind : int | None = None) -> Optional[tuple]:
        
        data_test = []
        if ind is None:
            self.batch_size = 1
        else:
            data_test.append(dataset[ind])
            self.batch_size = 1

        stop = patience
        current_max_nce = eps
        best_nloop = 0
        all_ind_data = []
        for i in range(N_max):
            new_data_test, current_ind, all_nce = self.build_ind(data_test, siamese, verbose, compute_nce=True)
            test_nce= all_nce.mean()
            print(f"Model {i} has test nce: {test_nce}")
            delta = test_nce - current_max_nce
            if delta > 0:
                current_max_nce = test_nce
                best_model = siamese
                best_data = data_test
                best_nloop += 1
            if delta/current_max_nce > eps:
                stop = patience
            else:
                stop -= 1
                if stop == 0:
                    break
            if i == N_max-1:
                break
            data_test = new_data_test
            if verbose:
                all_ind_data.append(current_ind)
            test_loader = siamese_loader(data_test, batch_size=self.batch_size, shuffle=False)
        del data_test
        if verbose:
            return np.array(all_ind_data), best_model, best_data, best_nloop
        else:
            return best_model, best_data, best_nloop

        

class Streaming(Pipeline):
    def __init__(self, path_models: str, num_models: int | None = None):
        super().__init__(path_models, num_models)

    def train_data(self, data_train, data_val, siamese, L):
        train_loader = siamese_loader(data_train, batch_size=self.batch_size, shuffle=True)
        val_loader = siamese_loader(data_val, batch_size=self.batch_size, shuffle=False)
        
        train_siamese(train_loader, val_loader, siamese, self.device, self.path_models, self.cfg.training.epochs, self.cfg.training.log_freq, L, self.cfg.training.lr_stop, self.cfg.training.wandb)
        
        if self.cfg.training.wandb:
            wandb.finish()
            #os.system(f"rm -rf {self.path_models}/wandb")
        pass

    def train(self, cfg: DictConfig, path_dataset: str) -> None:
        self.path_dataset = path_dataset
        self.cfg = cfg
        self.batch_size = self.cfg.training.batch_size
        self.saving = cfg.saving
        node_embedder = get_model(self.cfg.model)
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        save_json(os.path.join(self.path_models, 'config.json'), config_dict)

        siamese = get_siamese(node_embedder)
        siamese.set_training_mode(lr=self.cfg.training.lr, scheduler_decay=self.cfg.training.scheduler_decay, scheduler_step=self.cfg.training.scheduler_step, lr_stop=self.cfg.training.lr_stop)

        data_train, data_val = get_data(self.cfg.dataset, self.path_dataset, self.saving)
        self.train_data(data_train, data_val, siamese, L=0)
        
        siamese = get_siamese(node_embedder)
        siamese.set_training_mode(lr=self.cfg.training.lr, scheduler_decay=self.cfg.training.scheduler_decay, scheduler_step=self.cfg.training.scheduler_step, lr_stop=self.cfg.training.lr_stop)
        for i in range(1, self.num_models):
            data_train, data_val = get_data(self.cfg.dataset, self.path_dataset, self.saving)
            self.train_data(data_train, data_val, siamese, L=i)
            
    def test(self, cfg_data: DictConfig, path_dataset: str, verbose: bool = False):
        data_test = get_data_test(cfg_data, path_dataset)
        config = load_json(os.path.join(self.path_models, 'config.json'))
        #self.batch_size = config['training']['batch_size']
        test_loader = siamese_loader(data_test, batch_size=1, shuffle=False)
        model_name = self.list_models[-1]
        siamese = get_siamese_name(os.path.join(self.path_models,model_name), config['model'])
        return all_qap_chain(test_loader, siamese, self.device, verbose)
