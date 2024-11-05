from abc import ABC, abstractmethod
from omegaconf import  OmegaConf, DictConfig
import torch
import os

from models import get_model, get_siamese, get_siamese_name, train_siamese, test_siamese
from loaders import siamese_loader, get_data, get_data_test
import loaders.data_generator as dg
from toolbox.utils import save_json, load_json

class Pipeline(ABC):
    def __init__(self, num_models: int,  path_models: str):
        self.num_models = num_models
        self.path_models = path_models
        self.list_models = []
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

    @abstractmethod
    def loop(self, cfg_data: DictConfig, path_dataset: str) -> None:
        pass

class Chaining(Pipeline):
    def __init__(self, num_models: int,  path_models: str):
        super().__init__(num_models, path_models)

    def build_ind(self, data, siamese):
        loader = siamese_loader(data, batch_size=self.batch_size, shuffle=False)
        ind_data = dg.all_ind(loader, siamese, self.device)
        new = dg.make_data_from_ind_label(data, ind_data)
        return new

    def train_data(self, data_train, data_val, siamese, L=1):
        train_loader = siamese_loader(data_train, batch_size=self.batch_size, shuffle=True)
        val_loader = siamese_loader(data_val, batch_size=self.batch_size, shuffle=False)
        
        train_siamese(train_loader, val_loader, siamese, self.device, self.path_models, self.cfg.training.epochs, self.cfg.training.log_freq, L=L)
        
        new_train = self.build_ind(data_train, siamese)
        new_val = self.build_ind(data_val, siamese)
        return new_train, new_val

    def train(self, cfg: DictConfig, path_dataset: str) -> None:
        self.path_dataset = path_dataset
        self.cfg = cfg
        self.batch_size = self.cfg.training.batch_size
        node_embedder = get_model(self.cfg.model)
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        save_json(os.path.join(self.path_models, 'config.json'), config_dict)

        siamese = get_siamese(node_embedder)
        siamese.set_training_mode(lr=self.cfg.training.lr, scheduler_decay=self.cfg.training.scheduler_decay, scheduler_step=self.cfg.training.scheduler_step, lr_stop=self.cfg.training.lr_stop)

        data_train, data_val = get_data(self.cfg.dataset, self.path_dataset)
        new_train, new_val = self.train_data(data_train, data_val, siamese, L=0)
        
        siamese = get_siamese(node_embedder)
        siamese.set_training_mode(lr=self.cfg.training.lr, scheduler_decay=self.cfg.training.scheduler_decay, scheduler_step=self.cfg.training.scheduler_step, lr_stop=self.cfg.training.lr_stop)
        for i in range(1, self.num_models):
            new_train, new_val = self.train_data(new_train, new_val, siamese, L=i)

    def loop(self, cfg_data: DictConfig, path_dataset: str):
        config = load_json(os.path.join(self.path_models, 'config.json'))
        data_test = get_data_test(cfg_data, path_dataset)
        self.batch_size = config['training']['batch_size']
        test_loader = siamese_loader(data_test, batch_size=self.batch_size, shuffle=False)

        self.list_models = sorted([file for file in os.listdir(self.path_models) if file.endswith('.ckpt')])
        
        for model_name in self.list_models:
            siamese = get_siamese_name(os.path.join(self.path_models,model_name), config['model'])
            test_siamese(test_loader, siamese, self.device)
            data_test = self.build_ind(data_test, siamese)
            test_loader = siamese_loader(data_test, batch_size=self.batch_size, shuffle=False)

