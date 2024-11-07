from torch.utils.data import DataLoader
import torch
import loaders.data_generator as dg

def collate_fn(samples_list, temperature=5.):
    input1_list = [input1 for input1, _ , _ in samples_list]
    input2_list = [input2 for _, input2, _ in samples_list]
    target_list = [target for _, _, target in samples_list]
    return {'input': torch.stack(input1_list)}, {'input': torch.stack(input2_list)}, torch.stack(target_list)

def siamese_loader(data, batch_size, shuffle=True):
    assert len(data) > 0
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=8, collate_fn=collate_fn)

def get_data(cfg_data, path_dataset):
    generator = dg.GAP_Generator
    gene_train = generator('train', cfg_data, path_dataset)
    gene_train.load_dataset()
    gene_val = generator('val', cfg_data, path_dataset)
    gene_val.load_dataset()
    return gene_train, gene_val

def get_data_test(cfg_data, path_dataset):
    generator = dg.GAP_Generator
    gene_test = generator('test', cfg_data, path_dataset)
    gene_test.load_dataset()
    return gene_test