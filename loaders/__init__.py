from torch.utils.data import DataLoader
import torch
import loaders.data_generator as dg


def collate_fn(samples_list, temperature=5.0):
    input1_list = [input1 for input1, _, _ in samples_list]
    input2_list = [input2 for _, input2, _ in samples_list]
    target_list = [target for _, _, target in samples_list]
    return (
        {"input": torch.stack(input1_list)},
        {"input": torch.stack(input2_list)},
        torch.stack(target_list),
    )


def siamese_loader(data, batch_size, shuffle=True):
    assert len(data) > 0
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        collate_fn=collate_fn,
    )


def get_data(cfg_data, path_dataset, saving=True):
    generator = dg.GAP_Generator
    gene_train = generator("train", cfg_data, path_dataset, saving=saving)
    gene_train.load_dataset()
    gene_val = generator("val", cfg_data, path_dataset, saving=saving)
    gene_val.load_dataset()
    return gene_train, gene_val


def get_data_nl(cfg_data, path_dataset, saving=False):
    generator = dg.Base_Generator
    gene_train = generator(
        name="yeast0_25_prod_noise",
        path_dataset="/home/lelarge/experiments-gnn-gap/data/MultiMagna/",
        num_examples=6,
        no_seed=True,
        saving=saving,
    )
    gene_train.load_dataset()
    return gene_train


def get_data_mm(cfg_data, path_dataset, saving=False):
    generator = dg.Base_Generator
    gene_train = generator(
        name="yeast0_25_noise005",
        path_dataset="/home/lelarge/experiments-gnn-gap/data/MultiMagna/",
        num_examples=6,
        no_seed=True,
        saving=saving,
    )
    # gene_train.load_dataset()
    gene_val = generator(
        # name="yeast05_25",
        name="yeast0_25_noise005_test",
        path_dataset="/home/lelarge/experiments-gnn-gap/data/MultiMagna/",
        num_examples=2,
        no_seed=True,
        saving=saving,
    )
    gene_val.load_dataset()
    return gene_train, gene_val


def get_data_road(cfg_data, path_dataset, saving=False):
    generator = dg.Base_Generator
    gene_train = generator(
        name="road_noise1_train",
        path_dataset="/home/lelarge/experiments-gnn-gap/data/inf-euroroad/",
        num_examples=6,
        no_seed=True,
        saving=saving,
    )
    # gene_train.load_dataset()
    gene_val = generator(
        # name="yeast05_25",
        name="road_noise1_test",
        path_dataset="/home/lelarge/experiments-gnn-gap/data/inf-euroroad/",
        num_examples=2,
        no_seed=True,
        saving=saving,
    )
    gene_val.load_dataset()
    return gene_train, gene_val


def get_data_ca(cfg_data, path_dataset, saving=False):
    generator = dg.Base_Generator
    gene_train = generator(
        name="ca_nets_noise1_train",
        path_dataset="/home/lelarge/experiments-gnn-gap/data/ca-netscience/",
        num_examples=6,
        no_seed=True,
        saving=saving,
    )
    # gene_train.load_dataset()
    gene_val = generator(
        # name="yeast05_25",
        name="ca_nets_noise1_test",
        path_dataset="/home/lelarge/experiments-gnn-gap/data/ca-netscience/",
        num_examples=2,
        no_seed=True,
        saving=saving,
    )
    gene_val.load_dataset()
    return gene_train, gene_val


def get_data_test(cfg_data, path_dataset, saving=True, label=True):
    generator = dg.GAP_Generator
    gene_test = generator("test", cfg_data, path_dataset, saving=saving, label=label)
    gene_test.load_dataset()
    return gene_test
