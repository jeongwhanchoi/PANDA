"""
Test rewired GNN performance on graph classifiation benchmarks.
"""

from attrdict import AttrDict
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj
from experiments.graph_classification import Experiment
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, digl, panda
import os
import wandb
import random


def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

DETERMINISTIC = False
SEED = 42
if DETERMINISTIC:
    init_seed(SEED)

mutag = list(TUDataset(root="data", name="MUTAG"))
enzymes = list(TUDataset(root="data", name="ENZYMES"))
proteins = list(TUDataset(root="data", name="PROTEINS"))
collab = list(TUDataset(root="data", name="COLLAB"))
imdb = list(TUDataset(root="data", name="IMDB-BINARY"))
reddit = list(TUDataset(root="data", name="REDDIT-BINARY"))
datasets = {"reddit": reddit, "imdb": imdb, "mutag": mutag, "enzymes": enzymes, "proteins": proteins, "collab": collab}

for key in datasets:
    if key in ["reddit", "imdb", "collab"]:
        for graph in datasets[key]:
            n = graph.num_nodes
            graph.x = torch.ones((n,1))

def average_spectral_gap(dataset):
    # computes the average spectral gap out of all graphs in a dataset
    spectral_gaps = []
    for graph in dataset:
        G = to_networkx(graph, to_undirected=True)
        spectral_gap = rewiring.spectral_gap(G)
        spectral_gaps.append(spectral_gap)
    return sum(spectral_gaps) / len(spectral_gaps)

def log_to_file(message, filename="results/graph_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

hyperparams = {
    "mutag": AttrDict({"output_dim": 2}),
    "enzymes": AttrDict({"output_dim": 6}),
    "proteins": AttrDict({"output_dim": 2}),
    "collab": AttrDict({"output_dim": 3}),
    "imdb": AttrDict({"output_dim": 2}),
    "reddit": AttrDict({"output_dim": 2})
}

result_list = []
args = get_args_from_input()

if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}

for key in datasets:
    args += hyperparams[key]
    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []
    energies = []
    print(f"TESTING: {key} ({args.rewiring})")
    dataset = datasets[key]
    if args.rewiring == "fosr":
        for i in range(len(dataset)):
            edge_index, edge_type, _ = fosr.edge_rewire(dataset[i].edge_index.numpy(), num_iterations=args.num_iterations)
            dataset[i].edge_index = torch.tensor(edge_index)
            dataset[i].edge_type = torch.tensor(edge_type)
    elif args.rewiring == "sdrf":
        for i in range(len(dataset)):
            dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(dataset[i], loops=args.num_iterations, remove_edges=False, is_undirected=True)
    elif args.rewiring == "digl":
        for i in range(len(dataset)):
            dataset[i].edge_index = digl.rewire(dataset[i], alpha=0.1, eps=0.05)
            m = dataset[i].edge_index.shape[1]
            dataset[i].edge_type = torch.tensor(np.zeros(m, dtype=np.int64))
    elif args.rewiring == "panda" and args.centrality != "degree_simple":
        for i in range(len(dataset)):
            dataset[i].centrality = panda.measure_centrality(dataset[i], centrality_measure=args.centrality,
                                                            index=i, save_path=f"centrality/{key}")
    else:
        pass

    if args.wandb:
        os.environ["WANDB_MODE"] = "run"
    else:
        os.environ["WANDB_MODE"] = "disabled"

    if args.wandb:
        if args.wandb_run_name != None:
            wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=args, allow_val_change=True, name=args.wandb_run_name)
        else:
            wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=args, allow_val_change=True)
        args = wandb.config
        wandb.config.update(args)
        wandb.define_metric("epoch_step")  # Customize axes - https://docs.wandb.ai/guides/track/log

    for trial in range(args.num_trials):
        train_acc, validation_acc, test_acc, energy = Experiment(args=args, dataset=dataset).run()
        train_accuracies.append(train_acc)
        validation_accuracies.append(validation_acc)
        test_accuracies.append(test_acc)
        energies.append(energy)
        print("trial:", trial)
        print("test acc:", test_acc)
    train_mean = 100 * np.mean(train_accuracies)
    val_mean = 100 * np.mean(validation_accuracies)
    test_mean = 100 * np.mean(test_accuracies)
    energy_mean = 100 * np.mean(energies)
    train_ci = 200 * np.std(train_accuracies)/(args.num_trials ** 0.5)
    val_ci = 200 * np.std(validation_accuracies)/(args.num_trials ** 0.5)
    test_ci = 200 * np.std(test_accuracies)/(args.num_trials ** 0.5)
    energy_ci = 200 * np.std(energies)/(args.num_trials ** 0.5)
    

    if not args.wandb:
        if args.rewiring != "none":
            log_to_file(f"RESULTS FOR {key} ({args.rewiring}), {args.num_iterations} ITERATIONS:\n")
        log_to_file(f"average acc: {test_mean}\n")
        log_to_file(f"plus/minus:  {test_ci}\n\n")
    else:
        if args.rewiring != "none":
            print(f"RESULTS FOR {key} ({args.rewiring}), {args.num_iterations} ITERATIONS:\n")
        print(f"average acc: {test_mean}\n")
        print(f"plus/minus:  {test_ci}\n\n")

    results = {
        "dataset": key,
        "rewiring": args.rewiring,
        "layer_type": args.layer_type,
        "test_mean": test_mean,
        "test_ci": test_ci,
        "val_mean": val_mean,
        "val_ci": val_ci,
        "train_mean": train_mean,
        "train_ci": train_ci,
        "energy_mean": energy_mean,
        "energy_ci": energy_ci
        }
    result_list.append(results)
    if args.wandb:
        wandb.log(results)
df = pd.DataFrame(result_list)
if args.wandb:
    metric_table = wandb.Table(dataframe=df)
    wandb_run.log({"metric_table": metric_table})
    wandb_run.finish()
else:
    with open('results/graph_classification_fa.csv', 'a') as f:
        df.to_csv(f, mode='a', header=f.tell()==0)
