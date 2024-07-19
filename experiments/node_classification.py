import torch
import numpy as np
from attrdict import AttrDict
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from math import inf

from models.node_model import GCN

default_args = AttrDict(
    {"learning_rate": 1e-3,
    "max_epochs": 1000000,
    "display": True,
    "device": None,
    "eval_every": 1,
    "stopping_criterion": "validation",
    "stopping_threshold": 1.01,
    "patience": 50,
    "train_fraction": 0.6,
    "validation_fraction": 0.2,
    "test_fraction": 0.2,
    "dropout": 0.0,
    "weight_decay": 1e-5,
    "hidden_dim": 32,
    "hidden_layers": None,
    "num_layers": 1,
    "batch_size": 64,
    "layer_type": "GCN",
    "num_relations": 1
    }
    )

class Experiment:
    def __init__(self, args=None, dataset=None, train_mask=None, validation_mask=None, test_mask=None):
        self.args = default_args + args
        self.dataset = dataset
        self.train_mask = train_mask
        self.validation_mask = validation_mask
        self.test_mask = test_mask
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.args.input_dim = self.dataset[0].x.shape[1]
        self.args.output_dim = torch.amax(self.dataset[0].y).item() + 1
        self.num_nodes = self.dataset[0].x.size(axis=0)

        if self.args.device is None:
            self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.hidden_layers is None:
            self.args.hidden_layers = [self.args.hidden_dim] * self.args.num_layers

        self.model = GCN(self.args).to(self.args.device)

        # randomly assign a train/validation/test split, or train/validation split if test already assigned
        if self.test_mask is None:
            node_indices = list(range(self.num_nodes))
            self.args.test_fraction = 1 - self.args.train_fraction - self.args.validation_fraction
            non_test, self.test_mask = train_test_split(node_indices, test_size=self.args.test_fraction)
            self.train_mask, self.validation_mask = train_test_split(non_test, test_size=self.args.validation_fraction/(self.args.validation_fraction + self.args.train_fraction))
        elif self.validation_mask is None:
            non_test = [i for i in range(self.num_nodes) if not i in self.test_mask]
            self.train_mask, self.validation_mask = train_test_split(non_test, test_size=self.args.validation_fraction/(self.args.validation_fraction + self.args.train_fraction))
        
    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer,  patience=25)

        if self.args.display:
            print("Starting training")
        best_test_acc = 0.0
        best_validation_acc = 0.0
        best_train_acc = 0.0
        train_goal = 0.0
        validation_goal = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        train_size = len(self.train_mask)
        batch = self.dataset.data.to(self.args.device)
        y = batch.y

        for epoch in range(self.args.max_epochs):
            self.model.train()
            total_loss = 0
            sample_size = 0
            optimizer.zero_grad()

            out = self.model(batch)
            loss = self.loss_fn(input=out[self.train_mask], target=y[self.train_mask])
            total_loss += loss.item()
            _, train_pred = out[self.train_mask].max(dim=1)
            train_correct = train_pred.eq(y[self.train_mask]).sum().item() / train_size

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

            new_best_str = ''

            if epoch % self.args.eval_every == 0:
                train_acc = self.eval(batch=batch, mask=self.train_mask)
                validation_acc = self.eval(batch=batch, mask=self.validation_mask)
                test_acc = self.eval(batch=batch, mask=self.test_mask)

                if self.args.stopping_criterion == "train":
                    if train_acc > train_goal:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve = 0
                        train_goal = train_acc * self.args.stopping_threshold
                        new_best_str = ' (new best train)'
                    elif train_acc > best_train_acc:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve += 1
                    else:
                        epochs_no_improve += 1
                elif self.args.stopping_criterion == 'validation':
                    if validation_acc > validation_goal:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve = 0
                        validation_goal = validation_acc * self.args.stopping_threshold
                        new_best_str = ' (new best validation)'
                    elif validation_acc > best_validation_acc:
                        best_train_acc = test_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve += 1
                    else:
                        epochs_no_improve += 1
                if self.args.display:
                    print(f'Epoch {epoch}, Train acc: {train_acc}, Validation acc: {validation_acc}{new_best_str}, Test acc: {test_acc}')
                if epochs_no_improve > self.args.patience:
                    if self.args.display:
                        print(f'{self.args.patience} epochs without improvement, stopping training')
                        print(f'Best train acc: {best_train_acc}, Best validation loss: {best_validation_acc}, Best test loss: {best_test_acc}')
                    return train_acc, validation_acc, test_acc

    def eval(self, batch, mask):
        self.model.eval()
        with torch.no_grad():
            sample_size = len(mask)
            _, pred = self.model(batch)[mask].max(dim=1)
            total_correct = pred.eq(batch.y[mask]).sum().item()
            acc = total_correct / sample_size
            return acc