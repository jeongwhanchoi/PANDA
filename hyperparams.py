import argparse
import ast
from attrdict import AttrDict

def get_args_from_input():
	parser = argparse.ArgumentParser(description='modify network parameters', argument_default=argparse.SUPPRESS)

	parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--max_epochs', type=int, default=300, help='maximum number of epochs for training')
	parser.add_argument('--layer_type', default='PANDA-GCN', help='type of layer in GNN (GCN, GIN, GAT, etc.)')
	parser.add_argument('--display', type=bool, default=True, help='toggle display messages showing training progress')
	parser.add_argument('--device', default=0, type=int, help='the gpu to use')
	parser.add_argument('--eval_every', type=int, default=1, help='calculate validation/test accuracy every X epochs')
	parser.add_argument('--stopping_criterion', type=str, default="validation", help='model stops training when this criterion stops improving (can be train, validation, or test)')
	parser.add_argument('--stopping_threshold', type=float, default=1.01, help="model perceives no improvement when it does worse than (best loss) * T")
	parser.add_argument('--patience', type=int, default=100, help='model stops training after P epochs with no improvement')
	parser.add_argument('--train_fraction', type=float, default=0.8, help='fraction of the dataset to be used for training')
	parser.add_argument('--validation_fraction', type=float, default=0.1, help='fraction of the dataset to be used for validation')
	parser.add_argument('--test_fraction', type=float, default=0.1, help='fraction of the dataset to be used for testing')
	parser.add_argument('--dropout', type=float, default=0.5, help='layer dropout probability')
	parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay added to loss function')
	parser.add_argument('--input_dim', type=int, default=None, help='input dimension')
	parser.add_argument('--output_dim', type=int, default=None, help='output dimension')
	parser.add_argument('--hidden_dim', type=int, default=64, help='width of hidden layer')
	parser.add_argument('--hidden_layers', type=ast.literal_eval, default=None, help='list containing dimensions of all hidden layers')
	parser.add_argument('--num_layers', type=int, default=4, help='number of hidden layers')
	parser.add_argument('--batch_size', type=int, default=64, help='number of samples in each training batch')
	parser.add_argument('--num_trials', type=int, default=1, help='number of times the network is trained'),
	parser.add_argument('--rewiring', type=str, default='none', help='type of rewiring to be performed'),
	parser.add_argument('--num_iterations', type=int, default=10, help='number of iterations of rewiring')
	parser.add_argument('--alpha', type=float, default=0.1, help='alpha hyperparameter for DIGL')
	parser.add_argument('--k', type=int, help='k hyperparameter for DIGL')
	parser.add_argument('--eps', type=float, default=0.001, help='epsilon hyperparameter for DIGL')
	parser.add_argument('--num_relations', type=int, default=2, help='num_relations')
	parser.add_argument('--dataset', type=str, default='mutag', help='name of dataset to use')
	parser.add_argument('--last_layer_fa', type=bool, default=False, help='whether or not to make last layer fully adjacent')
	parser.add_argument('--top_k', type=int, default=5, help='top k nodes to be selected for expansion')
	parser.add_argument('--exp_factor', type=float, default=2, help='expansion factor for PANDA-GCN')
	parser.add_argument('--centrality', type=str, default='degree_simple', help='degree, closeness, betweenness, eigenvector, katz, pagerank')
	parser.add_argument('--order', type=str, default='nn_ne_en', help='order of propagation')
	# wandb
	parser.add_argument('--wandb', default=False, action='store_true', help="flag if logging to wandb")
	parser.add_argument('--wandb_sweep', action='store_true',help="flag if sweeping")  # if not it picks up params in greed_params
	parser.add_argument('--wandb_entity', default="username", type=str)
	parser.add_argument('--wandb_project', default="PANDA-GCN", type=str)
	parser.add_argument('--wandb_run_name', default=None, type=str)
	parser.add_argument('--run_track_reports', action='store_true', help="run_track_reports")
	parser.add_argument('--save_wandb_reports', action='store_true', help="save_wandb_reports")

	arg_values = parser.parse_args()
	return AttrDict(vars(arg_values))