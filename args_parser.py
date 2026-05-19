import argparse
import math
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in {'true', '1', 'yes', 'y', 't'}:
        return True
    if v in {'false', '0', 'no', 'n', 'f'}:
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v}.')


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='run', type=str, help='Experiment name')
    parser.add_argument('--dump_path', default='logs/', type=str, help='Experiment dump path')
    parser.add_argument('--exp_id', default='', type=str, help='Experiment ID')
    parser.add_argument('--gpu', default='4', type=str)
    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--data_root', default='./data/', type=str)
    parser.add_argument('--dataset', default='sider', type=str)
    parser.add_argument('--test_dataset', default=None, type=str,
                        help='Target-domain dataset. Set None or same as --dataset for within-dataset few-shot.')

    parser.add_argument('--mol_num_layer', default=5, type=int)
    parser.add_argument('--GNN_emb_dim', default=256, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--JK', default='last', type=str)
    parser.add_argument('--mol_dropout', default=0.1, type=float)
    parser.add_argument('--mol_graph_pooling', default='mean', type=str)
    parser.add_argument('--mol_gnn_type', default='gin', type=str)
    parser.add_argument('--mol_batch_norm', default=1, type=int)
    parser.add_argument('--mol_pretrain_load_path', default=None)

    parser.add_argument('--rel_layer', default=2, type=int)
    parser.add_argument('--rel_edge_n_layer', default=2, type=int)
    parser.add_argument('--rel_top_k', default=None, type=int)
    parser.add_argument('--rel_edge_hidden_dim', default=100, type=int)
    parser.add_argument('--rel_dropout', default=0.1, type=float)
    parser.add_argument('--rel_pre_dropout', default=0.1, type=float)
    parser.add_argument('--rel_nan_w', default=1.0, type=float)
    parser.add_argument('--rel_nan_type', default='nan', type=str, choices=['nan', '0', '1'])
    parser.add_argument('--rel_batch_norm', default=1, type=int)
    parser.add_argument('--rel_edge_type', default=1, type=int)

    parser.add_argument('--inner_lr', default=0.5, type=float)
    parser.add_argument('--meta_lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--second_order', default=1, type=int)
    parser.add_argument('--inner_update_step', default=1, type=int)
    parser.add_argument('--update_step_test', default=1, type=int,
                        help='Number of inner adaptation steps at meta-test time. Defaults to inner_update_step.')

    parser.add_argument('--episode', default=3000, type=int)
    parser.add_argument('--n_support', default=10, type=int)
    parser.add_argument('--n_query', default=16, type=int)
    parser.add_argument('--eval_step', default=100, type=int)

    parser.add_argument('--nce_t', default=0.08, type=float)
    parser.add_argument('--contr_w', default=0.05, type=float)
    parser.add_argument('--proto_w', default=0.05, type=float)

    parser.add_argument('--pool_num', default=5, type=int)

    parser.add_argument('--adapter_hidden_dim', default=50, type=int)
    parser.add_argument('--mol_layer_norm', default=1, type=int)

    parser.add_argument('--checkpoint', default='./SPMM-main/Pretrain/checkpoint_SPMM.ckpt')
    parser.add_argument('--vocab_filename', default='./SPMM/vocab_bpe_300.txt')

    parser.add_argument('--cross_domain_source_weight', default=0.5, type=float)
    parser.add_argument('--cross_domain_target_weight', default=1.0, type=float)
    parser.add_argument('--target_train_per_class', default=None, type=int,
                        help='Number of labeled target molecules per class reserved for training.')
    parser.add_argument('--target_eval_min_per_class', default=None, type=int,
                        help='Minimum number of held-out target molecules per class for meta-testing.')

    args = parser.parse_args()
    args.device = 'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu'
    if args.test_dataset in {None, '', 'None'} or args.dataset == args.test_dataset:
        args.test_dataset = None

    if args.rel_top_k is None:
        args.rel_top_k = args.n_support - 1 if args.n_support > 1 else 1

    if args.update_step_test is None:
        args.update_step_test = args.inner_update_step

    per_class_needed = args.n_support + max(1, math.ceil(args.n_query / 2))
    if args.target_train_per_class is None:
        args.target_train_per_class = per_class_needed

    if args.target_eval_min_per_class is None:
        args.target_eval_min_per_class = per_class_needed

    return args
