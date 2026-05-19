
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from .graph_connector import InductiveGraphConnector


class NodeUpdateNetwork(MessagePassing):
    def __init__(self, in_emb_dim, out_emb_dim, num_bond_type=4, norm=False, batch_norm=False, edge_type=True,
                 dropout=0., aggr='mean'):
        super().__init__(aggr=aggr)
        self.edge_type = edge_type
        self.neigh_linear = nn.Linear(in_emb_dim, out_emb_dim)
        self.root_linear = nn.Linear(in_emb_dim, out_emb_dim)
        self.edge_emb = nn.Embedding(num_bond_type, out_emb_dim)
        nn.init.xavier_uniform_(self.edge_emb.weight.data)
        self.norm = norm
        self.batch_norm = nn.BatchNorm1d(out_emb_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr, edge_weight):

        if x.size(-1) != self.neigh_linear.in_features:
            raise RuntimeError(
                f"NodeUpdateNetwork input dim mismatch: "
                f"x.shape={tuple(x.shape)}, "
                f"expected last dim={self.neigh_linear.in_features}"
            )

        edge_embeddings = self.edge_emb(edge_attr[:, 0])
        neigh_x = self.neigh_linear(x)
        msg = self.propagate(edge_index, x=neigh_x, edge_attr=edge_embeddings, edge_weight=edge_weight)
        msg += self.root_linear(x)
        if self.norm:
            msg = F.normalize(msg, p=2, dim=-1)
        if self.batch_norm is not None:
            msg = self.batch_norm(msg)
        return self.dropout(self.act(msg))

    def message(self, x_j, edge_attr, edge_weight):
        if self.edge_type:
            return (x_j + edge_attr) * edge_weight
        return x_j * edge_weight


class Context_Encoder(nn.Module):
    def __init__(
        self,
        args,
        in_dim,
        num_layer,
        edge_hidden_dim,
        edge_n_layer,
        total_tasks,
        train_tasks,
        device=None,
        batch_norm=False,
        edge_type=True,
        top_k=-1,
        dropout=0.,
        pre_dropout=0.,
        nan_w=0.,
        nan_type='nan',
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.dropout = dropout
        self.total_tasks = total_tasks
        self.num_layer = num_layer
        self.nan_w = nan_w
        self.nan_type = nan_type
        self.top_k = max(1, int(top_k)) if top_k is not None else 1

        self.pre_dropout = nn.Dropout(pre_dropout) if pre_dropout > 0 else None
        self.task_emb = nn.Embedding(total_tasks, in_dim)
        if train_tasks < total_tasks:
            self.task_emb.weight.data[train_tasks:, :] = 0

        structure_in_dim = in_dim * 2
        self.add_module(
            'structure_layer',
            NodeUpdateNetwork(
                in_emb_dim=structure_in_dim,
                out_emb_dim=in_dim,
                batch_norm=batch_norm,
                num_bond_type=2,
                dropout=dropout,
            ),
        )

        for i in range(num_layer):
            self.add_module(
                f'molqv_layer{i}',
                NodeUpdateNetwork(
                    in_emb_dim=in_dim,
                    out_emb_dim=in_dim,
                    batch_norm=batch_norm,
                    num_bond_type=5,
                    dropout=dropout,
                ),
            )

        self.graph_pooling = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim),
        )


    def forward(self):
        raise NotImplementedError

    @staticmethod
    def _select_proto_by_label(proto_feat, labels):
        flat_labels = labels.view(-1).long()
        return proto_feat.index_select(0, flat_labels)

    def _run_domain_prop(self, sq_emb, s_proto, q_pv, support_y, query_y):
        s_n, q_n = len(support_y), len(query_y)
        molpv_emb = torch.cat([sq_emb, s_proto, q_pv], dim=0)

        for i in range(self.num_layer):
            molpv_emb, edge_index, edge_weight, edge_type = InductiveGraphConnector.build_sp_graph(
                molpv_emb,
                support_y,
                query_y,
                len(s_proto),
                device=self.device,
                k=self.top_k,
            )
            data = Data(x=molpv_emb, edge_index=edge_index, edge_type=edge_type, edge_w=edge_weight)
            molpv_emb = self._modules[f'molqv_layer{i}'](data.x, data.edge_index, data.edge_type, data.edge_w)
            molpv_emb = molpv_emb.contiguous()

        support_feat = molpv_emb[:s_n, :]
        query_feat = molpv_emb[s_n:(s_n + q_n), :]
        s_proto_feat = molpv_emb[(s_n + q_n):(s_n + q_n + len(s_proto)), :]
        q_pv_feat = molpv_emb[(s_n + q_n + len(s_proto)):, :]

        support_proto_by_label = self._select_proto_by_label(s_proto_feat, support_y)
        support_context = torch.cat([support_feat, support_proto_by_label], dim=-1)
        query_context = torch.cat([query_feat, q_pv_feat], dim=-1)

        task_context = torch.cat([q_pv_feat, s_proto_feat], dim=0).mean(0)
        support_mol_context = support_feat
        query_mol_context = query_feat
        graph_emb = task_context + torch.sigmoid(support_mol_context.mean(0))

        return {
            's_label': support_y.contiguous(),
            'q_label': query_y.contiguous(),
            'graph_emb': graph_emb,
            'task_context': task_context,
            'support_mol_context': support_mol_context,
            'query_mol_context': query_mol_context,
            'support_context': support_context,
            'query_context': query_context,
            'q_pv_feat': q_pv_feat,
            's_proto_feat': s_proto_feat,
        }

    def forward_inductive(self, sample_emb, task_id, s_proto, q_pv, support_y, query_y):
        s_n, q_n = len(support_y), len(query_y)

        if self.pre_dropout is not None:
            sample_emb = self.pre_dropout(sample_emb)

        sample_emb, edge_index, edge_weight, edge_type = InductiveGraphConnector.build_structure_graph(
            sample_emb,
            support_y,
            query_y,
            device=self.device,
            k=self.top_k,
        )

        data = Data(x=sample_emb, edge_index=edge_index, edge_type=edge_type, edge_w=edge_weight)
        sample_emb = self._modules['structure_layer'](data.x, data.edge_index, data.edge_type, data.edge_w)

        sq_emb = sample_emb[: s_n + q_n, :]
        out = self._run_domain_prop(sq_emb, s_proto, q_pv, support_y, query_y)

        return (
            out['s_label'],
            out['q_label'],
            out['graph_emb'],
            out['task_context'],
            out['support_mol_context'],
            out['query_mol_context'],
            out['support_context'],
            out['query_context'],
            out['q_pv_feat'],
            out['s_proto_feat'],
        )

    def forward_transferable(
        self,
        src_sample_emb,
        tgt_sample_emb,
        src_s_proto,
        src_q_pv,
        src_support_y,
        src_query_y,
        tgt_s_proto,
        tgt_q_pv,
        tgt_support_y,
        tgt_query_y,
        return_graph=False,
    ):
        if self.pre_dropout is not None:
            src_sample_emb = self.pre_dropout(src_sample_emb)
            tgt_sample_emb = self.pre_dropout(tgt_sample_emb)

        joint_emb, edge_index, edge_weight, edge_type = \
            InductiveGraphConnector.build_joint_structure_graph(
                src_sample_emb,
                tgt_sample_emb,
                device=self.device,
                k=self.top_k,
                cross_k=self.top_k,
            )

        graph_info = {
            "edge_index": edge_index.detach().cpu(),
            "edge_weight": edge_weight.detach().cpu(),
            "edge_type": edge_type.detach().cpu(),
            "src_n": len(src_support_y) + len(src_query_y),
            "tgt_n": len(tgt_support_y) + len(tgt_query_y),
        }

        data = Data(x=joint_emb, edge_index=edge_index, edge_type=edge_type, edge_w=edge_weight)
        joint_emb = self._modules['structure_layer'](
            data.x, data.edge_index, data.edge_type, data.edge_w
        )

        src_real_n = len(src_support_y) + len(src_query_y)
        tgt_real_n = len(tgt_support_y) + len(tgt_query_y)
        src_sq_emb = joint_emb[:src_real_n, :]
        tgt_sq_emb = joint_emb[src_real_n:(src_real_n + tgt_real_n), :]

        src_out = self._run_domain_prop(src_sq_emb, src_s_proto, src_q_pv, src_support_y, src_query_y)
        tgt_out = self._run_domain_prop(tgt_sq_emb, tgt_s_proto, tgt_q_pv, tgt_support_y, tgt_query_y)

        joint_graph_emb = 0.5 * (src_out['graph_emb'] + tgt_out['graph_emb'])

        if return_graph:
            return src_out, tgt_out, joint_graph_emb, graph_info

        return src_out, tgt_out, joint_graph_emb
