
import torch
from copy import deepcopy
import torch.nn.functional as F

nan_type_dict = {'0': 1, '1': 2, 'nan': 3}


class InductiveGraphConnector:

    @staticmethod
    def _knn_graph(feat, top_k, device, edge_type_val=0):
        num_nodes = feat.size(0)
        if num_nodes <= 1:
            empty_index = torch.empty(2, 0, dtype=torch.long, device=device)
            empty_weight = torch.empty(0, 1, dtype=feat.dtype, device=device)
            empty_type = torch.empty(0, 1, dtype=torch.long, device=device)
            return empty_index, empty_weight, empty_type

        feat = F.normalize(feat, p=2, dim=-1)
        sim = torch.matmul(feat, feat.T)
        sim.fill_diagonal_(-1e12)
        k = min(max(1, int(top_k)), num_nodes - 1)
        topk = torch.topk(sim, k=k, dim=1)

        src = torch.arange(num_nodes, device=device).unsqueeze(1).expand(num_nodes, k).reshape(-1)
        dst = topk.indices.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
        edge_weight = topk.values.reshape(-1, 1)
        edge_type = torch.full((edge_index.size(1), 1), int(edge_type_val), dtype=torch.long, device=device)
        return edge_index, edge_weight, edge_type

    @staticmethod
    def _cross_knn_graph(src_feat, dst_feat, top_k, device, dst_shift=0, edge_type_val=1):
        src_n = src_feat.size(0)
        dst_n = dst_feat.size(0)
        if src_n == 0 or dst_n == 0:
            empty_index = torch.empty(2, 0, dtype=torch.long, device=device)
            empty_weight = torch.empty(0, 1, dtype=src_feat.dtype, device=device)
            empty_type = torch.empty(0, 1, dtype=torch.long, device=device)
            return empty_index, empty_weight, empty_type

        src_feat = F.normalize(src_feat, p=2, dim=-1)
        dst_feat = F.normalize(dst_feat, p=2, dim=-1)
        sim = torch.matmul(src_feat, dst_feat.T)
        k = min(max(1, int(top_k)), dst_n)
        topk = torch.topk(sim, k=k, dim=1)

        src = torch.arange(src_n, device=device).unsqueeze(1).expand(src_n, k).reshape(-1)
        dst = (topk.indices + int(dst_shift)).reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
        edge_weight = topk.values.reshape(-1, 1)
        edge_type = torch.full((edge_index.size(1), 1), int(edge_type_val), dtype=torch.long, device=device)
        return edge_index, edge_weight, edge_type

    @staticmethod
    def build_structure_graph(sample_emb, support_y, query_y, device=None, k=2, bi_direct=True):
        s_n, q_n = len(support_y), len(query_y)
        all_n = len(sample_emb)
        real_n = s_n + q_n

        real_feat = sample_emb[:real_n, :]
        mm_edge_index, mm_edge_weight, mm_edge_type = InductiveGraphConnector._knn_graph(real_feat, k, device, edge_type_val=0)

        if all_n > real_n:
            gen_feat = sample_emb[real_n:, :]
            full_feat = F.normalize(sample_emb, p=2, dim=-1)
            gen_norm = full_feat[real_n:, :]
            real_norm = full_feat[:real_n, :]
            sim_rg = torch.matmul(real_norm, gen_norm.T)
            gk = min(max(1, int(k)), gen_feat.size(0))
            topk_rg = torch.topk(sim_rg, k=gk, dim=1)
            src = torch.arange(real_n, device=device).unsqueeze(1).expand(real_n, gk).reshape(-1)
            dst = (topk_rg.indices + real_n).reshape(-1)
            rg_edge_index = torch.stack([src, dst], dim=0)
            rg_edge_weight = topk_rg.values.reshape(-1, 1)
            rg_edge_type = torch.ones((rg_edge_index.size(1), 1), dtype=torch.long, device=device)

            edge_index = torch.cat([mm_edge_index, rg_edge_index], dim=1)
            edge_weight = torch.cat([mm_edge_weight, rg_edge_weight], dim=0)
            edge_type = torch.cat([mm_edge_type, rg_edge_type], dim=0)
        else:
            edge_index, edge_weight, edge_type = mm_edge_index, mm_edge_weight, mm_edge_type

        if bi_direct and edge_index.numel() > 0:
            edge_index_rev = edge_index[[1, 0]]
            edge_weight_rev = edge_weight.clone()
            edge_type_rev = edge_type.clone()
            edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
            edge_weight = torch.cat([edge_weight, edge_weight_rev], dim=0)
            edge_type = torch.cat([edge_type, edge_type_rev], dim=0)

        return sample_emb, edge_index.to(device), edge_weight.to(device), edge_type.to(device)

    @staticmethod
    def build_joint_structure_graph(src_sample_emb, tgt_sample_emb, device=None, k=2, cross_k=None, bi_direct=True):
        if cross_k is None:
            cross_k = k

        src_n = src_sample_emb.size(0)
        tgt_n = tgt_sample_emb.size(0)
        all_feat = torch.cat([src_sample_emb, tgt_sample_emb], dim=0)

        src_edge_index, src_edge_weight, src_edge_type = InductiveGraphConnector._knn_graph(
            src_sample_emb, k, device, edge_type_val=0)
        tgt_edge_index, tgt_edge_weight, tgt_edge_type = InductiveGraphConnector._knn_graph(
            tgt_sample_emb, k, device, edge_type_val=0)
        if tgt_edge_index.numel() > 0:
            tgt_edge_index = tgt_edge_index + src_n

        cross_edge_index, cross_edge_weight, cross_edge_type = InductiveGraphConnector._cross_knn_graph(
            src_sample_emb, tgt_sample_emb, cross_k, device, dst_shift=src_n, edge_type_val=1
        )

        edge_index = torch.cat([src_edge_index, tgt_edge_index, cross_edge_index], dim=1)
        edge_weight = torch.cat([src_edge_weight, tgt_edge_weight, cross_edge_weight], dim=0)
        edge_type = torch.cat([src_edge_type, tgt_edge_type, cross_edge_type], dim=0)

        if bi_direct and edge_index.numel() > 0:
            edge_index_rev = edge_index[[1, 0]]
            edge_weight_rev = edge_weight.clone()
            edge_type_rev = edge_type.clone()
            edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
            edge_weight = torch.cat([edge_weight, edge_weight_rev], dim=0)
            edge_type = torch.cat([edge_type, edge_type_rev], dim=0)

        return all_feat, edge_index.to(device), edge_weight.to(device), edge_type.to(device)

    @staticmethod
    def build_sp_graph(sample_emb, support_y, query_y, sproto_n, device=None, k=2, bi_direct=True):
        s_n, q_n = len(support_y), len(query_y)
        if k > q_n:
            k = q_n
        all_n = len(sample_emb)
        support_feat, query_feat = sample_emb[:s_n, :], sample_emb[s_n:(s_n + q_n), :]
        s_proto_feat = sample_emb[(s_n + q_n): (s_n + q_n + sproto_n), :]
        q_pv_feat = sample_emb[(s_n + q_n + sproto_n):, :]

        support_idx = torch.arange(0, s_n)
        query_idx = torch.arange(s_n, s_n + q_n)
        s_proto_idx = torch.arange(s_n + q_n, s_n + q_n + sproto_n) # num = n_class
        q_pv_idx = torch.arange(s_n + q_n + sproto_n, all_n) # num = q_n

        edge_index = []
        edge_weight = []
        edge_type = []

        for i in range(q_n):
            edge_index.append([query_idx[i].item(), q_pv_idx[i].item()])
            edge_weight.append(1.0)
            edge_type.append(0)

        query_norm = F.normalize(query_feat, p=2, dim=-1)
        s_proto_norm = F.normalize(s_proto_feat, p=2, dim=-1)
        sim_qspro = torch.matmul(query_norm, s_proto_norm.T)
        for i in range(q_n):
            for j in range(sproto_n):
                edge_index.append([query_idx[i].item(), s_proto_idx[j].item()])
                edge_weight.append(sim_qspro[i, j].item())
                edge_type.append(1)

        support_norm = F.normalize(support_feat, p=2, dim=-1)
        sim_sspro = torch.matmul(support_norm, s_proto_norm.T)
        for i in range(s_n):
            s_label = int(support_y[i].item())
            edge_index.append([support_idx[i].item(), s_proto_idx[s_label].item()])
            edge_weight.append(sim_sspro[i, s_label].item())
            edge_type.append(2)

        q_pv_norm = F.normalize(q_pv_feat, p=2, dim=-1)
        sim_sqpv = torch.matmul(support_norm, q_pv_norm.T)
        topk_sqpv = torch.topk(sim_sqpv, k=max(1, min(k, q_pv_feat.size(0))), dim=1)
        for i in range(s_n):
            for j, sim in zip(topk_sqpv.indices[i], topk_sqpv.values[i]):
                edge_index.append([support_idx[i].item(), q_pv_idx[j].item()])
                edge_weight.append(sim.item())
                edge_type.append(3)

        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).T
        edge_weight = torch.tensor(edge_weight, dtype=sample_emb.dtype, device=device).unsqueeze(-1)
        edge_type = torch.tensor(edge_type, dtype=torch.long, device=device).unsqueeze(-1)
        if bi_direct and edge_index.numel() > 0:
            edge_index_rev = edge_index[[1, 0]]
            edge_weight_rev = edge_weight.clone()
            edge_type_rev = edge_type.clone()
            edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
            edge_weight = torch.cat([edge_weight, edge_weight_rev], dim=0)
            edge_type = torch.cat([edge_type, edge_type_rev], dim=0)

        return sample_emb, edge_index, edge_weight, edge_type
