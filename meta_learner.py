import csv
import logging
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.data import Batch
from dataset import FewshotMolDataset, dataset_sampler
from models import MAML, NCESoftmaxLoss, FACML

logger = logging.getLogger()


class MetaLearner:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        self.dataset = FewshotMolDataset(root=args.data_root, name=args.dataset)
        self.test_dataset = FewshotMolDataset(root=args.data_root, name=args.test_dataset) if args.test_dataset is not None else None

        self.source_train_task_range = list(self.dataset.train_task_range)
        self.target_train_pool_map = {}
        self.target_eval_pool_map = {}

        self.target_train_per_class = int(getattr(args, 'target_train_per_class', args.n_support + math.ceil(args.n_query / 2)))
        self.target_eval_min_per_class = int(getattr(args, 'target_eval_min_per_class', args.n_support + math.ceil(args.n_query / 2)))

        if self.test_dataset is not None:
            (
                self.target_train_pool_map,
                self.target_eval_pool_map,
            ) = self._build_target_train_eval_pools(self.test_dataset)
            self.target_train_task_range = sorted(self.target_train_pool_map.keys())
            self.target_eval_task_range = sorted(self.target_eval_pool_map.keys())
        else:
            self.target_train_task_range = []
            self.target_eval_task_range = self.dataset.test_task_range

        self.data_pv = torch.load(args.data_root + args.dataset + '/data_pv.pt', map_location=args.device)
        self.test_data_pv = torch.load(args.data_root + args.test_dataset + '/data_pv.pt', map_location=args.device) if args.test_dataset is not None else self.data_pv

        total_task_num = max(self.dataset.total_tasks, self.test_dataset.total_tasks) if self.test_dataset is not None else self.dataset.total_tasks
        train_task_num = self.dataset.total_tasks
        model = FACML(args=args, task_num=total_task_num, train_task_num=train_task_num).to(self.device)

        self.maml = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)
        self.opt = torch.optim.AdamW(self.maml.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)

        self.cls_criterion = nn.BCEWithLogitsLoss()
        self.nce_loss = NCESoftmaxLoss(t=args.nce_t)

        self.cross_domain_source_weight = float(getattr(args, 'cross_domain_source_weight', 1.0))
        self.cross_domain_target_weight = float(getattr(args, 'cross_domain_target_weight', 1.0))

        self.task_coordinator = None
        self.source_task_ids = []
        self.source_task_proto_bank = {}
        self.source_task_emb_bank = None

        logger.info(
            'Transferable setting: %d source train tasks, %d target train tasks, %d target eval tasks.',
            len(self.source_train_task_range),
            len(self.target_train_task_range),
            len(self.target_eval_task_range),
        )

    @staticmethod
    def _safe_squeeze_pv(pv):
        return pv.squeeze(1) if pv.dim() > 2 else pv

    @staticmethod
    def _sample_without_replacement(candidates, num, rng):
        candidates = list(candidates)
        if len(candidates) == 0 or num <= 0:
            return []
        num = min(len(candidates), int(num))
        return rng.choice(candidates, num, replace=False).tolist()

    def _build_target_train_eval_pools(self, dataset_obj):

        rng = np.random.default_rng(getattr(self.args, 'random_seed', 0))
        train_pool_map = {}
        eval_pool_map = {}
        min_total_after_support = int(self.args.n_query)

        for task_id in range(dataset_obj.total_tasks):
            class0_idx, class1_idx = dataset_obj.index_list[task_id]
            class0_idx = list(class0_idx)
            class1_idx = list(class1_idx)

            max_train0 = len(class0_idx) - self.target_eval_min_per_class
            max_train1 = len(class1_idx) - self.target_eval_min_per_class
            if max_train0 <= 0 or max_train1 <= 0:
                continue

            train_take0 = min(self.target_train_per_class, max_train0)
            train_take1 = min(self.target_train_per_class, max_train1)

            if train_take0 <= self.args.n_support or train_take1 <= self.args.n_support:
                continue

            train0 = self._sample_without_replacement(class0_idx, train_take0, rng)
            train1 = self._sample_without_replacement(class1_idx, train_take1, rng)

            eval0 = [idx for idx in class0_idx if idx not in train0]
            eval1 = [idx for idx in class1_idx if idx not in train1]

            train_remain0 = train_take0 - self.args.n_support
            train_remain1 = train_take1 - self.args.n_support
            eval_remain0 = len(eval0) - self.args.n_support
            eval_remain1 = len(eval1) - self.args.n_support

            if min(train_remain0, train_remain1) < 1:
                continue
            if min(eval_remain0, eval_remain1) < 1:
                continue
            if (train_remain0 + train_remain1) < min_total_after_support:
                continue
            if (eval_remain0 + eval_remain1) < min_total_after_support:
                continue

            train_pool_map[int(task_id)] = [train0, train1]
            eval_pool_map[int(task_id)] = [eval0, eval1]

        return train_pool_map, eval_pool_map

    def _sample_task(self, dataset_obj, task_id, inductive=False):
        s_data, q_data = dataset_sampler(dataset_obj, self.args.n_support, self.args.n_query, tgt_id=task_id, inductive=inductive)
        s_batch = Batch.from_data_list(s_data).to(self.device)
        q_batch = Batch.from_data_list(q_data).to(self.device)
        return s_batch, q_batch

    def _sample_from_pool(self, dataset_obj, class0_pool, class1_pool, n_support, n_query=None):
        rng = np.random.default_rng()
        if n_query is None:
            n_query = self.args.n_query

        support0 = self._sample_without_replacement(class0_pool, n_support, rng)
        support1 = self._sample_without_replacement(class1_pool, n_support, rng)
        support_list = support0 + support1

        remain0 = [idx for idx in class0_pool if idx not in support_list]
        remain1 = [idx for idx in class1_pool if idx not in support_list]

        query_list = []
        if len(remain0) > 0:
            query_list += self._sample_without_replacement(remain0, 1, rng)
        if len(remain1) > 0:
            query_list += self._sample_without_replacement(remain1, 1, rng)

        remaining = [idx for idx in (remain0 + remain1) if idx not in query_list]
        query_list += self._sample_without_replacement(remaining, max(0, n_query - len(query_list)), rng)

        s_data = Batch.from_data_list(dataset_obj[support_list]).to(self.device)
        q_data = Batch.from_data_list(dataset_obj[query_list]).to(self.device)
        return s_data, q_data

    def _sample_test_task_from_pool(self, dataset_obj, class0_pool, class1_pool):
        return self._sample_from_pool(dataset_obj, class0_pool, class1_pool, n_support=self.args.n_support, n_query=self.args.n_query)

    def _compute_prototypes(self, pv, labels, num_classes=2):
        labels = labels.view(-1)
        prototypes = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() == 0:
                prototypes.append(pv.new_zeros(pv.size(-1)))
            else:
                prototypes.append(pv[mask].mean(dim=0))
        return torch.stack(prototypes, dim=0)

    def _build_episode(self, dataset_obj, pv_bank, task_id, inductive=False):
        s_data, q_data = self._sample_task(dataset_obj, task_id, inductive=inductive)
        return self._pack_episode(pv_bank, task_id, s_data, q_data)

    def _build_episode_from_pool(self, dataset_obj, pv_bank, task_id, class_pools):
        s_data, q_data = self._sample_from_pool(
            dataset_obj,
            class_pools[0],
            class_pools[1],
            n_support=self.args.n_support,
            n_query=self.args.n_query,
        )
        return self._pack_episode(pv_bank, task_id, s_data, q_data)

    def _pack_episode(self, pv_bank, task_id, s_data, q_data):
        sampled_task = torch.tensor([task_id], device=self.device)
        s_y = s_data.y[:, sampled_task]
        q_y = q_data.y[:, sampled_task]

        _, sd_s_emb = self.maml.module.foundation_model.get_smiles_emb(s_data)
        _, qd_s_emb = self.maml.module.foundation_model.get_smiles_emb(q_data)

        s_pv = self._safe_squeeze_pv(pv_bank[s_data['id']])
        q_pv = self._safe_squeeze_pv(pv_bank[q_data['id']])
        s_prototypes = self._compute_prototypes(s_pv, s_y)

        return {
            'task_id': int(task_id),
            'sampled_task': sampled_task,
            's_data': s_data,
            'q_data': q_data,
            's_y': s_y,
            'q_y': q_y,
            'sd_s_emb': sd_s_emb,
            'qd_s_emb': qd_s_emb,
            's_pv': s_pv,
            'q_pv': q_pv,
            's_prototypes': s_prototypes,
        }

    @staticmethod
    def _support_proto_loss(support_mol, s_proto_feat, s_y):
        loss = support_mol.new_tensor(0.0)
        flat_y = s_y.view(-1)
        for c in range(len(s_proto_feat)):
            mask = flat_y == c
            if mask.sum() == 0:
                continue
            class_mean = support_mol[mask].mean(dim=0)
            loss = loss + F.mse_loss(class_mean, s_proto_feat[c])
        return loss

    @staticmethod
    def _query_proto_loss(query_mol, q_pv_feat):
        loss = query_mol.new_tensor(0.0)
        for q_mol, pv_feat in zip(query_mol, q_pv_feat):
            loss = loss + F.mse_loss(q_mol, pv_feat)
        return loss

    def _forward_episode(self, model, episode, prototypes):

        return model(
            episode['s_data'],
            episode['q_data'],
            episode['sd_s_emb'],
            episode['qd_s_emb'],
            prototypes,
            episode['q_pv'],
            episode['s_y'],
            episode['q_y'],
            episode['sampled_task']
        )

    def _forward_transfer_pair(self, model, source_episode, target_episode):

        return model.forward_transferable(
            source_episode['s_data'],
            source_episode['q_data'],
            source_episode['sd_s_emb'],
            source_episode['qd_s_emb'],
            source_episode['s_prototypes'],
            source_episode['q_pv'],
            source_episode['s_y'],
            source_episode['q_y'],
            source_episode['sampled_task'],
            target_episode['s_data'],
            target_episode['q_data'],
            target_episode['sd_s_emb'],
            target_episode['qd_s_emb'],
            target_episode['s_prototypes'],
            target_episode['q_pv'],
            target_episode['s_y'],
            target_episode['q_y'],
            target_episode['sampled_task']
        )

    def _update_inner_single_domain(self, episode):
        model = self.maml.clone()
        model.train()
        prototypes = episode['s_prototypes']

        for _ in range(self.args.inner_update_step):
            s_logit, _, s_label, _, _, support_mol, _, _, s_proto_feat = self._forward_episode(
                model,
                episode,
                prototypes
            )
            inner_loss = self.cls_criterion(s_logit, s_label)
            proto_loss = self._support_proto_loss(support_mol, s_proto_feat, episode['s_y'])
            model.adapt(inner_loss + proto_loss, allow_nograd=True)

        _, q_logit, _, q_label, graph_f, _, query_mol, q_pv_feat, _ = self._forward_episode(
            model,
            episode,
            prototypes
        )
        eval_loss = self.cls_criterion(q_logit, q_label)
        q_proto_loss = self._query_proto_loss(query_mol, q_pv_feat)
        return eval_loss, graph_f, q_proto_loss

    def _update_inner_transferable(self, source_episode, target_episode):
        model = self.maml.clone()
        model.train()

        for _ in range(self.args.inner_update_step):
            src_out, tgt_out, _ = self._forward_transfer_pair(
                model,
                source_episode,
                target_episode
            )
            src_s_logit, _, src_s_label, _, _, src_support_mol, _, _, src_proto_feat = src_out
            tgt_s_logit, _, tgt_s_label, _, _, tgt_support_mol, _, _, tgt_proto_feat = tgt_out
            src_inner = self.cls_criterion(src_s_logit, src_s_label)
            tgt_inner = self.cls_criterion(tgt_s_logit, tgt_s_label)

            src_inner = src_inner + self._support_proto_loss(src_support_mol, src_proto_feat, source_episode['s_y'])
            tgt_inner = tgt_inner + self._support_proto_loss(tgt_support_mol, tgt_proto_feat, target_episode['s_y'])
            total_inner = self.cross_domain_source_weight * src_inner + self.cross_domain_target_weight * tgt_inner
            model.adapt(total_inner, allow_nograd=True)

        src_out, tgt_out, joint_graph_f = self._forward_transfer_pair(
            model,
            source_episode,
            target_episode
        )
        _, src_q_logit, _, src_q_label, _, _, src_query_mol, src_q_pv_feat, _ = src_out
        _, tgt_q_logit, _, tgt_q_label, tgt_graph_f, _, tgt_query_mol, tgt_q_pv_feat, _ = tgt_out

        src_eval = self.cls_criterion(src_q_logit, src_q_label)
        tgt_eval = self.cls_criterion(tgt_q_logit, tgt_q_label)
        total_eval = self.cross_domain_source_weight * src_eval + self.cross_domain_target_weight * tgt_eval

        q_proto_loss = 0.5 * (
                self._query_proto_loss(src_query_mol, src_q_pv_feat) +
                self._query_proto_loss(tgt_query_mol, tgt_q_pv_feat)
        )

        return total_eval, tgt_graph_f, q_proto_loss

    def train_step(self, epoch):
        eval_losses, graph_f1s, graph_f2s, q_proto_losses = [], [], [], []

        if self.test_dataset is None or len(self.target_train_task_range) == 0:
            replace = len(self.source_train_task_range) < self.args.pool_num
            task_ids = np.random.choice(self.source_train_task_range, self.args.pool_num, replace=replace)
            for task_id in task_ids:
                episode1 = self._build_episode(self.dataset, self.data_pv, int(task_id), inductive=False)
                episode2 = self._build_episode(self.dataset, self.data_pv, int(task_id), inductive=False)
                eval_loss1, graph_f1, q_proto_loss1 = self._update_inner_single_domain(episode1)
                eval_loss2, graph_f2, q_proto_loss2 = self._update_inner_single_domain(episode2)
                eval_losses += [eval_loss1, eval_loss2]
                graph_f1s.append(graph_f1)
                graph_f2s.append(graph_f2)
                q_proto_losses += [q_proto_loss1, q_proto_loss2]
        else:
            pair_num = self.args.pool_num
            replace_target = len(self.target_train_task_range) < pair_num
            target_ids = np.random.choice(self.target_train_task_range, pair_num, replace=replace_target)

            for tgt_task_id in target_ids:
                target_episode1 = self._build_episode_from_pool(
                    self.test_dataset,
                    self.test_data_pv,
                    int(tgt_task_id),
                    self.target_train_pool_map[int(tgt_task_id)],
                )
                source_task_id1 = int(np.random.choice(self.source_train_task_range))
                source_episode1 = self._build_episode(self.dataset, self.data_pv, int(source_task_id1), inductive=False)

                target_episode2 = self._build_episode_from_pool(
                    self.test_dataset,
                    self.test_data_pv,
                    int(tgt_task_id),
                    self.target_train_pool_map[int(tgt_task_id)],
                )
                source_task_id2 = int(np.random.choice(self.source_train_task_range))
                source_episode2 = self._build_episode(self.dataset, self.data_pv, int(source_task_id2), inductive=False)

                eval_loss1, graph_f1, q_proto_loss1 = self._update_inner_transferable(source_episode1, target_episode1)
                eval_loss2, graph_f2, q_proto_loss2 = self._update_inner_transferable(source_episode2, target_episode2)
                eval_losses += [eval_loss1, eval_loss2]
                graph_f1s.append(graph_f1)
                graph_f2s.append(graph_f2)
                q_proto_losses += [q_proto_loss1, q_proto_loss2]

        tgt_f1, tgt_f2 = torch.vstack(graph_f1s), torch.vstack(graph_f2s)
        loss_contr = self.nce_loss(tgt_f1, tgt_f2)
        q_proto_loss = torch.stack(q_proto_losses).mean()
        loss_cls = torch.stack(eval_losses).mean()

        self.opt.zero_grad()
        loss = loss_cls + loss_contr * self.args.contr_w + q_proto_loss * self.args.proto_w
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.maml.parameters(), 1.0)
        self.opt.step()
        return loss_cls.item()

    def test_step(self):

        eval_dataset = self.test_dataset if self.test_dataset is not None else self.dataset
        eval_pv_bank = self.test_data_pv if self.test_dataset is not None else self.data_pv

        task_auc = {}
        for task_i in self.target_eval_task_range:
            task_id = int(task_i)

            if self.test_dataset is not None:
                pool = self.target_eval_pool_map[task_id]
                s_data, q_data = self._sample_test_task_from_pool(eval_dataset, pool[0], pool[1])
            else:
                class0_pool, class1_pool = eval_dataset.index_list[task_id]
                s_data, q_data = self._sample_test_task_from_pool(eval_dataset, list(class0_pool), list(class1_pool))

            sampled_task = torch.tensor([task_id], device=self.device)
            s_y = s_data.y[:, sampled_task]
            q_y = q_data.y[:, sampled_task]
            _, sd_s_emb = self.maml.module.foundation_model.get_smiles_emb(s_data)
            _, qd_s_emb = self.maml.module.foundation_model.get_smiles_emb(q_data)
            s_pv = self._safe_squeeze_pv(eval_pv_bank[s_data['id']])
            q_pv = self._safe_squeeze_pv(eval_pv_bank[q_data['id']])
            prototypes = self._compute_prototypes(s_pv, s_y)

            model = self.maml.clone()
            model.train()

            for _ in range(int(getattr(self.args, 'update_step_test', self.args.inner_update_step))):

                s_logit, _, s_label, _, _, support_mol, _, _, s_proto_feat = model(
                    s_data,
                    q_data,
                    sd_s_emb,
                    qd_s_emb,
                    prototypes,
                    q_pv,
                    s_y,
                    q_y,
                    sampled_task
                )
                inner_loss = self.cls_criterion(s_logit, s_label)
                inner_loss = inner_loss + self._support_proto_loss(support_mol, s_proto_feat, s_y)
                model.adapt(inner_loss, allow_nograd=True)

            model.eval()
            y_true_all, y_pred_all = [], []
            with torch.no_grad():

                _, q_logit, _, q_label, _, _, _, _, _ = model(
                    s_data,
                    q_data,
                    sd_s_emb,
                    qd_s_emb,
                    prototypes,
                    q_pv,
                    s_y,
                    q_y,
                    sampled_task
                )
                y_true_all.append(q_label.cpu().view(-1))
                y_pred_all.append(torch.sigmoid(q_logit).cpu().view(-1))

            y_true = torch.cat(y_true_all, dim=0).numpy()
            y_pred = torch.cat(y_pred_all, dim=0).numpy()

            if len(np.unique(y_true)) < 2:
                task_auc[task_id] = np.nan
                continue

            task_auc[task_id] = roc_auc_score(y_true, y_pred)

        valid_auc = [v for v in task_auc.values() if not np.isnan(v)]
        score = float(np.mean(valid_auc)) if len(valid_auc) > 0 else float('nan')
        return score, task_auc


