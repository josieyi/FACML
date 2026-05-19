
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, WordpieceTokenizer
from SPMM.SPMM_models import SPMM
from .base_encoder import GNN_Encoder_Frozen, GNN_Encoder_with_Adapter
from .relation import Context_Encoder


class FACML(nn.Module):
    def __init__(self, args, task_num, train_task_num):
        super().__init__()
        self.args = args

        foundation_config = {
            'embed_dim': args.emb_dim,
            'bert_config_text': './SPMM/config_bert.json',
            'bert_config_property': './SPMM/config_bert_property.json',
        }
        self.tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
        self.tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.tokenizer.vocab,
            unk_token=self.tokenizer.unk_token,
            max_input_chars_per_word=250,
        )
        self.foundation_model = SPMM(config=foundation_config, tokenizer=self.tokenizer, no_train=True)

        if args.checkpoint:
            print('LOADING PRETRAINED MODEL..')
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            state_dict = checkpoint['state_dict']
            for key in list(state_dict.keys()):
                if 'queue' in key:
                    del state_dict[key]
            msg = self.foundation_model.load_state_dict(state_dict, strict=False)
            print('load checkpoint from %s' % args.checkpoint)
            print(msg)
        self.foundation_model = self.foundation_model.to(args.device)

        self.mol_encoder_frozen = GNN_Encoder_Frozen(
            num_layer=args.mol_num_layer,
            emb_dim=args.GNN_emb_dim,
            JK=args.JK,
            drop_ratio=args.mol_dropout,
            graph_pooling=args.mol_graph_pooling,
            gnn_type=args.mol_gnn_type,
            batch_norm=args.mol_batch_norm,
            load_path=args.mol_pretrain_load_path,
        )

        self.relation_net = Context_Encoder(
            args=args,
            in_dim=args.emb_dim,
            num_layer=args.rel_layer,
            edge_n_layer=args.rel_edge_n_layer,
            edge_hidden_dim=args.rel_edge_hidden_dim,
            total_tasks=task_num,
            train_tasks=train_task_num,
            device=args.device,
            batch_norm=args.rel_batch_norm,
            top_k=args.rel_top_k,
            dropout=args.rel_dropout,
            pre_dropout=args.rel_pre_dropout,
            nan_w=args.rel_nan_w,
            nan_type=args.rel_nan_type,
            edge_type=args.rel_edge_type,
        )

        self.mol_encoder = GNN_Encoder_with_Adapter(
            num_layer=args.mol_num_layer,
            emb_dim=args.GNN_emb_dim,
            JK=args.JK,
            drop_ratio=args.mol_dropout,
            graph_pooling=args.mol_graph_pooling,
            gnn_type=args.mol_gnn_type,
            batch_norm=args.mol_batch_norm,
            load_path=args.mol_pretrain_load_path,
            adapter_hidden_dim=args.adapter_hidden_dim,
            layer_norm=args.mol_layer_norm,
        )

        rep_dim = args.emb_dim * 2
        context_dim = 2 * args.emb_dim
        contextual_dim = args.emb_dim
        classifier_in_dim = rep_dim + context_dim + contextual_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, args.emb_dim),
            nn.BatchNorm1d(args.emb_dim),
            nn.ReLU(),
            nn.Linear(args.emb_dim, 1),
        )
        self.smiles_proj = nn.Linear(768, args.emb_dim)
        self.pv_proj = nn.Linear(53, args.emb_dim)

        self.property_mlp = nn.Sequential(
            nn.Linear(rep_dim * 2, rep_dim),
            nn.ReLU(),
            nn.Linear(rep_dim, rep_dim),
        )

    def encode_mol(self, data):
        return self.mol_encoder_frozen(data.x, data.edge_index, data.edge_attr, data.batch)

    def _build_episode_rep(self, data, smiles_emb):
        smiles_emb = self.smiles_proj(smiles_emb)
        graph_emb = self.encode_mol(data)
        rep = F.normalize(torch.cat([smiles_emb, graph_emb], dim=-1), p=2, dim=-1)
        return rep

    @staticmethod
    def _compute_support_prototypes(rep, labels, num_classes=2):
        labels = labels.view(-1)
        prototypes = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() == 0:
                prototypes.append(rep.new_zeros(rep.size(-1)))
            else:
                prototypes.append(rep[mask].mean(dim=0))
        return torch.stack(prototypes, dim=0)

    def _property_aware_encode(self, rep, support_proto):

        proto_context = support_proto.unsqueeze(0).expand(rep.size(0), -1, -1)
        tokens = torch.cat([rep.unsqueeze(1), proto_context], dim=1)  # [N, 3, d]
        attn_score = torch.matmul(tokens, tokens.transpose(1, 2)) / math.sqrt(tokens.size(-1))
        attn = torch.softmax(attn_score, dim=-1)
        contextual = torch.bmm(attn, tokens)[:, 0, :]
        return self.property_mlp(torch.cat([rep, contextual], dim=-1))


    def _build_task_inputs(self, rep, support_context, query_context, graph_f, task_context, s_data, q_data, s_y, s_proto_feat, q_pv_feat):
        contextualized_s_feat = self.mol_encoder(s_data, graph_f)
        contextualized_q_feat = self.mol_encoder(q_data, graph_f)
        s_input = torch.cat([rep['s_rep'], support_context, contextualized_s_feat], dim=-1)
        q_input = torch.cat([rep['q_rep'], query_context, contextualized_q_feat], dim=-1)

        return s_input, q_input

    def _build_reps(self, s_data, q_data, sd_s_emb, qd_s_emb, s_y):

        s_rep = self._build_episode_rep(s_data, sd_s_emb)
        q_rep = self._build_episode_rep(q_data, qd_s_emb)
        support_rep_proto = self._compute_support_prototypes(s_rep, s_y)

        s_rep = self._property_aware_encode(s_rep, support_rep_proto)
        q_rep = self._property_aware_encode(q_rep, support_rep_proto)

        sample_feat = torch.cat([s_rep, q_rep], dim=0)

        return {
            's_rep': s_rep,
            'q_rep': q_rep,
            'sample_feat': sample_feat,
        }

    def forward(self, s_data, q_data, sd_s_emb, qd_s_emb, s_prototypes, q_pv, s_y, q_y, sampled_task):

        rep = self._build_reps(s_data, q_data, sd_s_emb, qd_s_emb, s_y)

        s_proto_emb = self.pv_proj(s_prototypes)
        q_pv_emb = self.pv_proj(q_pv)

        (
            s_label,
            q_label,
            graph_f,
            task_context,
            support_mol_context,
            query_mol_context,
            support_context,
            query_context,
            q_pv_feat,
            s_proto_feat,
        ) = self.relation_net.forward_inductive(rep['sample_feat'], sampled_task, s_proto_emb, q_pv_emb, s_y, q_y)

        s_input, q_input = self._build_task_inputs(
            rep, support_context, query_context, graph_f, task_context, s_data, q_data, s_y, s_proto_feat, q_pv_feat
        )

        s_label = s_label.float().view(-1, 1)
        q_label = q_label.float().view(-1, 1)

        s_logit = self.classifier(s_input)
        q_logit = self.classifier(q_input)

        return (
            s_logit,
            q_logit,
            s_label,
            q_label,
            graph_f,
            support_mol_context,
            query_mol_context,
            q_pv_feat,
            s_proto_feat,
        )

    def forward_transferable(
        self,
        src_s_data,
        src_q_data,
        src_sd_s_emb,
        src_qd_s_emb,
        src_s_prototypes,
        src_q_pv,
        src_s_y,
        src_q_y,
        src_sampled_task,
        tgt_s_data,
        tgt_q_data,
        tgt_sd_s_emb,
        tgt_qd_s_emb,
        tgt_s_prototypes,
        tgt_q_pv,
        tgt_s_y,
        tgt_q_y,
        tgt_sampled_task,
    ):

        src_rep = self._build_reps(src_s_data, src_q_data, src_sd_s_emb, src_qd_s_emb, src_s_y)
        tgt_rep = self._build_reps(tgt_s_data, tgt_q_data, tgt_sd_s_emb, tgt_qd_s_emb, tgt_s_y)

        src_s_proto_emb = self.pv_proj(src_s_prototypes)
        src_q_pv_emb = self.pv_proj(src_q_pv)
        tgt_s_proto_emb = self.pv_proj(tgt_s_prototypes)
        tgt_q_pv_emb = self.pv_proj(tgt_q_pv)

        src_out, tgt_out, joint_graph_f = self.relation_net.forward_transferable(
            src_rep['sample_feat'],
            tgt_rep['sample_feat'],
            src_s_proto_emb,
            src_q_pv_emb,
            src_s_y,
            src_q_y,
            tgt_s_proto_emb,
            tgt_q_pv_emb,
            tgt_s_y,
            tgt_q_y,
        )

        src_s_input, src_q_input = self._build_task_inputs(
            src_rep,
            src_out['support_context'],
            src_out['query_context'],
            src_out['graph_emb'],
            src_out['task_context'],
            src_s_data,
            src_q_data,
            src_s_y,
            src_out['s_proto_feat'],
            src_out['q_pv_feat'],
        )
        tgt_s_input, tgt_q_input = self._build_task_inputs(
            tgt_rep,
            tgt_out['support_context'],
            tgt_out['query_context'],
            tgt_out['graph_emb'],
            tgt_out['task_context'],
            tgt_s_data,
            tgt_q_data,
            tgt_s_y,
            tgt_out['s_proto_feat'],
            tgt_out['q_pv_feat'],
        )

        src_s_label = src_out['s_label'].float().view(-1, 1)
        src_q_label = src_out['q_label'].float().view(-1, 1)
        tgt_s_label = tgt_out['s_label'].float().view(-1, 1)
        tgt_q_label = tgt_out['q_label'].float().view(-1, 1)

        src_s_logit = self.classifier(src_s_input)
        src_q_logit = self.classifier(src_q_input)
        tgt_s_logit = self.classifier(tgt_s_input)
        tgt_q_logit = self.classifier(tgt_q_input)

        return (
            (src_s_logit, src_q_logit, src_s_label, src_q_label, src_out['graph_emb'], src_out['support_mol_context'],
             src_out['query_mol_context'], src_out['q_pv_feat'], src_out['s_proto_feat']),
            (tgt_s_logit, tgt_q_logit, tgt_s_label, tgt_q_label, tgt_out['graph_emb'], tgt_out['support_mol_context'],
             tgt_out['query_mol_context'], tgt_out['q_pv_feat'], tgt_out['s_proto_feat']),
            joint_graph_f,
        )

