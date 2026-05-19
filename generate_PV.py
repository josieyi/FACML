
import re
import torch
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
from transformers import BertTokenizer, WordpieceTokenizer

from args_parser import args_parser
from dataset import FewshotMolDataset
from SPMM.SPMM_models import SPMM
import warnings
warnings.filterwarnings(action='ignore')
RDLogger.DisableLog('rdApp.error')

SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|"
    r"B|C|N|O|S|P|F|I|b|c|n|o|s|p|"
    r"\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|"
    r"%[0-9]{2}|[0-9])"
)

def tokenize_smiles_for_bert(smiles):
    smiles = smiles.strip()

    if smiles.startswith("[CLS]"):
        smiles = smiles[5:].strip()

    tokens = SMILES_TOKEN_PATTERN.findall(smiles)

    if len(tokens) == 0:
        tokens = list(smiles)

    return "[CLS] " + " ".join(tokens)

def p_fun(model, prop_input, text_embeds, text_atts):
    prop_embeds = model.property_encoder(inputs_embeds=prop_input, return_dict=True).last_hidden_state
    prob_atts = torch.ones(prop_input.size()[:-1], dtype=torch.long, device=prop_input.device)
    token_output = model.text_encoder.bert(
        encoder_embeds=prop_embeds,
        attention_mask=prob_atts,
        encoder_hidden_states=text_embeds,
        encoder_attention_mask=text_atts,
        return_dict=True,
        is_decoder=True,
        mode='fusion',
    ).last_hidden_state
    pred = model.property_mtr_head(token_output).squeeze(-1)[:, -1]
    return pred.unsqueeze(1)


@torch.no_grad()
def pv_generate(model, smiles_input, device):
    model.eval()
    mean = model.property_mean.to(device)
    std = model.property_std.to(device)

    data_loader = ['[CLS]' + d for d in smiles_input]
    text = [tokenize_smiles_for_bert(t) for t in data_loader]

    text_input = model.tokenizer(
        text,
        padding='longest',
        truncation=True,
        max_length=100,
        return_tensors="pt",
        add_special_tokens=False
    ).to(device)

    text_embeds = model.text_encoder.bert(
        text_input.input_ids[:, 1:],
        attention_mask=text_input.attention_mask[:, 1:],
        return_dict=True,
        mode='text'
    ).last_hidden_state

    prop_input = model.property_cls.expand(len(text_embeds), -1, -1)
    prediction = []
    for _ in range(53):
        output = p_fun(model, prop_input, text_embeds, text_input.attention_mask[:, 1:])
        prediction.append(output)
        output = model.property_embed(output.unsqueeze(2))
        prop_input = torch.cat([prop_input, output], dim=1)

    prediction = torch.stack(prediction, dim=-1).to(device)
    gather = prediction.squeeze(1) * std + mean
    return gather


def dump_dataset_pv(model, dataset_obj, out_path, device):
    loader = DataLoader(dataset_obj, batch_size=64, shuffle=False, num_workers=0)
    all_pv = [None] * len(dataset_obj)

    for batch in loader:
        batch = batch.to(device)
        pv = pv_generate(model, batch['smiles'], device)
        ids = batch['id'].view(-1).tolist()
        for local_idx, sample_id in enumerate(ids):
            all_pv[int(sample_id)] = pv[local_idx].detach().cpu()

    if any(v is None for v in all_pv):
        raise RuntimeError('Some molecules did not receive PV predictions; dataset ids may be inconsistent.')

    all_pv = torch.stack(all_pv, dim=0)
    torch.save(all_pv, out_path)
    print(f'Saved PV tensor to {out_path} | shape={tuple(all_pv.shape)}')


def main():
    args = args_parser()
    foundation_config = {
        'embed_dim': args.emb_dim,
        'bert_config_text': './SPMM/config_bert.json',
        'bert_config_property': './SPMM/config_bert_property.json',
    }

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
        vocab=tokenizer.vocab,
        unk_token=tokenizer.unk_token,
        max_input_chars_per_word=250,
    )
    foundation_model = SPMM(config=foundation_config, tokenizer=tokenizer, no_train=True)

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            if 'queue' in key:
                del state_dict[key]
        msg = foundation_model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    foundation_model = foundation_model.to(args.device)

    source_dataset = FewshotMolDataset(root=args.data_root, name=args.dataset)
    dump_dataset_pv(foundation_model, source_dataset, args.data_root + args.dataset + '/data_pv.pt', args.device)

    if args.test_dataset is not None:
        target_dataset = FewshotMolDataset(root=args.data_root, name=args.test_dataset)
        dump_dataset_pv(foundation_model, target_dataset, args.data_root + args.test_dataset + '/data_pv.pt', args.device)


if __name__ == '__main__':
    main()
