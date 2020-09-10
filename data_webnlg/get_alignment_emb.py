#coding=utf8
import sys
import torch
import torch.nn as nn
from pytorch_transformers import *

model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-uncased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights, cache_dir="./temp_dir").to('cuda')
CLS_ID = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
soft_max_row = nn.Softmax(dim=1)
soft_max_col = nn.Softmax(dim=0)

def phrase_emb(phrase):
    tokens = tokenizer.tokenize(phrase)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.unsqueeze(torch.tensor(ids), 0).to('cuda')
    last_hidden_states = model(ids, attention_mask=~(ids == 0))[0]

    ids = torch.squeeze(ids)
    last_hidden_states = torch.squeeze(last_hidden_states)
    cls_idxes = (ids == CLS_ID).nonzero().view(-1)
    ph_embs = torch.cat([torch.unsqueeze(last_hidden_states[idx], 0) for idx in cls_idxes], 0)

    return ph_embs

def one_pair(tripples, keys, terminals):
    new_tripples = []
    for i in range(len(tripples)):
        tri = tripples[i]
        k = keys[i]
        idx = tri.find(k)
        if idx == -1:
            idx = 0
        new_tripples.append(tri[idx:])

    tri_embs = phrase_emb('[CLS] '+' [CLS] '.join(new_tripples))
    tml_embs = phrase_emb('[CLS] '+' [CLS] '.join(terminals))
    scores = torch.mm(tri_embs, torch.transpose(tml_embs, 0, 1))
    tri2tml = soft_max_row(scores)
    tml2tri = soft_max_col(scores)

if __name__ == '__main__':
    input_dir = "./data-rst/"
    src_path = input_dir+"webnlg_"+sys.argv[1]+'_src.jsonl'
    tgt_path = input_dir+"webnlg_"+sys.argv[1]+'_tgt.jsonl'
    src_list = [line.strip() for line in open(src_path)]
    tgt_list = [line.strip() for line in open(tgt_path)]

    output_dir = "./data-alg/"
    src_path = output_dir+"webnlg_"+sys.argv[1]+'_src.jsonl'
    tgt_path = output_dir+"webnlg_"+sys.argv[1]+'_tgt.jsonl'
    fpout_src = open(src_path, 'w')
    fpout_tgt = open(tgt_path, 'w')

    for i in range(len(src_list)):
        src = src_list[i]; tgt = tgt_list[i]

        # src data
        src_data = src.split('\t')
        tripples = src_data[:-1]
        keys = src_data[-1].split('|')
        # tgt data
        tgt_data = tgt.split('\t')
        non_terminals = tgt_data[0]
        terminals = tgt_data[1:]
        # processing
        one_pair(tripples, keys, terminals)

        #fpout_src.write(doc + '\t' + relations + '\n')
        #fpout_tgt.write(rst + '\t' + '\t'.join(toks) + '\n')
    fpout_src.close()
    fpout_tgt.close()
