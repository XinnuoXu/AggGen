import copy
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_

from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from models.optimizers import Optimizer

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))
    return optim


def build_optim_tok(args, model, checkpoint):
    """ Build optimizer """
    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_tok, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_tok)

    params = [(n, p) for n, p in list(model.named_parameters()) \
                                    if n.startswith('encoder') \
                                    or n.startswith('decoder') \
                                    or n.startswith('generator')]
    optim.set_parameters(params)
    return optim


def build_optim_hmm(args, model, checkpoint):
    """ Build optimizer """
    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_hmm, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_hmm)

    params = [(n, p) for n, p in list(model.named_parameters()) \
                                    if not (n.startswith('encoder')\
                                    or n.startswith('decoder')\
                                    or n.startswith('generator'))]
    optim.set_parameters(params)
    return optim


def get_generator(vocab_size, dec_hidden_size, device, logsf=True):
    if logsf:
        gen_func = nn.LogSoftmax(dim=-1)
    else:
        gen_func = nn.Softmax(dim=-1)
    generator = nn.Sequential(nn.Linear(dec_hidden_size, vocab_size),gen_func)
    generator.to(device)
    return generator


class HMMModel(nn.Module):
    def __init__(self, args, device, checkpoint=None, pretrain_model=None):
        super(HMMModel, self).__init__()
        self.args = args
        self.device = device

        with open(args.src_dict_path) as f:
            line = f.read().strip()
        self.src_dict = json.loads(line)

        with open(args.tgt_dict_path) as f:
            line = f.read().strip()
        self.tgt_dict = json.loads(line)

        with open(args.relation_path) as f:
            line = f.read().strip()
        self.relation_dict = json.loads(line)
        self.relation_size = len(self.relation_dict)

        # Relation embeddings
        self.tsf_softmax = nn.LogSoftmax(dim=1)
        self.ini_softmax = nn.LogSoftmax(dim=0)
        self.init_linear = nn.Linear(args.state_emb_size, 1)
        self.S_from = nn.Parameter(torch.Tensor(self.relation_size, args.state_emb_size))
        self.S_to = nn.Parameter(torch.Tensor(args.state_emb_size, self.relation_size))
        self.S_init = nn.Parameter(torch.Tensor(self.relation_size, args.state_emb_size))
        self.state_drop = nn.Dropout(args.state_dropout)

        # Combined relation transition
        self.S_ext_from = nn.Parameter(torch.Tensor(self.relation_size, args.state_emb_size))
        self.S_ext_to = nn.Parameter(torch.Tensor(args.state_emb_size, self.relation_size))

        # embeddings
        self.enc_vocab_size = len(self.src_dict)
        self.dec_vocab_size = len(self.tgt_dict)
        enc_embeddings = nn.Embedding(self.enc_vocab_size, args.enc_hidden_size, padding_idx=0)
        tgt_embeddings = nn.Embedding(self.dec_vocab_size, args.dec_hidden_size, padding_idx=0)

        # Encoder
        self.encoder = TransformerEncoder(args.enc_hidden_size, 
                                            args.enc_ff_size, 
                                            args.enc_heads, 
                                            args.enc_dropout, 
                                            args.enc_layers, 
                                            embeddings=enc_embeddings,
                                            max_pos=args.max_pos)

        # Deocder
        self.decoder = TransformerDecoder(args.dec_layers,
                                            args.dec_hidden_size, 
                                            heads=args.dec_heads,
                                            d_ff=args.dec_ff_size, 
                                            dropout=args.dec_dropout, 
                                            embeddings=tgt_embeddings,
                                            max_pos=args.max_pos)

        # Generator
        self.generator = get_generator(self.dec_vocab_size, args.dec_hidden_size, device, logsf=False)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        elif pretrain_model is not None:
            # tmperory code
            encoder_parameters = []
            for n, p in pretrain_model.items():
                if n.startswith('encoder'):
                    key = n[8:]
                    value = p
                    if key.startswith('pos_emb'):
                        value = value[:, :args.max_pos, :]
                    encoder_parameters.append((key, value))
            self.encoder.load_state_dict(dict(encoder_parameters))

            # tmperory code
            decoder_parameters = []
            for n, p in pretrain_model.items():
                if n.startswith('decoder'):
                    key = n[8:]
                    value = p
                    if key.startswith('pos_emb'):
                        value = value[:, :args.max_pos, :]
                    decoder_parameters.append((key, value))
            self.decoder.load_state_dict(dict(decoder_parameters))
            self.generator.load_state_dict(dict([(n[10:], p) for n, p in pretrain_model.items() if n.startswith('generator')]), strict=True)

            other_params = [self.S_from, self.S_to, self.S_init, self.S_ext_from, self.S_ext_to]
            for param in other_params:
                param.data.uniform_(-0.1, 0.1)
            other_linears = [self.init_linear]
            for param in other_linears:
                param.weight.data.uniform_(-0.1, 0.1)
        else:
            for module in self.encoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

            for module in self.comb_trans.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()

            other_params = [self.S_from, self.S_to, self.S_init, self.S_ext_from, self.S_ext_to]
            for param in other_params:
                param.data.uniform_(-0.1, 0.1)
            other_linears = [self.init_linear]
            for param in other_linears:
                param.weight.data.uniform_(-0.1, 0.1)

            if self.args.share_emb:
                assert self.enc_vocab_size == self.dec_vocab_size, \
                        "vocab for src and tgt should be the same"
                enc_embeddings.weight = tgt_embeddings.weight 

        self.to(device)

    def trans_logprobs(self):
        # Return transition matrix
        S_from = self.state_drop(self.S_from)
        S_to = self.S_to
        tscores = torch.mm(S_from, S_to)
        trans_logps = self.tsf_softmax(tscores)
        init_logps = self.ini_softmax(self.init_linear(self.S_init).squeeze())
        return init_logps, trans_logps

    def external_logprobs(self):
        # Return external transition matrix
        self.S_ext_from, self.S_ext_to
        S_ext_from = self.state_drop(self.S_ext_from)
        S_ext_to = self.S_ext_to
        tscores = torch.mm(S_ext_from, S_ext_to)
        ext_logps = self.tsf_softmax(tscores)
        return ext_logps

    def expand_src(self, src, top_vec, ex_idx):
        bsz, src_len = src.size()
        bsz, src_len, dim = top_vec.size()
        new_src = []
        new_emb = []
        for i in range(bsz):
            b_idx = ex_idx[i][0][0]
            e_idx = ex_idx[i][-1][1]
            ex_len = e_idx-b_idx
            new_src.append(src[i].unsqueeze(0).expand(ex_len, src_len))
            new_emb.append(top_vec[i].unsqueeze(0).expand(ex_len, src_len, dim))
        src = torch.cat(new_src)
        top_vec = torch.cat(new_emb)
        return src, top_vec

    def forward(self, src, tgt, mask_src, pmt_msk, ex_idx):
        init_logps, trans_logps = self.trans_logprobs()
        top_vec = self.encoder(src, mask_src)
        src, top_vec = self.expand_src(src, top_vec, ex_idx)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state, memory_masks=pmt_msk)
        return decoder_outputs, None


class PretrainModel(nn.Module):
    def __init__(self, args, device, checkpoint=None):
        super(PretrainModel, self).__init__()
        self.args = args
        self.device = device

        with open(args.src_dict_path) as f:
            line = f.read().strip()
        self.src_dict = json.loads(line)

        with open(args.tgt_dict_path) as f:
            line = f.read().strip()
        self.tgt_dict = json.loads(line)

        self.enc_vocab_size = len(self.src_dict)
        self.dec_vocab_size = len(self.tgt_dict)
        enc_embeddings = nn.Embedding(self.enc_vocab_size, args.enc_hidden_size, padding_idx=0)
        tgt_embeddings = nn.Embedding(self.dec_vocab_size, args.dec_hidden_size, padding_idx=0)

        # Encoder
        self.encoder = TransformerEncoder(args.enc_hidden_size, 
                                            args.enc_ff_size, 
                                            args.enc_heads, 
                                            args.enc_dropout, 
                                            args.enc_layers, 
                                            embeddings=enc_embeddings,
                                            max_pos=args.max_pos)
        # Deocder
        self.decoder = TransformerDecoder(args.dec_layers,
                                            args.dec_hidden_size, 
                                            heads=args.dec_heads,
                                            d_ff=args.dec_ff_size, 
                                            dropout=args.dec_dropout, 
                                            embeddings=tgt_embeddings,
                                            max_pos=args.max_pos)
        # Generator
        self.generator = get_generator(self.dec_vocab_size, args.dec_hidden_size, device, logsf=True)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.encoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if self.args.share_emb:
                assert self.enc_vocab_size == self.dec_vocab_size, \
                        "vocab for src and tgt should be the same"
                enc_embeddings.weight = tgt_embeddings.weight 

        self.to(device)


    def forward(self, src, tgt, segs, mask_src, mask_tgt):
        top_vec = self.encoder(src, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
