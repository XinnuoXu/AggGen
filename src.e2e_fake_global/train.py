#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
from others.logging import init_logger
from train_hmm import validate_abs, train_abs, test_abs 
from pretrain import vld_pretrain, pretrain, test_pretrain 

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test', 'pretrain', 'vld_pretrain', 'test_pretrain'])
    parser.add_argument("-test_data", default='test', type=str, choices=['test'])
    parser.add_argument("-data_path", default='../bert_data_new/cnndm')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-src_dict_path", default='../dict/')
    parser.add_argument("-tgt_dict_path", default='../dict/')
    parser.add_argument("-merge_dict_path", default='../../data/')
    parser.add_argument("-relation_path", default='../dict/')
    parser.add_argument("-pretrain_path", default='')
    parser.add_argument("-train_from", default='')
    parser.add_argument("-temp_dir", default='../temp')
    parser.add_argument("-pad_id", default=0, type=int)

    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-state_emb_size", default=128, type=int)
    parser.add_argument("-state_dropout", default=0.2, type=float)
    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    
    parser.add_argument("-dec_hidden_size", default=256, type=int)
    parser.add_argument("-dec_ff_size", default=512, type=int)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=2, type=int)
    parser.add_argument("-dec_heads", default=4, type=int)

    parser.add_argument("-enc_hidden_size", default=256, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=2, type=int)
    parser.add_argument("-enc_heads", default=4, type=int)
    parser.add_argument("-autogressive", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_pos", default=512, type=int)

    parser.add_argument("-s_beam_size", default=5, type=int)
    parser.add_argument("-active_patt_num", default=2, type=int)
    parser.add_argument("-agg_topk", default=3, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-lr_tok", default=1, type=float)
    parser.add_argument("-lr_hmm", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_tok", default=8000, type=int)
    parser.add_argument("-warmup_steps_hmm", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)

    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-log_attn_file', default='../logs/attn.log')
    parser.add_argument('-seed', default=777, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-tp_bias", default=0.1, type=float)
    parser.add_argument("-extreme", default= 0.9, type=float)
    parser.add_argument("-inference_mode", default='beam', type=str, choices=['beam', 'nucleus', 'topk'])
    parser.add_argument("-nucleus_p", default=0.95, type=float)
    parser.add_argument("-top_k", default=50, type=int)
    parser.add_argument("-lm_cov_bias", default=0.0, type=float)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if (args.mode == 'train'):
        train_abs(args, device_id)
    elif (args.mode == 'validate'):
        validate_abs(args, device_id)
    elif (args.mode == 'test'):
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
        test_abs(args, device_id, cp, step)
    elif (args.mode == 'pretrain'):
        pretrain(args, device_id)
    elif (args.mode == 'vld_pretrain'):
        vld_pretrain(args, device_id)
    if (args.mode == 'test_pretrain'):
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
        test_pretrain(args, device_id, cp, step)
