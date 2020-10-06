#coding=utf8

import sys, os
import json
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

def prepare_inference_data():
    s_out_path = "annotation_human/webnlg_test_src.jsonl"
    t_out_path = "annotation_human/webnlg_test_tgt.jsonl"
    fpout_src = open(s_out_path, 'w')
    fpout_tgt = open(t_out_path, 'w')

    s_path = "../Human_eva/scripts/webnlg_ann_src.jsonl"
    t_path = "../Human_eva/scripts/webnlg_ann_tgt.jsonl"

    src_lines = [line.strip() for line in open(s_path)]
    for i, line in enumerate(open(t_path)):
        flist = line.strip().split('\t')
        tgts = []
        for tgt in flist:
            tlist = tgt.split('[XXN]')
            tgt_str = ' '.join(tlist[2].split(' [SEP] '))
            tgts.append(tgt_str)
        tgt_line = ' <REF_SEP> '.join(tgts)
        fpout_src.write(src_lines[i] + '\n')
        fpout_tgt.write(tgt_line + '\n')
    fpout_src.close()
    fpout_tgt.close()


def post_process():
    sys.path.append('../data_webnlg/')
    from get_preprocess import data_processing

    ex_id_path = "../logs/abs_bert_cnndm.17000.example_id"
    example_ids = [int(line.strip()) for line in open(ex_id_path)]

    cand_path = "../logs/abs_bert_cnndm.17000.candidate"
    tgt_dict = {}
    for i, line in enumerate(open(cand_path)):
        line = line.strip().replace(' .', ' . <s> ').strip()
        line = data_processing(line)
        tgt_dict[example_ids[i]] = line

    cand_fpout = open("./annotation_human/pretrain_cands.txt", "w")
    for item in sorted(tgt_dict.items(), key = lambda d:d[0]):
        cand_fpout.write(item[1] + "\n")
    cand_fpout.close()


def get_srl():
    archive = load_archive("../data_webnlg/srl-model-2018.05.25.tar.gz", cuda_device=0)
    srl = Predictor.from_archive(archive)
    print ("Loading Done")

    lines = []
    for line in open("./annotation_human/pretrain_cands.txt"):
        line = ' '.join(line.strip().split(' ')[:-1])
        flist = line.split(' <s> ')
        lines.append(flist)

    fpout = open("./annotation_human/pretrain_cands.srl", "w")
    for sentences in lines:
        sentences = [{"sentence": line} for line in sentences]
        srl_res = srl.predict_batch_json(sentences)
        fpout.write(json.dumps(srl_res) + '\n')
    fpout.close()


def get_tree():
    sys.path.append('../data_webnlg/')
    from get_tree import one_summary

    input_tgt = []
    for line in open("./annotation_human/pretrain_cands.srl"):
        line = line.strip()
        input_tgt.append(json.loads(line))

    fpout = open("./annotation_human/pretrain_cands.tree", "w")
    for tgts in input_tgt:
        tgt_tree = [one_summary(o) for o in tgts]
        out_json = {"summary":tgt_tree}
        fpout.write(json.dumps(out_json) + "\n")
    fpout.close()


def get_rst():
    sys.path.append('../data_webnlg/')
    from get_rst import process_tgt

    fpout_tgt = open("./annotation_human/pretrain_cands.rst", "w")
    for line in open("./annotation_human/pretrain_cands.tree"):
        line = line.strip()
        summary = json.loads(line)["summary"]
        rst, toks = process_tgt(summary)
        fpout_tgt.write(rst + '\t' + '\t'.join(toks) + '\n')
    fpout_tgt.close()


def get_algn():
    from annotators import get_alignment
    input_dir = "./annotation_human/"
    src_in_path = input_dir+"/webnlg_test_src.jsonl"
    tgt_in_path = input_dir+"pretrain_cands.rst"
    output_dir = "./annotation_human/"
    src_out_path = output_dir+"pretrain_cands_src.jsonl"
    tgt_out_path = output_dir+"pretrain_cands_tgt.jsonl"
    get_alignment(src_in_path, tgt_in_path, src_out_path, tgt_out_path)

def evaluate():
    from evaluate_planning import NMI
    from evaluate_planning import Kendalltau

    CAND_HMM_PATH = './annotation_human/pretrain_cands_tgt.jsonl'
    print ('grouping res (NMI):')
    nmi_obj = NMI(CAND_HMM_PATH)
    cands = []
    for line in open(CAND_HMM_PATH):
        sequence = line.split('\t')[1].split('|')
        clustering = {}
        for i, slot in enumerate(sequence):
            types = slot.split('&')
            for t in types:
                clustering[t] = i
        cands.append(clustering)
    golds = nmi_obj.load_gold()
    srcs = nmi_obj.load_src()
    cand_score, ref_score = nmi_obj.get_nmi(cands, golds, srcs)
    print ('cand_score:', cand_score)
    print ('ref_score:', ref_score)

    print ('sorting res (K-Tau):')
    k_obj = Kendalltau(CAND_HMM_PATH)
    golds = k_obj.load_gold()
    srcs = k_obj.load_src()
    cand_score, ref_score, max_ref_score = k_obj.get_kendall(cands, golds, srcs)
    print ('cand_score:', cand_score)
    print ('ref_score:', ref_score)
    print ('max_ref_score:', max_ref_score)

if __name__ == '__main__':
    if sys.argv[1] == "prepare":
        prepare_inference_data()
    elif sys.argv[1] == "segmentation":
        post_process()
        get_srl()
        get_tree()
        get_rst()
    elif sys.argv[1] == "alignment":
        get_algn()
    elif sys.argv[1] == "evaluate":
        evaluate()
