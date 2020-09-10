#coding=utf8
import sys
import os

black_list = ['[PAD]', '[CLS]']

def sort_src(src):
    sset = set()
    for s in src:
        sset.add(s)
    return ' [SEP] '.join(sset)

def one_model(model_id, req):
    # SRC
    srcs = []; tripple_nums = []
    for line in open('../logs/abs_bert_cnndm.' + model_id + '.raw_src'):
        tokens = line.strip().split()
        tokens = [tok for tok in tokens if tok not in black_list][:-1]
        src = ' '.join(tokens)
        slist = src.split(' [SEP] ')
        src = sort_src(slist)
        srcs.append(src)
        tripple_nums.append(len(slist))

    # GOLD
    gold_dict = {}
    for i, line in enumerate(open('../logs/abs_bert_cnndm.' + model_id + '.gold')):
        if tripple_nums[i] != req:
            continue
        gold_dict[srcs[i]] = line.strip().replace('  ', ' ')

    # CAND
    cand_dict = {}
    for i, line in enumerate(open('../logs/abs_bert_cnndm.' + model_id + '.candidate')):
        if tripple_nums[i] != req:
            continue
        cand_dict[srcs[i]] = line.strip()

    # HMM Pattern
    hmm_dict = {}
    if os.path.exists('../logs/abs_bert_cnndm.' + model_id + '.hmm'):
        for i, line in enumerate(open('../logs/abs_bert_cnndm.' + model_id + '.hmm')):
            if tripple_nums[i] != req:
                continue
            hmm_dict[srcs[i]] = line.strip()

    # Switch key
    src_map = {}; cand_map = {}; hmm_map = {}
    for src in gold_dict:
        gold = gold_dict[src]
        cand = cand_dict[src]
        if src in hmm_dict:
            hmm = hmm_dict[src]
            hmm_map[gold] = hmm
        cand_map[gold] = cand
        src_map[gold] = src

    return src_map, cand_map, hmm_map

def pair(src_1, src_2, cand_1, cand_2, hmm_1, hmm_2):
    pair_dict = {}
    for key in cand_1:
        if key in hmm_1:
            hmm_1_str = hmm_1[key]
        else:
            hmm_1_str = ""
        if key in hmm_2:
            hmm_2_str = hmm_2[key]
        else:
            hmm_2_str = ""
        pair_dict[key] = (src_1[key], src_2[key], cand_1[key], cand_2[key], hmm_1_str, hmm_2_str)
    return pair_dict

if __name__ == '__main__':
    src_1, cand_1, hmm_1 = one_model(sys.argv[1], int(sys.argv[3]))
    src_2, cand_2, hmm_2 = one_model(sys.argv[2], int(sys.argv[3]))
    pair_dict = pair(src_1, src_2, cand_1, cand_2, hmm_1, hmm_2)

    fpout = open('compare_'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3], 'w')
    for key in pair_dict:
        fpout.write(">>I"+sys.argv[1]+"<<\t"+pair_dict[key][0]+'\n')
        fpout.write(">>I"+sys.argv[2]+"<<\t"+pair_dict[key][1]+'\n')
        fpout.write("---------------------------\n")
        fpout.write(">>M"+sys.argv[1]+"<<\t"+pair_dict[key][2] +'\n')
        fpout.write(">>M"+sys.argv[2]+"<<\t"+pair_dict[key][3] +'\n')
        if pair_dict[key][4] != '':
            fpout.write(">>H"+sys.argv[1]+"<<\t"+pair_dict[key][4] +'\n')
        if pair_dict[key][5] != '':
            fpout.write(">>H"+sys.argv[2]+"<<\t"+pair_dict[key][5] +'\n')
        fpout.write("---------------------------\n")
        fpout.write(">>GOLD<<\t"+key +'\n\n\n')
    fpout.close()
