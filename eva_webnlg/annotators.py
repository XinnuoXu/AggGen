#coding=utf8
import copy
import sys
import collections
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
THRESHOLD=0.5

def clean_tokens(tokens):
    tokens = [tok.lower() for tok in tokens]
    tokens = [tok.replace('.', '') for tok in tokens]
    return tokens

def match_answer(tripple, phrase):
    tokens = clean_tokens(phrase.split())
    tri_tokens = clean_tokens(tripple.split())
    matched_toks = 0
    for tok in tri_tokens:
        if tok in tokens:
            matched_toks += 1
    #if float(matched_toks)/len(tri_tokens) >= THRESHOLD:
    return matched_toks / len(tri_tokens)

def one_triple_to_phrase(trip, phrases):
    alignment = []
    for i in range(len(phrases)):
        percentage = match_answer(trip, phrases[i])
        alignment.append(percentage)
    max_value = max(alignment)
    max_index = alignment.index(max_value)
    return max_index

def one_phrase_relation(relations, keys, phrase):
    phrase = [wordnet_lemmatizer.lemmatize(item, pos="n") for item in phrase.lower().split()]
    phrase = [wordnet_lemmatizer.lemmatize(item, pos="v") for item in phrase]
    match_res = []
    for rel in relations:
        match_num = 0
        for r in rel:
            if r in phrase:
                match_num += 1
        match_res.append(match_num)
    max_value = max(match_res)
    if max_value == 0:
        return ""
    max_index = match_res.index(max_value)
    for i in range(len(match_res)):
        if match_res[i] == max_value and i != max_index:
            return ""
    return keys[max_index]


def clean_key(k):
    if k.find('_') == -1:
        split_upper = ['']
        for i in range(len(k)):
            if k[i].isupper():
                split_upper.append(k[i].lower())
            else:
                split_upper[-1] += k[i]
        k = split_upper
    else:
        k = k.lower().split('_')
    k = [wordnet_lemmatizer.lemmatize(item, pos="n") for item in k]
    k = [wordnet_lemmatizer.lemmatize(item, pos="v") for item in k]
    k = [item for item in k if len(item) > 3 and (item != 'nasa')]
    return k


def get_rel_and_answer(tripples, keys):
    answers = []
    relations = []
    for i in range(len(tripples)):
        tri = tripples[i]
        k = keys[i]
        idx = tri.find(k)
        if idx == -1:
            idx = 0
        answers.append(tri[idx:].replace(k, '').strip())
        clean_k = clean_key(k)
        relations.append(clean_k)
        #new_tripples.append(tri[idx:].strip())
    return answers, relations

def one_pair(tripples, keys, terminals, non_terminals):
    answers, relations = get_rel_and_answer(tripples, keys)
    answer_relations = []
    for i, ans in enumerate(answers):
        answer_relations.append(' '.join(relations[i]) + ' ' + ans)
    alignments = [[] for t in terminals]
    for i, trip in enumerate(answer_relations):
        alignment = one_triple_to_phrase(trip, terminals)
        alignments[alignment].append(keys[i])
    return '|'.join(['&&'.join(alg) for alg in alignments])

def get_alignment():
    input_dir = "./annotation_human/"
    src_path = input_dir+"rst_ann_src.jsonl"
    tgt_path = input_dir+"rst_ann_tgt.jsonl"
    src_list = [line.strip() for line in open(src_path)]
    tgt_list = [line.strip() for line in open(tgt_path)]

    output_dir = "./annotation_human/"
    src_path = output_dir+"webnlg_ann_src.jsonl"
    tgt_path = output_dir+"webnlg_ann_tgt.jsonl"
    fpout_src = open(src_path, 'w')
    fpout_tgt = open(tgt_path, 'w')

    for i in range(len(src_list)):
        src = src_list[i]; tgt = tgt_list[i]
        # src data
        src_data = src.split('\t')
        tripples = src_data[:-1]
        keys = src_data[-1].split('|')
        key_count = {}
        new_keys = []
        # deal with same keys
        for item in keys:
            if item not in key_count:
                key_count[item] = 1
                new_keys.append(item)
            else:
                key_count[item] += 1
                new_keys.append(item + '_' + str(key_count[item]))
        keys = new_keys
        src = '\t'.join(tripples) + '\t' + '|'.join(keys)
        # tgt data
        tgt_data = tgt.split('\t')
        non_terminals = tgt_data[0]
        terminals = tgt_data[1:]
        # processing
        alignments = one_pair(tripples, keys, terminals, non_terminals)
        fpout_src.write(src + '\n')
        fpout_tgt.write(non_terminals + '\t' + alignments + '\t' + '\t'.join(terminals).replace('-lrb-', '(').replace('-rrb-', ')') + '\n')

    fpout_src.close()
    fpout_tgt.close()

if __name__ == '__main__':
    if sys.argv[1] == "alg":
        # Step2: Get rule-based alignments
        #        Also, generate the files need for viterbi annotator
        #        Go to ../src/ to run 1) sh ann_json.sh 2) sh ann_data.sh 3) annotation.sh for the viterbi annotation result
        #        The viterbi annotation will be stored in ../logs/abs_bert_cnndm.{chekpoint}.anno
        get_alignment()
    elif sys.argv[1] == "rst":
        # Step1: Get rst fils from the raw human annotations
        output_dir = "./annotation_human/"
        src_path = output_dir+"rst_ann_src.jsonl"
        tgt_path = output_dir+"rst_ann_tgt.jsonl"
        plan_path = output_dir+"rst_ann_gt_plan.jsonl"
        fpout_src = open(src_path, 'w')
        fpout_tgt = open(tgt_path, 'w')
        fpout_plan = open(plan_path, 'w')

        s_path = "../Human_eva/scripts/webnlg_ann_src.jsonl"
        t_path = "../Human_eva/scripts/webnlg_ann_tgt.jsonl"
        
        src_lines = [line.strip() for line in open(s_path)]
        for i, line in enumerate(open(t_path)):
            flist = line.strip().split('\t')
            for tgt in flist:
                tlist = tgt.split('[XXN]')
                patt = tlist[0]
                alg = tlist[1]
                segs = tlist[2].split(' [SEP] ')
                fpout_src.write(src_lines[i] + '\n')
                fpout_tgt.write(patt + '\t' + '\t'.join(segs) + '\n')
                fpout_plan.write(alg + '\n')
        fpout_src.close()
        fpout_tgt.close()
        fpout_plan.close()
