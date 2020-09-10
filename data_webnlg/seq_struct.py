#coding=utf8

import sys, os
import json
import re
import copy
import multiprocessing
from nltk.stem import WordNetLemmatizer

def label_classify(item):
    if item[0] == '(':
        if item[1] == 'F':
            return "fact"
        else:
            return "phrase"
    elif item[0] == ')':
        return "end"
    elif item[0] == '*':
        return "reference"
    return "token"

def one_sentence_summ(tree, wordnet_lemmatizer):
    label_stack = []
    fact_stack = []
    factid_stack = []
    factlex_stack = []
    phrase_stack = []
    res = []; f_res = []; f_struc = []; f_patt = []
    idx = 0; f_idx = 0
    while idx < len(tree):
        tok = tree[idx]
        cls = label_classify(tok)
        if cls == "fact":
            if len(tok.split('-')) > 1:
                fact_lexi = wordnet_lemmatizer.lemmatize(tok.split('-')[1].lower(), pos="v")
                if fact_lexi == '':
                    fact_lexi = '-'
            else:
                fact_lexi = '[UNK]'
            if len(factid_stack) == 0:
                parent_id = 0
                parent_v = fact_lexi
            else:
                parent_id = factid_stack[-1]
                parent_v = factlex_stack[-1]
            if len(label_stack) > 0 and label_stack[-1] == "phrase":
                argument = phrase_stack[-1]
            else:
                argument = '[ROOT]'
            f_res.append(fact_lexi + '|' + 'V' + '|' + '[FACT-'+str(f_idx)+']')
            f_struc.append(fact_lexi + '|' + argument + '|' + parent_v + "|" + str(parent_id))
            label_stack.append(cls)
            fact_stack.append(tok)
            factlex_stack.append(fact_lexi)
            factid_stack.append(f_idx)
            f_patt.append([])
            f_idx += 1
        elif cls == "phrase":
            label_stack.append(cls)
            phrase_stack.append(tok[1:])
            f_patt[factid_stack[-1]].append(tok[1:])
            if len(fact_stack) > 0:
                res.append('[BEG]|' + tok[1:] + '|' + factlex_stack[-1])
        elif cls == "end":
            pop_cls = label_stack.pop()
            if pop_cls == "fact":
                fact_stack.pop()
                factlex_stack.pop()
                factid_stack.pop()
            else:
                phrase_stack.pop()
        elif cls == "token":
            if tok.find('|') == -1:
                if len(label_stack) > 0 and label_stack[-1] == "phrase":
                    argument = phrase_stack[-1]
                else:
                    argument = 'UNKNOWN'
                if len(fact_stack) > 0:
                    res.append(tok + '|' + argument + '|' + factlex_stack[-1])
        idx += 1
    return res, f_res, f_struc, ['|'.join(item) for item in f_patt]

def clean_tree(tree):
    ctree = []; idx = 0
    label_stack = []
    while idx < len(tree):
        tok = tree[idx]
        cls = label_classify(tok)
        if cls == "fact":
            label_stack.append(cls)
            ctree.append(tok)
        elif cls == "phrase":
            label_stack.append(cls)
            ctree.append(tok)
        elif cls == "end":
            pop_cls = label_stack.pop()
            ctree.append(tok)
        elif cls == "token":
            if len(label_stack) > 0 and label_stack[-1] == 'fact':
                j = len(ctree) - 1
                while j >= 0:
                    if ctree[j] != ')':
                        break
                    j -= 1
                ctree.insert(j+1, tok)
            else:
                ctree.append(tok)
        idx += 1
    return ctree

def add_seg(res):
    for_tag = ""
    new_res = []
    for item in res:
        flist = item.split('|')
        tag = '|'.join(flist[1:])
        if for_tag != "" and for_tag != tag:
            new_res.append("[END]|" + for_tag)
        for_tag = tag
        new_res.append(item)
    new_res.append("[END]|" + for_tag)
    return new_res

def one_file(args):
    # read in
    wordnet_lemmatizer = WordNetLemmatizer()
    (doc, summary) = args
    doc = '\t'.join(doc.split(' <TSP> '))

    ress = []; f_ress = []; f_strucs = []; f_patts = []
    for summary_tree in summary:
        res, f_res, f_struc, f_patt = one_sentence_summ(clean_tree(summary_tree.split()), wordnet_lemmatizer)
        #if len(f_res) == 0:
        #print ("[SUMMARY_TREE]", summary_tree)
        #print ("res", res)
        #print ("f_res", f_res)
        #print ("f_struc", f_struc)
        #print ("f_patts", f_patt)
        #print ('--------------------')
        ress += res
        f_ress += f_res
        f_strucs += f_struc
        f_patts += f_patt
    #print ("ress", ress)
    #print ("f_ress", f_ress)
    #print ("f_struc", f_strucs)
    #print ("f_patts", f_patts)
    #print ('********************\n')
    return doc, ress, f_ress, f_strucs, f_patts

if __name__ == '__main__':
    input_dir = "./data-tree/"
    output_dir = "./data-struct/"
    input_path = input_dir+"webnlg_"+sys.argv[1]+'.jsonl'
    output_src_path = output_dir+"webnlg_"+sys.argv[1]+'_src.jsonl'
    output_tgt_path = output_dir+"webnlg_"+sys.argv[1]+'_tgt.jsonl'

    thread_num = 1
    pool = multiprocessing.Pool(processes=thread_num)

    batch = []
    fpout_src = open(output_src_path, 'w')
    fpout_tgt = open(output_tgt_path, 'w')
    for i, line in enumerate(open(input_path)):
        json_obj = json.loads(line.strip())
        document = json_obj['document']
        summary = json_obj['summary']
        batch.append((document, summary))
        if i%thread_num == 0:
            res = pool.map(one_file, batch)
            for r in res:
                (doc, res, f_res, f_struc, f_patt) = r
                if len(res) == 0:
                    continue
                fpout_src.write(doc + "\n")
                fpout_tgt.write(' '.join(f_res) + "\t" + ' '.join(res) + "\t" + ' '.join(f_struc) + '\t' + '\t'.join(f_patt) + "\n")
            del batch[:]
    if len(batch) != 0:
        res = pool.map(one_file, batch)
        for r in res:
            (doc, res, f_res, f_struc, f_patt) = r
            if len(res) == 0:
                continue
            fpout_src.write(doc + "\n")
            fpout_tgt.write(' '.join(f_res) + "\t" + ' '.join(res) + "\t" + ' '.join(f_struc) + '\t' + '\t'.join(f_patt) + "\n")
    fpout_src.close()
    fpout_tgt.close()
