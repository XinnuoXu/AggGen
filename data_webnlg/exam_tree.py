#coding=utf8

def process_summary(summary):
    toks = summary.strip().split()
    new_toks = []
    for item in toks:
        if item[0] == '(' or item[0] == ')':
            continue
        new_toks.append(item)
    summary = ' '.join(new_toks)
    summary = summary.replace('-lrb-', '(').replace('-rrb-', ')')
    return summary

def load_tree(tag):
    tree_pairs = {}
    trees = {}
    for line in open('data-tree/webnlg_'+tag+'.jsonl'):
        json_obj = json.loads(line.strip())
        summaries = json_obj["summary"]
        document = json_obj["document"]
        trees[document] = '\t'.join(summaries)
        summaries = [process_summary(s) for s in summaries]
        tree_pairs[document] = ' '.join(summaries)
    return tree_pairs, trees

def load_raw_srl(tag):
    src_path = './data/'+tag+'-webnlg-src.txt'
    tgt_path = './data/'+tag+'-webnlg-tgt.txt'
    srcs = [line.strip() for line in open(src_path)]
    raw_pairs = {}
    for i, line in enumerate(open(tgt_path)):
        line = line.strip().replace(' <s> ', ' ').replace(' <s>', '')
        raw_pairs[srcs[i]] = line

    srl_pairs = {}
    srl_path = './data-srl/'+tag+'-webnlg-tgt.txt'
    for i, line in enumerate(open(srl_path)):
        srl_pairs[srcs[i]] = line.strip()

    return raw_pairs, srl_pairs

if __name__ == '__main__':
    import sys
    import json
    tag = 'train'
    #tag = 'dev'
    tree_pairs, trees = load_tree(tag)
    raw_pairs, srl_pairs = load_raw_srl(tag)
    for key in raw_pairs:
        if tree_pairs[key] != raw_pairs[key]:
            print (raw_pairs[key])
            print (tree_pairs[key])
            print ('--------------------')
            print (trees[key])
            print ('--------------------')
            print (srl_pairs[key])
            print ('\n\n')