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

def move_arg_before_R(tree, idx, root=1):
    # move args before R-ARGM-LOC,R-ARG0 to parent
    CHILD_LABEL='(CHILD_'
    label_stack = []
    ctree = []
    before_R = []
    finish_one_phrase = False
    init_idx = idx
    while idx < len(tree):
        tok = tree[idx]
        cls = label_classify(tok)
        if cls == "fact":
            if (len(ctree) > 0 and finish_one_phrase) or idx > init_idx:
                re_before_R, stree, idx = move_arg_before_R(tree, idx, 0)
                #print ("re_before_R:", re_before_R)
                #print ("stree:", stree)
                #print ("ctree:", ctree)
                #print ("finish_one_phrase:", finish_one_phrase)
                if len(re_before_R) == 0:
                    ctree += stree
                else:
                    re_before_R = [CHILD_LABEL+item[1:] if item[0] == '(' and (not item.startswith('(LHP')) else item for item in re_before_R]
                    if finish_one_phrase or init_idx == 0:
                        ctree = ctree[:-1] + re_before_R + [ctree[-1]] + stree
                    else:
                        ctree = re_before_R + ctree + stree
                #print ('\n\n')
            else:
                ctree.append(tok)
                label_stack.append(cls)
        elif cls == "phrase":
            if tok.startswith('(R-') and (not root) and ctree[0].startswith('(F'):
                before_R = copy.deepcopy(ctree[1:])
                del ctree[1:]
            ctree.append(tok)
            label_stack.append(cls)
        elif cls == "end":
            ctree.append(tok)
            pop_type = label_stack.pop()
            if pop_type == 'phrase':
                finish_one_phrase = True
            if len(label_stack) == 0:
                return before_R, ctree, idx
        elif cls == "token":
            ctree.append(tok)
        idx += 1


def loose_hanging_phrase(tree):
    LHP_PRE='(LHP_P_'
    LHP_SUF='(LHP_S_'
    last_pop = ''
    ctree = []; idx = 0
    label_stack = []
    phrase_stack = []
    lhphrase = []
    while idx < len(tree):
        tok = tree[idx]
        cls = label_classify(tok)
        if cls == "fact":
            label_stack.append(cls)
            ctree.append(tok)
            if len(phrase_stack)>0 and len(lhphrase) > 0:
                #if len(' '.join(lhphrase)) > 1:
                if len(' '.join(lhphrase)) > 0:
                    ctree.append(LHP_PRE+phrase_stack[-1])
                    ctree.extend(lhphrase)
                    ctree.append(')')
            del lhphrase[:]
            last_pop = ''
        elif cls == "phrase":
            label_stack.append(cls)
            phrase_stack.append(tok[1:])
            ctree.append(tok)
            last_pop = ''
        elif cls == "end":
            pop_cls = label_stack.pop()
            if pop_cls == 'phrase':
                ph_tag=phrase_stack.pop()
            if pop_cls == 'phrase' and last_pop == 'fact' and len(lhphrase)>0:
                if len(lhphrase) == 1:
                    k = len(ctree)-1
                    while k > -1:
                        if ctree[k] != ')':
                            break
                        k -= 1
                    ctree = ctree[:k+1] + lhphrase + ctree[k+1:] + [')']
                else:
                    ctree = ctree[:-1]
                    ctree.append(LHP_SUF+ph_tag)
                    ctree.extend(lhphrase+[')', ')', ')'])
            else:
                ctree.extend(lhphrase)
                ctree.append(tok)
            last_pop = pop_cls
            del lhphrase[:]
        elif cls == "token":
            lhphrase.append(tok)
        idx += 1
    return ctree

def sub_tree(tree, idx):
    ph_stack = ["F"]
    subtrees = []
    phrases = []
    tokens = []
    while idx < len(tree):
        tok = tree[idx]
        label = label_classify(tok)
        if label == 'fact':
            if len(tokens) > 0:
                phrases.append(' '.join(tokens))
                subtrees.append('__tok__')
                del tokens[:]
            if len(ph_stack) > 1:
                subtrees.append(ph_stack[-1])
            subtrees.append(tok)
            #print ("IN>>>", tok)
            s_tree, sub_phs, idx = sub_tree(tree, idx+1)
            phrases.extend(sub_phs)
            subtrees = subtrees + s_tree
            if len(ph_stack) > 1:
                subtrees = subtrees + [')', ')']
            else:
                subtrees = subtrees + [')']
            #print ("OUT<<<", phrases)
            #print ("OUT<<<", subtrees)
            #print ("OUT<<<", '~~~~~~~~~')
        elif label == 'phrase':
            ph_stack.append(tok)
        elif label == 'token':
            tokens.append(tok)
        elif label == 'end':
            ph_stack.pop()
            #print (ph_stack)
            #print (tokens)
            if len(ph_stack) == 0:
                if len(tokens) > 0:
                    phrases.append(' '.join(tokens))
                    subtrees.append('__tok__')
                return subtrees, phrases, idx
        idx += 1
    return subtrees, phrases, idx

def at_left(subtrees, relations, idx):
    l_sub = subtrees[:idx]
    r_sub = subtrees[idx+2:]
    l_re = relations[:idx]
    r_re = relations[idx+2:]

    l_tree = subtrees[idx]
    r_tree = subtrees[idx+1]
    relation = '('+relations[idx+1]
    new_node = [relation]
    new_node.append(l_tree)
    new_node.append(r_tree)
    new_node.append(')')
    new_node = ' '.join(new_node)

    subtrees = l_sub + [new_node] + r_sub
    relations = l_re + ['TOK'] + r_re
    return subtrees, relations

def at_right(subtrees, relations, idx):
    l_sub = subtrees[:idx-1]
    r_sub = subtrees[idx+1:]
    l_re = relations[:idx-1]
    r_re = relations[idx+1:]

    l_tree = subtrees[idx-1]
    r_tree = subtrees[idx]
    relation = '('+relations[idx-1]+'-of'
    new_node = [relation]
    new_node.append(l_tree)
    new_node.append(r_tree)
    new_node.append(')')
    new_node = ' '.join(new_node)

    subtrees = l_sub + [new_node] + r_sub
    relations = l_re + ['TOK'] + r_re
    return subtrees, relations

def list2rst(subtrees, relations):
    while len(subtrees) > 1:
        idx = relations.index('TOK')
        if idx == 0:
            subtrees, relations = at_left(subtrees, relations, idx)
        else:
            subtrees, relations = at_right(subtrees, relations, idx)
    return subtrees

def tree2rst(tree, idx):
    subtrees = []
    relations = []
    phrase_stack =[]
    fact_stack = []
    label_stack = []
    curr_fact = ""
    while idx < len(tree):
        tok = tree[idx]
        label = label_classify(tok)
        if label == 'fact':
            if len(phrase_stack) > 0:
                rst, idx = tree2rst(tree, idx)
                subtrees.append(' '.join(rst))
            else:
                label_stack.append(label)
                fact_stack.append(tok[1:])
        elif label == 'phrase':
            label_stack.append(label)
            phrase_stack.append(tok[1:])
        elif label == 'token':
            subtrees.append(fact_stack[-1])
            relations.append('TOK')
        elif label == 'end':
            l_pop = label_stack.pop()
            if l_pop == 'phrase':
                relations.append(phrase_stack.pop())
            else:
                fact_stack.pop()
            if len(fact_stack) == 0:
                subtrees = list2rst(subtrees, relations)
                return subtrees, idx
        idx += 1

def cands_for_Rv(tree):
    idx = 0
    tok_stack = []
    cls_stack = []
    cands_stack = {}
    for tok in tree:
        if label_classify(tok) == 'fact':
            cands_stack[tok] = True
    while idx < len(tree):
        tok = tree[idx]
        cls = label_classify(tok)
        if cls == "fact":
            tok_stack.append(tok)
            cls_stack.append(cls)
        elif cls == "phrase":
            if tok.startswith('(R'):
                cands_stack[tok_stack[-1]] = False
            tok_stack.append(tok)
            cls_stack.append(cls)
        elif cls == "end":
            tok_stack.pop()
            cls_stack.pop()
        idx += 1
    return cands_stack

def fake_R_v(tree):
    ctree = []; idx = 0
    tok_stack = []
    cls_stack = []
    cands_stack = cands_for_Rv(tree)
    while idx < len(tree):
        tok = tree[idx]
        cls = label_classify(tok)
        if cls == "fact":
            tok_stack.append(tok)
            cls_stack.append(cls)
            ctree.append(tok)
        elif cls == "phrase":
            if cands_stack[tok_stack[-1]] and (tok == '(V' or tok == '(ARGM-LOC') and len(cls_stack) > 1 and \
                cls_stack[-1] == "fact" and cls_stack[-2] == "phrase" and \
                tok_stack[-2] == "(ARG2":
                    tok = '(R-' + tok[1:]
                    cands_stack[tok_stack[-1]] = False
            tok_stack.append(tok)
            cls_stack.append(cls)
            ctree.append(tok)
        elif cls == "end":
            tok_stack.pop()
            pop_cls = cls_stack.pop()
            ctree.append(tok)
        elif cls == "token":
            ctree.append(tok)
        idx += 1
    return ctree

def upmerge_facts(tree, idx=0):
    ctree = []
    cls_stack = ["fact"]
    tok_stack = ["(F-verb"]
    last_verb = False
    while idx < len(tree):
        tok = tree[idx]
        cls = label_classify(tok)
        if cls == "fact":
            if len(ctree) > 0:
                sub_tree, last_verb, idx = upmerge_facts(tree, idx)
                del sub_tree[0]
                del sub_tree[-1]
                if tok_stack[-1] == '(ARG2' and last_verb:
                    del ctree[-1]
                    ctree.extend(sub_tree)
                    tok_stack.pop()
                    cls_stack.pop()
                    idx += 1
                else:
                    ctree.append(tok)
                    ctree.extend(sub_tree)
                    ctree.append(')')
            else:
                ctree.append(tok)
        elif cls == "phrase":
            cls_stack.append(cls)
            tok_stack.append(tok)
            if tok in ['(R-V']:
                last_verb = True
            else:
                last_verb = False
            ctree.append(tok)
        elif cls == "end":
            cls_stack.pop()
            tok_stack.pop()
            ctree.append(tok)
            if len(cls_stack) == 0:
                return ctree, last_verb, idx
        elif cls == "token":
            ctree.append(tok)
        idx += 1


def hard_code_tricks(tree):
    tree_str = ' '.join(tree)
    tree_str = tree_str.replace('(V comes ) (ARG2 from )', '(V comes from )')
    tree_str = tree_str.replace('(V come ) (ARG2 from )', '(V come from )')
    tree_str = tree_str.replace('(V came ) (ARG2 from )', '(V came from )')
    return tree_str.split()

def one_tree(tree):
    original_tree = tree

    # Clean Tree
    tree = clean_tree(tree.split())
    tree = hard_code_tricks(tree)

    # Deal with loose hanging phrase
    tree = loose_hanging_phrase(tree)
    if len(tree) == 0:
        return 'F1-null', [original_tree]

    # Re-arrange some arguments
    tree = fake_R_v(tree)
    #print (' '.join(tree))
    _, tree, _ = move_arg_before_R(tree, 0)
    #print (' '.join(tree))
    if len(tree) == 0:
        return 'F1-null', [original_tree]

    # Upmerge some subtree
    tree, _, _ = upmerge_facts(tree, idx=0)

    # Build RST
    tree, tokens, _ = sub_tree(tree, 0)
    rst_tree, _ = tree2rst(tree, 0)

    #print (rst_tree[0], tokens)
    #print ('\n\n')
    return rst_tree[0], tokens

def process_src(doc):
    doc = doc.split(' <TSP> ')
    strings = []; relations = []
    for sentence in doc:
        sent_str = []
        relation = ""
        tripples = sentence.strip().split(' ')
        for item in tripples:
            tripple = item.split('|')
            tok = tripple[0]
            relation = tripple[2]
            sent_str.append(tok)
        strings.append(' '.join(sent_str))
        relations.append(relation)
    return '\t'.join(strings), '|'.join(relations)

def merge_and(tgt_rsts, is_and):
    idx = is_and.index(True)
    l_tree = tgt_rsts[idx-1]
    r_tree = tgt_rsts[idx]

    l_list = tgt_rsts[:idx-1]
    r_list = tgt_rsts[idx+1:]
    l_and = is_and[:idx-1]
    r_and = is_and[idx+1:]

    new_tree = ['(AND']
    new_tree.append(l_tree)
    new_tree.append(r_tree)
    new_tree.append(')')
    new_tree = ' '.join(new_tree)

    tgt_rsts = l_list + [new_tree] + r_list
    is_and = l_and + [False] + r_and

    return tgt_rsts, is_and

def merge_seq(tgt_rsts):
    new_tree = ['(S-LIST']
    l_list = tgt_rsts[0]
    r_list = tgt_rsts[1]
    if len(l_list) == 1:
        l_list = ' '.join(['(ROOT', l_list[0], ')'])
    if len(r_list) == 1:
        r_list = ' '.join(['(ROOT', r_list[0], ')'])
    new_tree.append(l_list)
    new_tree.append(r_list)
    new_tree.append(')')
    new_tree = ' '.join(new_tree)
    return [new_tree] + tgt_rsts[2:]

def merge_sentences(tgt_rsts, tgt_toks):
    is_and = [tok[0].split()[0].lower() == 'and' for tok in tgt_toks]
    is_and[0] = False
    # Merge 'and'
    while sum(is_and) > 0:
        tgt_rsts, is_and = merge_and(tgt_rsts, is_and)
    # Merge s-list
    if len(tgt_rsts) == 1:
        tgt_rsts[0] = '(S '+tgt_rsts[0]+' )'
    else:
        while len(tgt_rsts) > 1:
            tgt_rsts = merge_seq(tgt_rsts)
    non_terminal = tgt_rsts[0]
    terminal = []
    for i, phrases in enumerate(tgt_toks):
        '''
        if i<len(tgt_toks)-1:
            if not tgt_toks[i+1][0].startswith('and'):
                phrases[-1] += ' .'
        else:
            phrases[-1] += ' .'
        '''
        terminal.extend(phrases)
    return non_terminal, terminal


def split_tree(tree):
    ctree = []; idx = 0
    label_stack = []
    new_trees = []
    tree = tree.split()
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
            if len(label_stack) == 0:
                new_trees.append(' '.join(ctree))
                ctree = []
        elif cls == "token":
            ctree.append(tok)
        idx += 1
    if len(ctree) > 0:
        return [' '.join(tree)]
    return new_trees


def process_tgt(tgt):
    tgt_rsts = []; tgt_toks = []
    new_tgt = []
    for tree in tgt:
        new_tgt.extend(split_tree(tree))
    for tree in new_tgt:
        rst, toks = one_tree(tree)
        tgt_rsts.append(rst)
        tgt_toks.append(toks)
    non_terminal, terminal = merge_sentences(tgt_rsts, tgt_toks)
    return non_terminal, terminal

def one_file(args):
    (doc, summary) = args
    doc = doc.strip()
    if doc == '':
        return '', '', '', []
    doc, relations = process_src(doc)
    if ' '.join(summary) == '':
        return doc, relations, '', []
    rst, toks = process_tgt(summary)
    return doc, relations, rst, toks

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_dir = "./data-tree/"
        output_dir = "./data-rst/"
        input_path = input_dir+"e2e_"+sys.argv[1]+'.jsonl'
        output_src_path = output_dir+"e2e_"+sys.argv[1]+'_src.jsonl'
        output_tgt_path = output_dir+"e2e_"+sys.argv[1]+'_tgt.jsonl'

        thread_num = 30
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
                    (doc, relations, rst, toks) = r
                    if doc == '':
                        continue
                    fpout_src.write(doc + '\t' + relations + '\n')
                    fpout_tgt.write(rst + '\t' + '\t'.join(toks) + '\n')
                del batch[:]
        if len(batch) != 0:
            res = pool.map(one_file, batch)
            for r in res:
                (doc, relations, rst, toks) = r
                fpout_src.write(doc + '\t' + relations + '\n')
                fpout_tgt.write(rst + '\t' + '\t'.join(toks) + '\n')
        fpout_src.close()
        fpout_tgt.close()
    else:
        #one_tree('(F1-retired (ARG0 (F2-was (ARG1 Buzz Aldrin ) , (ARG1 whose real name ) (V was ) (ARG2 Edwin E Aldrin Jr ) ) , ) (ARGM-TMP has now ) (V retired ) ) .')
        #one_tree('(F1-is (ARG1 He ) (V is ) (ARG2 (F2-born (ARG1 an American ) (V born ) (ARGM-LOC in Glen Ridge , New Jersey ) ) ) ) ')
        #one_tree('(F1-became (ARGM-TMP After (F2-graduating (V graduating ) (ARG1 from MIT ) (ARGM-TMP in 1963 ) (ARGM-MNR with a doctorate in science ) ) ) (ARG1 he ) (V became ) (ARG2 a fighter pilot ) )')
        #one_tree('(F1-operated (ARG1 Al Asad airbase ) (V is operated ) (ARG0 by (F2-participated (ARG0 the United States Air Force ) (R-ARG0 who ) (V participated ) (ARG1 in the battles during Operation Enduring Freedom ) ) ) )')
        #one_tree('(F1-make (ARGM-DIS and ) (ARGM-DIS also ) (V make ) (ARG1 use ) (ARG2 of ) (ARG1 the aircraft fighter General Dynamics F-16 Fighting Falcon ) ) ')
        #process_tgt(['(F1-born (ARG1 (F2-known (ARG1 (F3-Retired (V Retired ) (ARG1 American fighter pilot ) ) Buzz Aldrin ) , (ARGM-ADV also ) (V known ) (ARG2 as Edwin E. Aldrin Jr. ) ) , ) (V was born ) (ARGM-LOC in Glen Ridge , New Jersey ) ) '])
        #process_tgt(['(F1-was (ARG1 (F2-began (ARG0 He ) ) ) (V began working for NASA in 1963 was ) (ARG2 a member of (F3-run (ARG1 Apollo 11 ) (R-ARG1 which ) (V was run ) (ARG0 by NASA ) ) ) ) .'])
        #process_tgt(["(F1-known (ARG1 India ) (V is known ) (ARG2 for the river Ganges ) )", "(F1-is (ARGM-DIS and ) (ARG1 the largest city being Mumbai ) (V is ) (ARGM-DIS also ) (ARG2 home to (F2-positioned (ARG1 the state of Kerala ) (R-ARG1 which ) (V is positioned ) (ARG2 with Mahe ) (ARG2 to (F3-'s (ARG1 it ) (V 's ) (ARG2 northwest ) ) ) ) ) ) .", "(F1-is (ARG1 Kerala ) (V is ) (ARGM-DIS also ) (ARG2 (F2-located (ARGM-LOC where ) (ARG1 AWH Engineering College ) (V is located ) (ARGM-LOC within the city of Kuttikkattoor ) ) ) ) .", "(F1-has (ARG0 The college ) (ARGM-TMP currently ) (V has ) (ARG1 250 members of staff ) ) ."])
        #one_tree('(F1-is (ARG1 Celery ) (V is ) (ARG2 an ingredient of Bakso , a dish from the country of Indonesia , (F2-is (ARGM-LOC where ) (ARG1 Jusuf Kalla ) (V is ) (ARG2 a leader ) ) ) ) .')
        #one_tree('(F1-is (ARG1 Bakso ) (V is ) (ARG2 a dish from the country of Indonesia , (F2-is (ARGM-LOC where ) (ARG1 the capital ) (V is ) (ARG2 Jakarta ) ) ) )')
        #one_tree('(F1-followed (ARGM-DIS and ) (V followed ) (ARG1 by (F2-written (ARG1 Ring of Fire II ) (R-ARG1 which ) (V is written ) (ARGM-MNR in English ) ) ) ) .')
        #one_tree('(F1-is (ARG1 It ) (V is ) (ARG2 (F2-has (ARG0 a dessert ) (R-ARG0 that ) (V has ) (ARG1 the main ingredients of ground almond , jam , butter and eggs ) ) ) )')
        #one_tree('(F1-comes (ARG1 Bhajji ) (V comes ) (ARG3 from (F2-is (ARGM-LOC the country India ) , (R-ARGM-LOC where ) (ARG1 the currency ) (V is ) (ARG2 the rupee ) ) ) )')
        #one_tree('(F1-is (ARG1 (F2-led (ARG1 The Flemish region ) , (R-ARG1 which ) (V is led ) (ARG0 by the Flemish government ) ) ) , (V is ) (ARG2 part of Belgium ) ) .')
        #one_tree('(F1-runs (ARG0 The Flemish Government ) (V runs ) (ARG1 the flemish region in (F2-served (ARGM-LOC Belgium ) (R-ARGM-LOC where ) (ARG2 Antwerp ) (V is served ) (ARG0 by Antwerp International airport ) ) ) ) .')
        #one_tree('(F1-are (ARG1 The US (F3-based (V based ) (ARG1 Mason School ) ) of Business ) (V are ) (ARG2 the current tenants of (F2-designed (ARG1 Alan B Miller Hall in Virginia ) (R-ARG1 which ) (V was designed ) (ARG0 by the architect Robert A M Stern ) ) ) ) .')
        #one_tree("(F1-is (ARGM-DIS and ) (ARG1 the largest city being Mumbai ) (V is ) (ARGM-DIS also ) (ARG2 home to (F2-positioned (ARG1 the state of Kerala ) (R-ARG1 which ) (V is positioned ) (ARG2 with Mahe ) (ARG2 to (F3-'s (ARG1 it ) (V 's ) (ARG2 northwest ) ) ) ) ) ) .")
        #one_tree('(F1-is (ARG1 Bread ) (V is ) (ARG2 an ingredient of (F2-is (ARG1 Ajoblanco ) , (R-ARG1 which ) (V is ) (ARG2 from Spain ) ) ) )')
        #one_tree('(F1-is (ARGM-DIS and ) (ARG1 Hilmi G\u00fcner ) , (V is dedicated ) (ARG2 to (F2-killed (ARG1 the Ottoman Army soldiers ) (V killed ) (ARG2 in the Battle of Baku ) ) ) ) .')
        #one_tree('(F1-is (ARG1 AWH Engineering College in Kuttikkattoor , India ) (V is ) (ARG2 (F2-has (LHP_P_ARG2 in ) (ARG0 the state of Kerala ) (R-ARG0 which ) (V has ) (ARG1 Mahe ) (ARGM-LOC to its northwest ) ) ) )')
        #one_tree("(F1-is (ARG1 The Baku Turkish Martyrs ' Memorial ) (V is located ) (ARG2 (F2-known (LHP_P_ARG2 in ) (ARGM-LOC Azerbaijan , ) (R-ARGM-LOC where ) (ARG1 it ) (V is known ) (ARG2 as Türk Sehitleri Aniti ) ) ) )")
        #one_tree('(F1-is (ARG1 He ) (V is ) (ARG2 (F2-born (ARG1 an American ) (V born ) (ARGM-LOC in Glen Ridge , New Jersey ) ) ) )')
        #one_tree('(F1-is (ARG1 Andrews County Airport ) (V is ) (ARG2 (F2-is (LHP_P_ARG2 in Texas in ) (ARGM-LOC the U.S.A. ) (R-ARGM-LOC where ) (ARG1 Spanish ) (V is ) (ARG2 (F3-spoken (LHP_P_ARG2 one of ) (ARG1 the languages ) (V spoken ) ) ) ) ) )')

        '''
        one_tree('(F1-is (ARGM-DIS and ) (V is ) (ARG2 (F2-come (LHP_P_ARG2 where ) (ARG1 Bakewell tarts ) (V come ) (ARG2 from ) ) ) )')
        one_tree('(F1-is (ARG1 Italy ) (V is ) (ARG2 (F2-comes (LHP_P_ARG2 the country ) (ARG1 Amatriciana sauce ) (V comes ) (ARG2 from ) ) ) )')
        one_tree('(F1-is (ARG1 Philippines ) (V is ) (ARG2 (F2-comes (LHP_P_ARG2 the country the dish ) (ARG1 Batchoy ) (V comes ) (ARG2 from ) ) ) )')
        one_tree('(F1-is (ARGM-DIS and ) (ARG1 English ) (V is ) (ARG2 (F2-spoken (ARG1 the language ) (V spoken ) ) ) )')
        one_tree('(F1-is (ARG1 Ayam penyet ) (V is ) (ARG2 (F2-made (ARG1 a Javanese dish ) (V made ) ) ) (ARGM-LOC nationwide ) )')

        one_tree('(F1-is (ARG1 Another part of New York City ) (V is ) (ARG2 (F2-is (LHP_P_ARG2 Manhattan ) (ARGM-LOC where ) (ARG1 Gale Brewer ) (V is ) (ARG2 the leader ) ) ) )')
        one_tree('(F1-is (ARG1 Beef kway teow ) (V is ) (ARG2 (F2-is (LHP_P_ARG2 a dish of ) (ARGM-LOC Singapore ) (ARGM-LOC where ) (ARG1 Tony Tan ) (V is ) (ARG2 a leader ) ) ) )')
        one_tree('(F1-is (ARG1 Agra Airport ) (V is ) (ARG2 (F2-is (LHP_P_ARG2 in India ) (ARGM-LOC where ) (ARG1 one of its leaders ) (V is ) (ARG2 T.S. Thakur ) ) ) )')
        one_tree('(F1-serves (ARG0 Athens International Airport ) (V serves ) (ARG2 (F2-is (LHP_P_ARG2 the city of Athens in ) (ARGM-LOC Greece ) (ARGM-LOC where ) (ARG1 Alexis Tsipras ) (V is ) (ARG2 the leader ) ) ) )')
        '''
        #one_tree('(F1-is (ARG1 Arròs negre ) (V is ) (ARG2 a traditional dish from Spain ) ) (F2-includes , (ARG2 it ) (V includes ) (ARG1 squid . ) )')
        #tree_list = split_tree('(F1-is (ARG1 Arròs negre ) (V is ) (ARG2 a traditional dish from Spain ) ) (F2-includes , (ARG2 it ) (V includes ) (ARG1 squid . ) )')
        #one_tree('(F1-is (ARG1 The Phoenix ) (V is ) (ARG2 (F2-is (ARG1 (F3-sells (ARG0 a riverside restaurant ) (R-ARG0 that ) (V sells ) (ARG1 traditional English food , ) ) ) (ARGM-ADV however ) (V is ) (ARG2 quite expensive . ) ) ) )')
        one_tree('(F1-rated (ARG1 (F2-serves (LHP_P_ARG1 Aromi , ) (ARG0 the family friendly coffee shop ) (R-ARG0 that ) (V serves ) (ARG1 English food ) (ARGM-LOC by the riverside ) ) ) (ARGM-MNR has been average ) (V rated , ) )')
