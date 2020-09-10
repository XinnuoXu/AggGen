import sys, os
import json
import re
import multiprocessing

pattern = re.compile(r'\[[^\]]+\]')

def tags_check(tags):
    # ignore tags with less than(include) two components
    if len(set([tag for tag in tags if tag != "O"])) < 3:
        return False
    return True

def one_verb(v_item, slen):
    if ("verb" not in v_item) \
        or ("tags" not in v_item) \
        or ("description" not in v_item) \
        or "B-V" not in v_item["tags"]:
        return -1, -1, -1, []
    if not tags_check(v_item["tags"]):
        return -1, -1, -1, []

    b_idx = 0; e_idx = slen - 1
    # Fact begin index
    while b_idx < slen and v_item["tags"][b_idx] == "O":
        b_idx += 1
    # Fact end index
    while e_idx >= 0 and v_item["tags"][e_idx] == "O":
        e_idx -= 1
    e_idx += 1
    # Fact verb index
    v_idx = v_item["tags"].index("B-V")
    if b_idx >= e_idx:
        return -1, -1, -1, []
    # Extract semantic roles
    segments = []; seg_b_idx = -1; last_seg_label = ""
    for i in range(b_idx, e_idx):
        if v_item["tags"][i][0] == "B":
            if seg_b_idx != -1:
                segments.append((seg_b_idx, i, last_seg_label))
            last_seg_label = v_item["tags"][i][2:]
            seg_b_idx = i
        elif v_item["tags"][i][0] == "O":
            if seg_b_idx != -1:
                segments.append((seg_b_idx, i, last_seg_label))
            last_seg_label = ""; seg_b_idx = -1
    if seg_b_idx != -1:
        segments.append((seg_b_idx, e_idx, last_seg_label))
    return b_idx, e_idx, v_idx, segments

def get_rid_of_confliction(seg_collection, segments, b_idx, e_idx, v_idx):
    #print ("\nsegments: ", segments)
    #print ("seg_collection: ", seg_collection)
    min_idx = 10000
    for item in seg_collection.keys():
        min_idx = min(min_idx, min(item))
    max_idx = -1
    for item in seg_collection.keys():
        max_idx = max(max_idx, max(item))
    if e_idx < min_idx or b_idx > max_idx:
        return seg_collection, segments, b_idx, e_idx, v_idx

    replace_key = None
    for i, key in enumerate(seg_collection):
        if b_idx >= key[0] and e_idx <= key[1]:
            bad_case = False
            for j, k in enumerate(seg_collection):
                if j == i:
                    continue
                for seg in segments:
                    #if seg[0] >= k[0] and seg[1] <= k[1]:
                    if not (seg[1] <= k[0] or seg[0] >= k[1]):
                        bad_case = True
                        break
                if bad_case:
                    break
            if bad_case:
                print ("bad_case")
                return seg_collection, [], -1, -1, -1
            replace_key = key
            break
    
    if replace_key == None:
        for key in seg_collection:
            if v_idx >= key[0] and e_idx <= key[1]:
                for idx, seg in enumerate(segments):
                    if seg[0] >= key[0]:
                        replace_key = key
                        b_idx = seg[0]
                        segments = segments[idx:]
                        break
                break

    if replace_key == None:
        for key in seg_collection:
            if b_idx >= key[0] and v_idx <= key[1]:
                for idx in range(len(segments)-1, -1, -1):
                    seg = segments[idx]
                    if seg[1] <= key[1]:
                        replace_key = key
                        segments = segments[:idx+1]
                        e_idx = seg[1]
                        break
                break

    if replace_key != None:
        if replace_key[0] < b_idx:
            new_key_pre = (replace_key[0], b_idx)
            seg_collection[new_key_pre] = seg_collection[replace_key]
        if replace_key[1] < e_idx:
            new_key_suf = (e_idx, replace_key[1])
            seg_collection[new_key_suf] = seg_collection[replace_key]
        del seg_collection[replace_key]
        return seg_collection, segments, b_idx, e_idx, v_idx

    return seg_collection, [], -1, -1, -1

def get_tree(facts, words, slen):
    fact_id = 1; f_labels = [""] * slen; e_labels = [""] * slen; o_labels = [""] * slen; seg_collection = {}
    for item in sorted(facts.items(), key = lambda d:d[1][0], reverse = True):
        (plen, b_idx, e_idx, v_idx, segments) = item[1]
        if len(seg_collection) > 0:
            seg_collection, segments, b_idx, e_idx, v_idx = get_rid_of_confliction(seg_collection, segments, b_idx, e_idx, v_idx)
            if len(segments) == 0:
                continue
        if b_idx < 0 or e_idx > slen:
            continue
        # Label facts
        overlap = ""; b_idx = -1
        for seg in segments:
            (seg_b, seg_e, seg_type) = seg
            if (seg_b, seg_e) not in seg_collection:
                if b_idx == -1:
                    b_idx = seg_b
                    f_labels[b_idx] += "(F" + str(fact_id) + "-" + words[v_idx] + " "
                    e_labels[e_idx-1] += " )";
                seg_collection[(seg_b, seg_e)] = "F" + str(fact_id) + "-" + seg_type
                f_labels[seg_b] += overlap; overlap = ""
                f_labels[seg_b] += "(" + seg_type + " "
                e_labels[seg_e-1] += " )"
            else:
                overlap += "(" + seg_type + " *trace-" + seg_collection[(seg_b, seg_e)] + "* ) "
        fact_id += 1
    for i in range(0, slen):
        if words[i] == '(':
            words[i] = "-lrb-"
        elif words[i] == ')':
            words[i] = "-rrb-"
        words[i] = f_labels[i] + words[i] + e_labels[i]
    return words
        
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

def merge_tokens(tree):
    type_stack = []; orphans = []
    new_tree = []
    for i, tok in enumerate(tree):
        if len(tok) == 0:
            continue
        cls = label_classify(tok)
        if cls == "fact":
            type_stack.append(cls)
            new_tree.append(tok)
        elif cls == "phrase":
            type_stack.append(cls)
            new_tree.append(tok)
            if len(orphans) > 0:
                new_tree.extend(orphans)
                del orphans[:]
        elif cls == "end":
            new_tree.append(tok)
            pop_type = type_stack.pop()
        else:
            if len(type_stack) > 0 and type_stack[-1] == "phrase":
                new_tree.append(tok)
            else:
                if len(tok) > 1:
                    orphans.append(tok)
                elif len(orphans) > 0:
                    orphans.append(tok)
                else:
                    new_tree.append(tok)
    if len(orphans) > 0:
        return tree
    return new_tree

def final_touch_merge(tree):
    type_stack = []; orphans = []; new_tree = []
    for i, tok in enumerate(tree):
        if len(tok) == 0:
            continue
        cls = label_classify(tok)
        if cls == "fact":
            type_stack.append(cls)
            new_tree.append(tok)
        elif cls == "phrase":
            if len(orphans) > 0:
                idx = len(new_tree)-1
                while idx >= 0:
                    if new_tree[idx] == ')' or new_tree[idx][0] == '(':
                        idx -= 1
                        continue
                    break
                if idx > 0:
                    new_tree = new_tree[:idx+1] + orphans + new_tree[idx+1:]
                    new_tree.append(tok)
                else:
                    new_tree.append(tok)
                    new_tree.extend(orphans)
                orphans = []
            else:
                new_tree.append(tok)
            type_stack.append(cls)
        elif cls == "end":
            new_tree.append(tok)
            pop_type = type_stack.pop()
        else:
            if len(type_stack) > 0 and type_stack[-1] == "phrase":
                new_tree.append(tok)
            else:
                orphans.append(tok)
    if len(orphans) > 0:
        idx = len(new_tree)-1
        while idx >= 0:
            if new_tree[idx] == ')' or new_tree[idx][0] == '(':
                idx -= 1
                continue
            break
        new_tree = new_tree[:idx+1] + orphans + new_tree[idx+1:]
    return new_tree

def final_tune(tree, slen):
    b_idx = 0; e_idx = slen - 1
    new_tree = merge_tokens(" ".join(tree).split(" "))
    new_tree = final_touch_merge(new_tree)
    return new_tree

def merge_verbs(prev_fact, v_idx):
    (span, b_idx, e_idx, pre_v_idx, segments) = prev_fact
    if v_idx-pre_v_idx == 1:
        remove_index = -1; vi = 0
        after_verb_turple = None
        while vi < len(segments):
            item = segments[vi]
            if item[2] == 'V':
                verb_turple = (item[0], item[1]+1, item[2])
                if vi < len(segments) - 1 and segments[vi+1][0] == v_idx:
                    after_verb_turple = (segments[vi+1][0]+1, segments[vi+1][1], segments[vi+1][2])
                    if after_verb_turple[0] == after_verb_turple[1]:
                        remove_index = vi+1
                break
            vi += 1
        segments[vi] = verb_turple
        if remove_index > -1:
            del segments[remove_index]
        elif after_verb_turple is not None:
            segments[vi+1] = after_verb_turple
    #if v_idx-pre_v_idx == -1:
        # No such cases
    return (span, b_idx, e_idx, pre_v_idx, segments)

def one_summary(item):
    if "words" not in item or "verbs" not in item:
        return 
    words = item["words"]; slen = len(words); facts = {}; summary = " ".join(words)
    for i, v_item in enumerate(item["verbs"]):
        b_idx, e_idx, v_idx, segments = one_verb(v_item, slen)
        if b_idx == -1 or len(segments) == 1:
            continue
        if (b_idx, e_idx) in facts:
            facts[(b_idx, e_idx)] = merge_verbs(facts[(b_idx, e_idx)], v_idx)
            continue
        facts[(b_idx, e_idx)] = (e_idx - b_idx, b_idx, e_idx, v_idx, segments)
    tree = get_tree(facts, words, slen)
    tree = final_tune(tree, slen)
    return " ".join(tree)

def one_file(args):
    (srcs, tgts) = args
    tgts = [one_summary(o) for o in tgts]
    out_json = {}
    out_json["summary"] = tgts
    out_json["document"] = srcs
    return out_json

if __name__ == '__main__':
    '''
    tree_srl = [{"verbs": [{"verb": "authored", "description": "1634 : [ARG1: the Bavarian Crisis] , [V: authored] [ARG0: by Virginia DeMarce]", "tags": ["O", "O", "B-ARG1", "I-ARG1", "I-ARG1", "O", "B-V", "B-ARG0", "I-ARG0", "I-ARG0"]}], "words": ["1634", ":", "the", "Bavarian", "Crisis", ",", "authored", "by", "Virginia", "DeMarce"]}, {"verbs": [{"verb": "was", "description": "and Eric Flint , [V: was] preceded by Grantville Gazette III .", "tags": ["O", "O", "O", "O", "B-V", "O", "O", "O", "O", "O", "O"]}, {"verb": "preceded", "description": "and [ARG1: Eric Flint] , was [V: preceded] [ARG0: by Grantville Gazette III] .", "tags": ["O", "B-ARG1", "I-ARG1", "O", "O", "B-V", "B-ARG0", "I-ARG0", "I-ARG0", "I-ARG0", "O"]}], "words": ["and", "Eric", "Flint", ",", "was", "preceded", "by", "Grantville", "Gazette", "III", "."]}]

    for item in tree_srl:
        print (one_summary(item))

    '''
    if sys.argv[1].startswith('test'):
        src_file="[DATA]-webnlg-src-unique.txt".replace("[DATA]", sys.argv[1])
        tgt_file=("[DATA]-reference"+sys.argv[2]+".lex").replace("[DATA]", sys.argv[1])
        file_tag = sys.argv[1] + '_' + sys.argv[2]
    else:
        src_file="[DATA]-webnlg-src.txt".replace("[DATA]", sys.argv[1])
        tgt_file="[DATA]-webnlg-tgt.txt".replace("[DATA]", sys.argv[1])
        file_tag = sys.argv[1]

    input_dir = "./data-srl/"
    src_dir = "./data/"
    output_dir = "./data-tree/"

    thread_num = 30
    pool = multiprocessing.Pool(processes=thread_num)

    input_src = [line.strip() for line in open(src_dir+src_file)]
    input_tgt = []
    for line in open(input_dir+tgt_file):
        line = line.strip()
        input_tgt.append(json.loads(line))

    batch = []
    fpout = open(output_dir+"webnlg_"+file_tag+'.jsonl', 'w')
    for i, item in enumerate(input_tgt):
        batch.append((input_src[i], item))
        if i%30 == 0:
            res = pool.map(one_file, batch)
            for r in res:
                fpout.write(json.dumps(r)+'\n')
            del batch[:]
    if len(batch) != 0:
        res = pool.map(one_file, batch)
        for r in res:
            fpout.write(json.dumps(r)+'\n')
    fpout.close()
