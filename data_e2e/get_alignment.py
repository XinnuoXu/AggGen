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
    if float(matched_toks)/len(tri_tokens) >= THRESHOLD:
        return matched_toks
    return 0

def one_phrase_answer(answers, keys, phrase):
    alignment = []
    for i in range(len(answers)):
        percentage = match_answer(answers[i], phrase)
        alignment.append(percentage)
    max_value = max(alignment)
    max_index = alignment.index(max_value)
    if max_value > 0:
        return keys[max_index]
    else:
        return ""

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


def get_rel_and_answer(tripples):
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

def clean_overlap(alignments):
    overlaps = set()
    cnt = collections.Counter(alignments)
    for item in cnt:
        if cnt[item] > 1:
            overlaps.add(item)
    new_list = []
    for item in alignments:
        if item in overlaps:
            new_list.append('')
        else:
            new_list.append(item)
    return new_list

def final_touch(keys, re_alignments, an_alignments, terminals, tripples, non_terminals):
    # Filter confliction
    for i in range(len(re_alignments)):
        if re_alignments[i] == '':
            continue
        if re_alignments[i] not in an_alignments:
            continue
        idx = an_alignments.index(re_alignments[i])
        if idx != i:
            an_alignments[idx] = ''
            re_alignments[i] = ''

    # Merge re_alignments and an_alignments
    alignments = []
    for i in range(len(re_alignments)):
        re_a = re_alignments[i]
        an_a = an_alignments[i]
        if re_a == an_a:
            alignments.append(re_a)
        elif re_a == '':
            alignments.append(an_a)
        elif an_a == '':
            alignments.append(re_a)
        else:
            alignments.append(re_a+'&&'+an_a)

    # Automatically implement if only one slot left empty
    empty_slots = []
    for i, item in enumerate(alignments):
        if item == '':
            empty_slots.append(i)
    rest_key = []
    for k in keys:
        if k not in alignments:
            rest_key.append(k)
    if len(keys) == len(alignments) and len(empty_slots) == 1:
        filted_out = False
        for i in range(len(alignments)):
            if alignments[i].find('&&') > -1:
                filted_out = True
                alignments[i] = ''
        if not filted_out:
            rest_key = rest_key[0]
            for i in range(len(alignments)):
                if alignments[i] == '':
                    alignments[i] = rest_key

    # Only one fact in tgt
    if len(alignments) == 1 and alignments[0] == '':
        alignments[0] = '&&'.join(keys)

    return alignments

def one_pair(tripples, keys, terminals, non_terminals):
    answers, relations = get_rel_and_answer(tripples)
    re_alignments = []
    for phrase in terminals:
        alignment = one_phrase_relation(relations, keys, phrase)
        re_alignments.append(alignment)
    an_alignments = []
    for phrase in terminals:
        alignment = one_phrase_answer(answers, keys, phrase)
        an_alignments.append(alignment)
    an_alignments = clean_overlap(an_alignments)
    alignments = final_touch(keys, re_alignments, an_alignments, terminals, tripples, non_terminals)
    return '|'.join(alignments)

if __name__ == '__main__':
    input_dir = "./data-rst/"
    src_path = input_dir+"e2e_"+sys.argv[1]+'_src.jsonl'
    tgt_path = input_dir+"e2e_"+sys.argv[1]+'_tgt_clean.jsonl'
    src_list = [line.strip() for line in open(src_path)]
    tgt_list = [line.strip() for line in open(tgt_path)]

    output_dir = "./data-alg/"
    src_path = output_dir+"e2e_"+sys.argv[1]+'_src.jsonl'
    tgt_path = output_dir+"e2e_"+sys.argv[1]+'_tgt.jsonl'
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
