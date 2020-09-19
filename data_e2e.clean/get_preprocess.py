#coding=utf8

import sys, os
import json
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

with open('basic_verbs.txt') as f:
    line = f.read().strip()
    verb_dict = json.loads(line)

def wrong_split(lines):
    new_lines = []
    pre_no_verbs = []
    for line in lines:
        has_verb = -1
        for i, tok in enumerate(line.lower().split()):
            lemm_tok = wordnet_lemmatizer.lemmatize(tok, pos="v")
            if lemm_tok in verb_dict:
                has_verb = i
                if i == 0:
                    continue
                else:
                    break
        if has_verb < 1:
            if len(new_lines) == 0:
                pre_no_verbs.append(line)
            else:
                new_lines[-1] += (' ' + line)
        else:
            new_lines.append(line)
    if len(new_lines) == 0:
        return lines
    if len(pre_no_verbs) > 0:
        new_lines[0] = ' '.join(pre_no_verbs) + ' ' + new_lines[0]
    return new_lines

def split_and(line):
    line = line.strip()
    flist = line.split(' and ')
    if len(flist) == 1:
        return line

    chunk_list = []
    pre_no_verb = []
    for chunk in flist:
        has_verb = False
        for tok in chunk.split(' '):
            lemm_tok = wordnet_lemmatizer.lemmatize(tok, pos="v")
            if lemm_tok in verb_dict:
                has_verb = True
                break
        if has_verb:
            if len(chunk_list) > 0:
                #chunk = '<and> ' + chunk
                chunk = '<s> and ' + chunk
            chunk_list.append(chunk)
        else:
            if len(chunk_list) == 0:
                pre_no_verb.append(chunk)
            else:
                chunk_list[-1] += (' and ' + chunk)

    if len(chunk_list) == 0:
        return line
    if len(pre_no_verb) > 0:
        chunk_list[0] = ' and '.join(pre_no_verb) + ' and ' + chunk_list[0]

    return ' '.join(chunk_list)

def split_but(line):
    line = line.strip()
    flist = line.split(' but ')
    if len(flist) == 1:
        return line

    chunk_list = []
    pre_no_verb = []
    for chunk in flist:
        has_verb = False
        for tok in chunk.split(' '):
            lemm_tok = wordnet_lemmatizer.lemmatize(tok, pos="v")
            if lemm_tok in verb_dict:
                has_verb = True
                break
        if has_verb:
            if len(chunk_list) > 0:
                chunk = '<s> but ' + chunk
            chunk_list.append(chunk)
        else:
            if len(chunk_list) == 0:
                pre_no_verb.append(chunk)
            else:
                chunk_list[-1] += (' but ' + chunk)

    if len(chunk_list) == 0:
        return line
    if len(pre_no_verb) > 0:
        chunk_list[0] = ' but '.join(pre_no_verb) + ' but ' + chunk_list[0]

    return ' '.join(chunk_list)

def data_processing(line):
    flist = line.strip().split('<s>')
    lines = []
    for item in flist:
        item = item.strip()
        if len(item) == 0:
            continue
        lines.append(item)
    lines = wrong_split(lines)
    new_lines = [split_and(item) for item in lines]
    new_lines = [split_but(item) for item in new_lines]
    new_line = ' <s> '.join(new_lines) + ' <s>'
    new_line = new_line.replace('It s ', 'Its ')
    return new_line

if __name__ == '__main__':
    filename=sys.argv[1]
    fpout = open('./data-preprocess/' + filename, 'w')
    for line in open('./data/'+filename):
        new_line = data_processing(line.strip())
        fpout.write(new_line + '\n')
    fpout.close()

    #line = "It sells Chinese food and is children friendly ."
    #print (data_processing(line.strip()))
