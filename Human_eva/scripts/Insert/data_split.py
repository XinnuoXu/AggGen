#coding=utf8

import os
import sys
import json
import random
from os import listdir
from os.path import isfile, join

def file_split(input_path, output_path, split_num):
    with open(input_path, 'r') as infile:
        full_obj = json.load(infile)
    batch_size = len(full_obj) / split_num
    objs = []; i = 0
    while i < len(full_obj):
        if i > 0 and i % batch_size == 0:
            file_id = int(i/batch_size)
            fpout = open(output_path+str(file_id)+'.json', 'w')
            fpout.write(json.dumps(objs))
            fpout.close()
            del objs[:]
        objs.append(full_obj[i])
        i += 1
    if len(objs) > 0:
        file_id = int(i/batch_size)
        fpout = open(output_path+str(file_id)+'.json', 'w')
        fpout.write(json.dumps(objs))
        fpout.close()

def exam_split(input_path, output_path):
    with open(input_path, 'r') as infile:
        full_obj = json.load(infile)
    source_ids = set([obj['ID'] for obj in full_obj])
    target_files = [join(output_path, f) for f in listdir(output_path)]
    
    tgt_ids = []
    for fpath in target_files:
        with open(fpath, 'r') as infile: 
            split_obj = json.load(infile)
            tgt_ids.extend([obj['ID'] for obj in split_obj])
    tgt_ids = set(tgt_ids)

    if len(source_ids) != len(tgt_ids):
        return False
    if len(source_ids-tgt_ids) > 0:
        return False
    if len(tgt_ids-source_ids) > 0:
        return False

    print (source_ids)
    print (tgt_ids)
    return True

if __name__ == '__main__':
    SOURCE_PATH = './selected_example.json'
    TARGET_PATH = '../split_examples/split_example_'
    TARGET_DIR = '../split_examples/'
    SPLIT_NUM = 10

    if sys.argv[1] == 'split':
        file_split(SOURCE_PATH, TARGET_PATH, SPLIT_NUM)

    elif sys.argv[1] == 'exam':
        print (exam_split(SOURCE_PATH, TARGET_DIR))
