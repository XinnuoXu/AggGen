#coding=utf8

import sys
import random

if __name__ == '__main__':
    input_path = sys.argv[1]
    with open(input_path) as f:
        line = f.read().strip()
        examples = line.split('\n\n\n')
        examples = random.sample(examples, 100)
        selected_dict = {}
        for example in examples:
            lines = example.split('\n')
            for line in lines:
                if not line.startswith('<<CAND>>'):
                    continue
                flist = line.split(': ')
                label = flist[0]
                sentence = ': '.join(flist[1:])
                if label not in selected_dict:
                    selected_dict[label] = []
                selected_dict[label].append(sentence)
    for label in selected_dict:
        print (label)
        for sent in selected_dict[label]:
            print (sent)
        print ()
