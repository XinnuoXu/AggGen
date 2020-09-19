import sys, os
import json

verb_dict = {}

def one_summary(obj):
    if ("words" not in obj) or ("verbs" not in obj):
        return 
    for item in obj["verbs"]:
        v = item["verb"]
        if v not in verb_dict:
            verb_dict[v] = len(verb_dict)

if __name__ == '__main__':
    filename="train-e2e-tgt.txt"
    input_dir = "./data-srl/"
    tgts = [json.loads(line.strip()) for line in open(input_dir+filename)]
    for tgt in tgts:
        for item in tgt:
            one_summary(item)
    fpout = open('basic_verbs.txt', 'w')
    fpout.write(json.dumps(verb_dict))
    fpout.close()
