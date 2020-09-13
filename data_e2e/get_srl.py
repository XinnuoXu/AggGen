#coding=utf8

import sys, os
import json
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

#archive = load_archive("bert-base-srl-2019.06.17.tar.gz", cuda_device=0)
archive = load_archive("srl-model-2018.05.25.tar.gz", cuda_device=0)
srl = Predictor.from_archive(archive)
print ("Loading Done")

def get_srl(inputs, output):
    fpout = open(output, "w")
    for sentences in inputs:
        sentences = [{"sentence": line} for line in sentences]
        srl_res = srl.predict_batch_json(sentences)
        fpout.write(json.dumps(srl_res) + '\n')
    fpout.close()

def data_processing(filename):
    lines = []
    for line in open('./data-preprocess/'+filename):
        line = ' '.join(line.strip().split(' ')[:-1])
        flist = line.split(' <s> ')
        lines.append(flist)
    get_srl(lines, "./data-srl/" + filename)

if __name__ == '__main__':
    data_processing(sys.argv[1])
    #line = "India is known for the river Ganges <s> and the largest city being Mumbai is also home to the state of Kerala which is positioned with Mahe to it 's northwest . <s> Kerala is also where AWH Engineering College is located within the city of Kuttikkattoor . <s> The college currently has 250 members of staff . <s>"
    #line = ' '.join(line.strip().split(' ')[:-1])
    #flist = line.split(' <s> ')
    #get_srl([flist], "tmp.txt")

