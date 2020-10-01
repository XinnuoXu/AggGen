#coding=utf8

import json
import sklearn
from sklearn import metrics

def annotation_mapping(relations, ann):
    ann_split = ann.split('|')
    map_dict = {}
    for i, ann in enumerate(ann_split):
        for item in ann.split('&'):
            if item == '':
                continue
            map_dict[item] = i
    map_res = []
    for item in relations:
        if item not in map_dict:
            map_res.append(-1)
        else:
            map_res.append(map_dict[item])
    return map_res

if __name__ == '__main__':
    with open('../annotation.res', 'r') as infile:
        json_obj = json.load(infile)
    fpout_src = open('../webnlg_ann_src.jsonl', 'w')
    fpout_tgt = open('../webnlg_ann_tgt.jsonl', 'w')
    kp_scores = 0.0
    qualified_refs = 0
    for example_id in json_obj:
        example = json_obj[example_id]
        srcs = example['srcs']
        src_keys = srcs.keys()
        src_values = srcs.values()
        src_str = '\t'.join(src_values) + '\t' + '|'.join(src_keys)
        relations = srcs.keys()
        references = []
        for tgt_tag in example:
            if not tgt_tag.startswith('tgt_'):
                continue
            one_ref = example[tgt_tag]
            tgts = ' '.join(json.loads(one_ref['tgts']))
            annotation = one_ref['annotations']
            if len(annotation) < 3:
                continue
            annotation_split = []
            for ann in annotation:
                ann_split = annotation_mapping(relations, ann)
                annotation_split.append(ann_split)
            qualified = True
            for item in annotation_split:
                if len(item) == 0:
                    qualified = False
                    break
            if not qualified:
                continue

            correct_annotation = ""
            kp_score = metrics.cohen_kappa_score(annotation_split[0], annotation_split[1])
            if kp_score == 1:
                correct_annotation = annotation[0]

            kp_score = metrics.cohen_kappa_score(annotation_split[0], annotation_split[2])
            if kp_score == 1:
                correct_annotation = annotation[0]

            kp_score = metrics.cohen_kappa_score(annotation_split[1], annotation_split[2])
            if kp_score == 1:
                correct_annotation = annotation[1]
            
            if correct_annotation == "":
                continue

            tgt_str = '(S)[XXN]'+correct_annotation+'[XXN]'+tgts
            references.append(tgt_str)
        fpout_tgt.write('\t'.join(references) + '\n')
        fpout_src.write(src_str + '\n')

    fpout_src.close()
    fpout_tgt.close()
