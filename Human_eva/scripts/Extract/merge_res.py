import json
import os, sys

def load_src(json_obj):
    merge_ann = {}
    for ex in json_obj:
        example_id = ex["example_id"]
        srcs = ex["srcs"]
        if example_id in merge_ann:
            continue
        merge_ann[example_id] = {}
        merge_ann[example_id]["srcs"] = srcs
    return merge_ann

def load_tgt(json_obj, merge_ann):
    for ex in json_obj:
        tgt_id = ex["tgt_id"]
        tgt_tag = "tgt_"+str(tgt_id)
        tgts = ex["tgts"]
        sanity_check = ex["sanity_check"]
        example_id = ex["example_id"]
        merge_ann[example_id][tgt_tag] = {}
        merge_ann[example_id][tgt_tag]["sanity_check"] = sanity_check
        merge_ann[example_id][tgt_tag]["tgts"] = tgts
        merge_ann[example_id][tgt_tag]["annotations"] = []
        merge_ann[example_id][tgt_tag]["mtruk_code"] = []
    return merge_ann

def load_annotation(json_obj, merge_ann):
    for ex in json_obj:
        example_id = ex["example_id"]
        tgt_id = ex["tgt_id"]
        tgt_tag = "tgt_"+str(tgt_id)
        annotation_res = ex["annotation_res"]
        is_closed = ex["is_closed"]
        is_filled = ex["is_filled"]
        mtruk_code = ex["mtruk_code"]
        if is_closed and is_filled:
            merge_ann[example_id][tgt_tag]["annotations"].append(annotation_res)
            merge_ann[example_id][tgt_tag]["mtruk_code"].append(mtruk_code)
    return merge_ann

if __name__ == '__main__':
    for i in range(3): 
        input_path = "../annotation_"+str(i)+".json"
        with open(input_path, 'r') as infile:
            json_obj = json.load(infile)
        if i == 0:
            merge_ann = load_src(json_obj)
            merge_ann = load_tgt(json_obj, merge_ann)
        merge_ann = load_annotation(json_obj, merge_ann)

    fpout = open("../annotation_all.json", "w")
    fpout.write(json.dumps(merge_ann))
    fpout.close()
