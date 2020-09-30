import json
import os, sys

def load_annotation(json_obj, merge_ann):
    for ex in json_obj:
        example_id = ex["example_id"]
        srcs = ex["srcs"]
        if example_id not in merge_ann:
            merge_ann[example_id] = {}
            merge_ann[example_id]["srcs"] = srcs
        tgt_tag = "tgt_"+str(ex["tgt_id"])
        sanity_check = ex["sanity_check"]
        tgts = ex["tgts"]
        if tgt_tag not in merge_ann[example_id]:
            merge_ann[example_id][tgt_tag] = {}
            merge_ann[example_id][tgt_tag]["sanity_check"] = sanity_check
            merge_ann[example_id][tgt_tag]["tgts"] = tgts
            merge_ann[example_id][tgt_tag]["annotations"] = []
            merge_ann[example_id][tgt_tag]["mtruk_code"] = []
        for ann in ex["annotations"]:
            annotation_res = ann["annotation_res"]
            mtruk_code = ann["mtruk_code"]
            is_filled = ann["is_filled"]
            #is_closed = ann["is_closed"]
            #if is_closed and is_filled:
            merge_ann[example_id][tgt_tag]["annotations"].append(annotation_res)
            merge_ann[example_id][tgt_tag]["mtruk_code"].append(mtruk_code)
    return merge_ann

if __name__ == '__main__':
    merge_ann = {}
    for filename in os.listdir('../'):
        if not (filename.startswith('annotation') and filename.endswith('.json')):
            continue
        print (filename)
        input_path = "../"+filename
        with open(input_path, 'r') as infile:
            json_obj = json.load(infile)
        merge_ann = load_annotation(json_obj, merge_ann)
    fpout = open("../annotation.res", "w")
    fpout.write(json.dumps(merge_ann))
    fpout.close()
