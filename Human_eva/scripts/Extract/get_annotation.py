import json
import math
import os, sys
from collections import Counter
from itertools import chain
from flask_sqlalchemy import SQLAlchemy
sys.path.append(os.path.abspath('../../'))
from backend.models import Dataset, Example, ExStatus, AnnotationResult
from backend.app import create_app

app = create_app()
db = SQLAlchemy(app)

def extract_src(src_json):
    srcs = {}
    for key in src_json:
        if key == 'results':
            continue
        srcs[key] = src_json[key]
    return srcs

# Read example table
q_results = db.session.query(Example, Dataset).join(Dataset).all()
record_info = {}
for ex, _ in q_results:
    example_id = ex.ex_id
    sanity_check = ex.sanity_check
    tgt_id = ex.tgt_id
    record_id = ex.id
    tgts = ex.tgt_json
    srcs = extract_src(json.loads(ex.src_json))
    record = {"example_id":example_id,
              "sanity_check":sanity_check,
              "tgt_id":tgt_id,
              "tgts":tgts,
              "srcs":srcs}
    record_info[record_id] = record

# Read Status
ex_status = db.session.query(ExStatus)
for es in ex_status:
    record_id = es.ex_id
    result_id = es.id
    is_closed = es.is_closed
    if 'annotation_info' not in record_info[record_id]:
        record_info[record_id]['annotation_info'] = []
    record_info[record_id]['annotation_info'].append({'result_id':result_id, 'is_closed':is_closed})

def extract_annotation(result_json):
    annotation = []
    for seg in result_json:
        ann = '&&'.join(seg["Selection"])
        annotation.append(ann)
    return '|'.join(annotation)

# Read Annotation
annotations = {}
annotation_res = db.session.query(AnnotationResult)
for ann in annotation_res:
    try:
        result_json = json.loads(ann.result_json)
    except:
        print (ann.status_id)
    result = extract_annotation(result_json)
    result_id = ann.status_id
    mtruk_code = ann.mturk_code
    is_filled = ann.is_filled
    annotations[result_id] = {"annotation_res":result,
                              "mtruk_code":mtruk_code,
                              "is_filled":is_filled}

# Merge input and annoataion
outputs = []
for record_id in record_info:
    record = record_info[record_id]
    if 'annotation_info' not in record:
        continue
    record['annotations'] = []
    for ann_info in record['annotation_info']:
        result_id = ann_info['result_id']
        if result_id not in annotations:
            continue
        ann = annotations[result_id]
        record['annotations'].append(ann)
    if len(record['annotations']) > 0:
        outputs.append(record)


# Write out
fpout = open("../annotation.json", "w")
fpout.write(json.dumps(outputs))
fpout.close()
