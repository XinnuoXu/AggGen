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

# Read example table
q_results = db.session.query(Example, Dataset).join(Dataset).all()
record_info = {}
for ex, _ in q_results:
    example_id = ex.ex_id
    sanity_check = ex.sanity_check
    tgt_id = ex.tgt_id
    record_id = ex.id
    record = {"example_id":example_id,
              "sanity_check":sanity_check,
              "tgt_id":tgt_id}
    record_info[record_id] = record

# Read Status
ex_status = db.session.query(ExStatus)
for es in ex_status:
    record_id = es.ex_id
    result_id = es.id
    is_closed = es.is_closed
    record_info[record_id]['result_id'] = result_id
    record_info[record_id]['is_closed'] = is_closed

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
    result_json = json.loads(ann.result_json)
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
    inp = record_info[record_id]
    result_id = inp['result_id']
    if result_id not in annotations:
        continue
    ann = annotations[result_id]
    for key in ann:
        inp[key] = ann[key]
    outputs.append(inp)


# Write out
fpout = open("../annotation.json", "w")
fpout.write(json.dumps(outputs))
fpout.close()
