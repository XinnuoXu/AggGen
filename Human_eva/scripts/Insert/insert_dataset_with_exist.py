import json
import os
import sys
from os import listdir
from os.path import isfile, join

sys.path.append(os.path.abspath('../../'))
from backend.models import Example, Dataset
from backend.app import create_app
from flask_sqlalchemy import SQLAlchemy

def load_exist_annotation():
    ann_path = '../annotation.json' 
    exist_dict = {}
    with open(ann_path, 'r') as infile:
        objs = json.load(infile)
    for obj in objs:
        example_id = obj['example_id']
        tgt_id = obj['tgt_id']
        if example_id not in exist_dict:
            exist_dict[example_id] = []
        exist_dict[example_id].append(tgt_id)
    return exist_dict

def init_database(db, input_path, exist_annotation, db_id, batch_size):
    with open(input_path, 'r') as infile:
        json_obj = json.load(infile)

    db_name = "WebNLG_" + str(db_id)
    dataset = Dataset(name=db_name)
    db.session.add(dataset)
    db.session.commit()

    example_id = 300
    for obj in json_obj:
        for i in range(3):
            if obj['ID'] in exist_annotation \
                and i in exist_annotation[obj['ID']]:
                    continue
            if example_id > 300 and example_id % batch_size == 0:
                db_id += 1
                db_name = "WebNLG_" + str(db_id)
                dataset = Dataset(name=db_name)
                db.session.add(dataset)
                db.session.commit()
            example = Example(
                id=example_id,
                dataset_id=dataset.id,
                ex_id=obj['ID'],
                tgt_id=i,
                src_json=json.dumps(obj['SRC']),
                tgt_json=json.dumps(obj['TGT-'+str(i)]),
                sanity_check=json.dumps(obj['CHK-'+str(i)]) )
            db.session.add(example)
            db.session.commit()
            example_id += 1
    return example_id

if __name__ == '__main__':
    app = create_app()
    db_app = SQLAlchemy(app)

    start_db_id = 11
    input_dir = '../split_examples/'
    batch_size = 30
    exist_annotation = load_exist_annotation()
    init_database(db_app, './selected_example.json', exist_annotation, start_db_id, batch_size)
