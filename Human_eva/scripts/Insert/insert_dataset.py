import json
import os
import sys
from os import listdir
from os.path import isfile, join

sys.path.append(os.path.abspath('../../'))
from backend.models import Example, Dataset
from backend.app import create_app
from flask_sqlalchemy import SQLAlchemy

def init_database(db, input_path, db_id, example_id):
    # Insert dataset
    db_name = "WebNLG_" + str(db_id)
    dataset = Dataset(name=db_name)
    db.session.add(dataset)
    db.session.commit()

    with open(input_path, 'r') as infile:
        json_obj = json.load(infile)

    for obj in json_obj:
        for i in range(3):
            example_id += 1
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
    return example_id

if __name__ == '__main__':
    app = create_app()
    db_app = SQLAlchemy(app)

    input_dir = '../split_examples/'
    example_id = -1
    for i, f in enumerate(listdir(input_dir)):
        example_id = init_database(db_app, join(input_dir, f), i+1, example_id)
