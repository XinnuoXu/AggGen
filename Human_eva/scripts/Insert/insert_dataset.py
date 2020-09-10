import json
import os
import sys

sys.path.append(os.path.abspath('../../'))
from backend.models import Example, Dataset
from backend.app import create_app
from flask_sqlalchemy import SQLAlchemy

def init_database(db):
    # Insert dataset
    dataset = Dataset(name="WebNLG")
    db.session.add(dataset)
    db.session.commit()

    input_path = '../selected_example.json'
    with open(input_path, 'r') as infile:
         input_json = json.load(infile)

    idx = -1
    with open(input_path, 'r') as infile:
        json_obj = json.load(infile)
    for obj in json_obj:
        for i in range(3):
            idx += 1
            example = Example(
                id=idx,
                dataset_id=dataset.id,
                ex_id=obj['ID'],
                tgt_id=i,
                src_json=json.dumps(obj['SRC']),
                tgt_json=json.dumps(obj['TGT-'+str(i)]),
                sanity_check=json.dumps(obj['CHK-'+str(i)]) )
            db.session.add(example)
            db.session.commit()

if __name__ == '__main__':
    app = create_app()
    db_app = SQLAlchemy(app)
    init_database(db_app)
