import pandas as pd
import json

from backend.models import AnnotationResult
from backend.app import create_app
from flask_sqlalchemy import SQLAlchemy

if __name__ == '__main__':
    app = create_app()
    db = SQLAlchemy(app)

    results = db.session.query(AnnotationResult).all()
    for result in results:
        if result.result_json:
            result_json = json.loads(result.result_json)
        else:
            db.session.delete(result)
            db.session.commit()
