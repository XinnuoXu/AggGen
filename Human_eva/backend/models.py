import json
from enum import Enum
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.ext.declarative import declared_attr
db = SQLAlchemy()

class ProjectType(Enum):
    ANNOTATION = 'Annotation'

class Example(db.Model):
    __tablename__ = 'example'

    id = db.Column(db.INTEGER, primary_key=True, nullable=False)
    ex_id = db.Column(db.INTEGER, nullable=False)
    tgt_id = db.Column(db.INTEGER, nullable=False)
    src_json = db.Column(db.Text, nullable=False)
    sanity_check = db.Column(db.Text, nullable=False)
    ex_statuses = db.relationship('ExStatus', backref='example', lazy=True)
    dataset_id = db.Column(db.INTEGER, db.ForeignKey('dataset.id'), nullable=True)
    tgt_json = db.Column(db.Text, nullable=True)

    @classmethod
    def get_dict(cls, id):
        if not id:
            return None
        document = cls.query.get(id)
        return json.loads(document.src_json)

    @classmethod
    def add_results(cls, doc_id, results):
        document = cls.query.filter_by(id=doc_id).first()
        src_json = json.loads(document.src_json)
        src_json['results'] = results
        document.src_json = json.dumps(src_json)
        document.has_highlight = True
        db.session.commit()

    def to_dict(self):
        return self.src_json


class ExStatus(db.Model):
    __tablename__ = 'ex_status'

    id = db.Column(db.INTEGER, primary_key=True, nullable=False)
    total_exp_results = db.Column(db.Integer, nullable=False)
    is_closed = db.Column(db.Boolean, nullable=False, default=False)

    ex_id = db.Column(db.INTEGER, db.ForeignKey('example.id'), nullable=False)
    proj_id = db.Column(db.INTEGER, db.ForeignKey('annotation_project.id'), nullable=False)

    results = db.relationship('AnnotationResult', backref='ex_status', lazy=True)

    @classmethod
    def close(cls, id):
        ex_status = cls.query.filter_by(id=id).first()
        ex_status.is_closed = True
        db.session.commit()


class AnnotationResult(db.Model):
    __tablename__ = 'annotation_result'

    id = db.Column(db.INTEGER, primary_key=True, nullable=False)
    finished_at = db.Column(db.DateTime, default=datetime.utcnow)
    opened_at = db.Column(db.DateTime, default=datetime.utcnow)
    result_json = db.Column(db.Text, nullable=False)
    status_id = db.Column(db.INTEGER, db.ForeignKey('ex_status.id'), nullable=False)
    mturk_code = db.Column(db.String(255), nullable=True)
    is_filled = db.Column(db.Boolean, nullable=True)

    @classmethod
    def del_result(cls, result):
        db.session.delete(result)
        db.session.commit()

    @classmethod
    def create_empty_result(cls, status_id):
        import random
        random.seed(datetime.now())
        an_id = random.sample(range(1, 1000000000), 1)[0]
        result = AnnotationResult(
            id=an_id,
            result_json='',
            status_id=status_id,
            is_filled=False)
        db.session.add(result)
        db.session.commit()
        return result.id

    @classmethod
    def update_result(cls, **kwargs):
        if 'result_id' in kwargs:
            result_id = kwargs['result_id']
            result = AnnotationResult.query.get(result_id)
        else:
            import random
            random.seed(datetime.now())
            an_id = random.sample(range(1, 1000000000), 1)[0]
            result = AnnotationResult(
                id=an_id,
                result_json='',
                status_id=kwargs['status_id'],
                is_filled=False)
        result.finished_at = datetime.utcnow()
        result.status_id = kwargs['status_id']
        result.result_json = json.dumps(kwargs['result_json'])
        result.mturk_code = kwargs['mturk_code']
        result.is_filled = True
        db.session.commit()
        return result


class Dataset(db.Model):
    __tablename__ = 'dataset'
    id = db.Column(db.INTEGER, primary_key=True, nullable=False)
    name = db.Column(db.String(255), nullable=False)

    documents = db.relationship('Example', backref='dataset', lazy=True)
    annotation_projects = db.relationship('AnnotationProject', backref='dataset', lazy=True)

    def to_dict(self):
        return dict(name=self.name)


class BaseProject(object):
    id = db.Column(db.INTEGER, primary_key=True, nullable=False)
    name = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(25), nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    finished_at = db.Column(db.DateTime, nullable=True)

    is_active = db.Column(db.Boolean, nullable=False, default=True)

    @declared_attr
    def dataset_id(cls):
        return db.Column(db.INTEGER, db.ForeignKey('dataset.id'), nullable=False)

    @classmethod
    def deactivate(cls, id):
        project = cls.query.get(id)
        project.is_active = False
        db.session.commit()

    def to_dict(self):
        return dict(
            id=self.id, name=self.name, category=self.category,
            created_at=self.created_at, finished_at=self.finished_at,
            is_active=self.is_active
        )


class AnnotationProject(db.Model, BaseProject):
    __tablename__ = 'annotation_project'

    ex_statuses = db.relationship('ExStatus', backref='project', lazy=True)

    def get_dict(self):
        return dict(id=self.id, name=self.name, category=self.category,
                    created_at=self.created_at, finished_at=self.finished_at,
                    is_active=self.is_active)

    @classmethod
    def create_project(cls, **kwargs):
        dataset = Dataset.query.filter_by(name=kwargs['dataset_name']).first()
        # noinspection PyArgumentList
        project = AnnotationProject(name=kwargs['name'], category=kwargs['category'], dataset_id=dataset.id)
        db.session.add(project)
        db.session.commit()
        for document in dataset.documents:
            ex_status = ExStatus(
                proj_id=project.id,
                ex_id=document.id,
                total_exp_results=kwargs['total_exp_results'])
            db.session.add(ex_status)
            db.session.commit()
        return project


