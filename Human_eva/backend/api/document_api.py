import http
import json
from flask import jsonify, request

from . import api
from backend.models import Example, AnnotationResult


@api.route('/example/<ex_id>', methods=['GET'])
def api_document_get(doc_id):
    if request.method == 'GET':
        src_json = json.dumps(Example.get_dict(doc_id))
        if src_json:
            return jsonify(src_json), http.HTTPStatus.Ok


@api.route('/example/get_one', methods=['GET'])
def api_document_get_one():
    documents = Example.query.all()
    for document in documents:
        for doc_status in document.doc_statuses:
            n_results = len(AnnotationResult.query.filter_by(id=doc_status.id).all())
            if doc_status.total_exp_results == n_results:
                continue
            else:
                return jsonify(document.to_dict())
    return '', http.HTTPStatus.NO_CONTENT
