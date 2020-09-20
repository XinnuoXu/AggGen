import json
import copy
import http
import urllib.parse
import random
import string
from datetime import datetime, timedelta

from flask import jsonify, request

from . import api
from backend.models import Example, AnnotationProject, AnnotationResult, Dataset, ExStatus, ProjectType


def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))


@api.route('/result/<project_type>/<status_id>', methods=['GET'])
def api_get_result_id(project_type, status_id):
    if project_type.lower() == ProjectType.ANNOTATION.value.lower():
        result_id = AnnotationResult.create_empty_result(status_id)
    return jsonify(dict(result_id=result_id))


@api.route('/project/<project_type>/<project_category>/<project_id>/single_doc', methods=['GET'])
def api_project_single_doc(project_type, project_category, project_id):
    random.seed(datetime.now())
    if project_type.lower() == ProjectType.ANNOTATION.value.lower():
        project = AnnotationProject.query.get(project_id)
        if not project:
            return '', http.HTTPStatus.NOT_FOUND
        else:
            # Clean open project, very dirty
            for ex_status in project.ex_statuses:
                print ('>>>',ex_status.id)
                results = AnnotationResult.query.filter_by(status_id=ex_status.id, is_filled=False).all()
                
                for result in results:
                    if result.opened_at:
                        delta = datetime.utcnow() - result.opened_at
                        if delta >= timedelta(minutes=15):
                            AnnotationResult.del_result(result)
            # Retrieve result
            random_ex_statuses = list(project.ex_statuses)
            random.shuffle(random_ex_statuses)
            min_result = 999
            n_results_list = []
            for ex_status in random_ex_statuses:
                n_results = AnnotationResult.query.filter_by(status_id=ex_status.id).count()
                if n_results < min_result:
                    min_result = n_results
                n_results_list.append(n_results)
            for idx, ex_status in enumerate(random_ex_statuses):
                if ex_status.total_exp_results > n_results_list[idx] == min_result:
                    document = Example.query.filter_by(id=ex_status.ex_id).first()
                    turk_code = '%s_%s_%s' % (ex_status.ex_id, randomword(5), project.id)

                    paired_data = []
                    tgt_obj = json.loads(document.tgt_json)

                    check_id = 0
                    for i, tgt in enumerate(tgt_obj):
                        pre = ' '.join(tgt_obj[:i])
                        curr = '<span class=\"highlight\">'+ tgt + '</span>'
                        tmp_str = pre + ' ' + curr
                        if i < len(tgt_obj) - 1:
                            post = ' '.join(tgt_obj[i+1:])
                            tmp_str += ' ' + post
                        tmp_str = "<span>&#127807; <b>Highlight Section "+str(i+1)+": </b></span><br>" + tmp_str
                        tmp_data = {'Sentence':tmp_str}

                        src_obj = json.loads(document.src_json)
                        src_list = []
                        for key in src_obj:
                            rcd = src_obj[key].replace(key, ' ['+key+'] ')
                            src_list.append({'Relation':key, 'Record':rcd, 'ID':str(check_id)})
                            check_id += 1
                        tmp_data['Input'] = src_list
                        tmp_data['Selection'] = []
                        paired_data.append(tmp_data)

                    tmp_data = {'Sentence':"&#127800; <b>Now tick triples that are <mark>NOT</mark> mentioned in highlight sections above: </b>"}
                    src_obj = json.loads(document.src_json)
                    src_list = []
                    for key in src_obj:
                        rcd = src_obj[key].replace(key, ' ['+key+'] ')
                        src_list.append({'Relation':key, 'Record':rcd, 'ID':str(check_id)})
                        check_id += 1
                    tmp_data['Input'] = src_list
                    tmp_data['Selection'] = []
                    paired_data.append(tmp_data)
  
                    paired_data = json.dumps(paired_data)

                    ex_id = ex_status.ex_id
                    return jsonify(dict(paired_data=paired_data,
                                        ex_id=ex_id,
                                        ex_status_id=ex_status.id,
                                        turk_code=turk_code,
                                        sanity_check=document.sanity_check,
                                        ))
            return '', http.HTTPStatus.NOT_FOUND
    else:
        return '', http.HTTPStatus.BAD_REQUEST


@api.route('/project/<project_type>', methods=['POST'])
def api_project_create(project_type):
    if request.method == 'POST':
        data = request.get_json()
        project = None
        if project_type.lower() == ProjectType.ANNOTATION.value.lower():
            project = AnnotationProject.create_project(**data)
        else:
            return '', http.HTTPStatus.BAD_REQUEST
        if project:
            return '', http.HTTPStatus.CREATED
        else:
            return '', http.HTTPStatus.CONFLICT


@api.route('/project/get/<project_type>/<project_name>', methods=['GET'])
def api_project_get(project_type, project_name):
    if project_type.lower() == ProjectType.ANNOTATION.value.lower():
        projects = AnnotationProject.query.filter_by(name=project_name).all()
    else:
        return '', http.HTTPStatus.BAD_REQUEST
    if len(projects) == 0:
        return '', http.HTTPStatus.NOT_FOUND
    else:
        result_json = {}
        for project in projects:
            result_json[project.id] = project.get_dict()
        return jsonify(result_json)


@api.route('/project/save_result/<project_type>', methods=['POST'])
def api_project_save_result(project_type):
    data = request.get_json()
    if project_type.lower() == ProjectType.ANNOTATION.value.lower():
        result = AnnotationResult.update_result(**data)
    if result:
        return '', http.HTTPStatus.CREATED
    else:
        return '', http.HTTPStatus.CONFLICT


@api.route('/project/<project_id>/close', methods=['POST'])
def api_project_close(project_id):
    project = AnnotationProject.query.filter_by(id=project_id).first()
    if not project or project.is_active is False:
        return '', http.HTTPStatus.NOT_MODIFIED
    else:
        for ex_status in project.ex_statuses:
            results = AnnotationResult.query.filter_by(status_id=ex_status.id).all()
            results_json = {}
            for result in results:
                try:
                    res_json = json.loads(result.result_json)
                except:
                    continue
                results_json[result.id] = res_json
            if len(results_json) != 0:
                Example.add_results(ex_status.ex_id, results_json)
                ExStatus.close(ex_status.id)
        AnnotationProject.deactivate(project_id)
        return '', http.HTTPStatus.OK


@api.route('/project/all_progress/<project_type>', methods=['GET'])
def api_project_progress_all(project_type):
    project_type = project_type.lower()
    if project_type == ProjectType.ANNOTATION.value.lower():
        projects = AnnotationProject.query.filter_by(is_active=True).all()
    else:
        return '', http.HTTPStatus.BAD_REQUEST

    if len(projects) == 0:
        return '', http.HTTPStatus.NOT_FOUND
    else:
        result_json = {'projects': []}
        for project in projects:
            project_json = project.to_dict()
            project_json['dataset_name'] = Dataset.query.filter_by(id=project.dataset_id).first().name
            total_n_results = 0
            total_total_exp_results = 0
            if project_type == ProjectType.ANNOTATION.value.lower():
                for ex_status in project.ex_statuses:
                    n_results = AnnotationResult.query.filter_by(status_id=ex_status.id).count()
                    total_n_results += n_results
                    total_total_exp_results += ex_status.total_exp_results
            project_json['progress'] = total_n_results/total_total_exp_results
            project_json['no'] = len(result_json['projects']) + 1

            category = project.category.lower()
            project_json['link'] = urllib.parse.urljoin(
            request.host_url,
            '#/{type}/{category}/{id}/1'.format(
                type=project_type,
                category=category,
                id=project_json['id']
            ))
            result_json['projects'].append(project_json)
        return jsonify(result_json)


@api.route('/project/progress/<project_type>/<project_id>', methods=['GET'])
def api_project_progress(project_type, project_id):
    project_type = project_type.lower()
    if project_type == ProjectType.ANNOTATION.value.lower():
        project = AnnotationProject.query.get(project_id)
    else:
        return '', http.HTTPStatus.BAD_REQUEST
    if not project:
        return '', http.HTTPStatus.CONFLICT
    else:
        progress_json = None
        if project_type == ProjectType.ANNOTATION.value.lower():
            progress_json = {
                'documents': [],
                'name': ''
            }
        if project_type == ProjectType.ANNOTATION.value.lower():
            for ex_status in project.ex_statuses:
                document = Example.query.get(ex_status.doc_id)
                result_jsons = []
                for result in ex_status.results:
                    result_jsons.append(result.result_json)
                exp_results = ex_status.total_exp_results
                progress_json['documents'].append({
                    'no': len(progress_json['documents']) + 1,
                    'name': document.doc_id,
                    'progress': len(ex_status.results)/exp_results,
                    'result_jsons': result_jsons
                })
            progress_json['name'] = project.name
            return jsonify(progress_json)
        else:
            return '', http.HTTPStatus.BAD_REQUEST


@api.route('/doc_status/progress/<ex_status_id>', methods=['GET'])
def api_doc_status_progress(ex_status_id):
    doc_status = ExStatus.query.filter_by(id=ex_status_id).first()
    if not doc_status:
        return '', http.HTTPStatus.NOT_FOUND
    n_results = len(AnnotationResult.query.filter_by(id=doc_status.id).all())
    progress = "{0:.2f}".format(n_results/doc_status.total_exp_results)
    return jsonify(dict(progress=progress))
