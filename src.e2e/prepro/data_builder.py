import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import copy
import torch
import subprocess
from collections import Counter
from os.path import join as pjoin
from prepro.data_pretrain import PreproPretrainJson, PreproPretrainData
from prepro.data_hmm import PreproHMMJson, PreproHMMData


def pretrain_to_json(args):
    obj = PreproPretrainJson(args)
    obj.preprocess()

def pretrain_to_data(args):
    obj = PreproPretrainData(args)
    obj.preprocess()

def hmm_to_json(args):
    obj = PreproHMMJson(args)
    obj.preprocess()

def hmm_to_data(args):
    obj = PreproHMMData(args)
    obj.preprocess()

