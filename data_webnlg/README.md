# Convert WebNLG dataset to Flat re-ified representations

## Setup

- Install dependencies:

``
pip install spacy, networkx
``
- Download dataset. We just need the 3 dataset folders (train/dev/test).
    - [webnlg_challenge_2017](https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/webnlg_challenge_2017)

## How to Use
Run conversion code `converter.py` using the provided script with only one argument which contains the dataset 
(don't forget the trailing '/' in the end): 
 
```
convert_webnlg_to_factsum.sh data/webnlg_challenge_2017/
```
This should create all necessary files in the dataset folder. The most important ones to use for **training** are:

- `{dev/train}-webnlg-src.txt`: contains the input graphs re-ified and flattened. Re-ified RDF triples are delimited 
with the special `<TSP>` token.  

    For example: `John|ARG0|like Smith|ARG0|like like|V|like banana|ARG0|hasColour hasColour|V|hasColour yellow|ARG1|hasColour`   

- `{dev/train}-webnlg-tgt.txt`: contains the output text sentence delimited with `<s>`.

In order to do **inference** we are probably going to use these:

- `{dev/train/test-seen/unseen}-webnlg-src-unique.txt`: contains single source graphs (in contrast to the files above 
used in training, which contain repeated source inputs for several target lexicalisations).

- `{dev/train/test-seen/unseen}-webnlg-reference{0,1,2}.lex`: multiple references aligned to the source graphs. 
These can be used directly with `multi-bleu.perl` against our predictions as follows:

    ``
    multi-bleu.perl dev-webnlg-reference1.lex dev-webnlg-reference2.lex < predictions.txt
    ``
## How to process data for training
```
mkdir data-srl
mkdir data-tree
mkdir data-struct

sh data_process.sh
```
