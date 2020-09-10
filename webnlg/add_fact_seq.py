#coding=utf8
import sys

if __name__ == '__main__':
    input_dir = "./data-alg/"
    src_path = input_dir+"webnlg_"+sys.argv[1]+'_src.jsonl'
    tgt_path = input_dir+"webnlg_"+sys.argv[1]+'_tgt.jsonl'
    src_list = [line.strip() for line in open(src_path)]
    tgt_list = [line.strip() for line in open(tgt_path)]

    output_dir = "./data-fact_seq/"
    src_path = output_dir+"webnlg_"+sys.argv[1]+'_src.jsonl'
    tgt_path = output_dir+"webnlg_"+sys.argv[1]+'_tgt.jsonl'
    fpout_src = open(src_path, 'w')
    fpout_tgt = open(tgt_path, 'w')

    for i in range(len(src_list)):
        src = src_list[i]; tgt = tgt_list[i]
        # src data
        src_data = src.split('\t')
        tripples = src_data[:-1]
        keys = src_data[-1].split('|')
        # tgt data
        tgt_data = tgt.split('\t')
        non_terminals = tgt_data[0]
        alignments = tgt_data[1].split('|')
        terminals = tgt_data[2:]
        # processing
        if len(keys) > len(alignments):
            print (src)
            print (tgt)
            print ('\n')

        #fpout_src.write(src + '\n')
        #fpout_tgt.write(non_terminals + '\t' + alignments + '\t' + '\t'.join(terminals) + '\n')

    fpout_src.close()
    fpout_tgt.close()
