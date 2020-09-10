#coding=utf8

train_src = './data/train-webnlg-src.txt'
train_tgt = './data/train-webnlg-tgt.txt'
dev_src = './data/dev-webnlg-src.txt'
dev_tgt = './data/dev-webnlg-tgt.txt'

NUM_FOR_FILE = 2000

def one_pair(src_path, tgt_path, tag):
    src = [line.strip() for line in open(src_path)]
    tgt = [line.strip() for line in open(tgt_path)]
    fpout_s = open('tmp/' + tag + '_src.0', 'w')
    fpout_t = open('tmp/' + tag + '_tgt.0', 'w')
    for i, s in enumerate(src):
        t = tgt[i]
        if i>0 and i%NUM_FOR_FILE==0:
            fpout_s.close(); fpout_t.close()
            fpout_s = open('tmp/' + tag + '_src.' + str(int(i/NUM_FOR_FILE)), 'w')
            fpout_t = open('tmp/' + tag + '_tgt.' + str(int(i/NUM_FOR_FILE)), 'w')
        fpout_s.write(s + '\n')
        fpout_t.write(t + '\n')
    fpout_s.close(); fpout_t.close()

if __name__ == '__main__':
    one_pair(train_src, train_tgt, 'train')
    one_pair(dev_src, dev_tgt, 'dev')
