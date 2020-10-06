#coding=utf8

if __name__ == '__main__':
    import sys
    BASE_DIR = '../logs/'
    ref_dir = BASE_DIR + sys.argv[1]
    cand_dir = BASE_DIR + sys.argv[1]
    checkpoint = sys.argv[2]

    ref_path = ref_dir + '/abs_bert_cnndm.[CP].gold'.replace('[CP]', checkpoint)
    cand_path = cand_dir + '/abs_bert_cnndm.[CP].candidate'.replace('[CP]', checkpoint)

    ref_fpout = open("tmp/ref.txt", 'w')
    cand_fpout = open("tmp/cand.txt", 'w')

    cands = [line.strip() for line in open(cand_path)]
    for cand in cands:
        cand_fpout.write(cand.lower() + '\n')

    for line in open(ref_path):
        flist = line.strip().split('<ref_sep>')
        for ref in flist:
            ref_fpout.write(ref.strip().lower() + "\n")

    ref_fpout.close()
    cand_fpout.close()

    print ('run "java -Xmx2G -jar meteor-1.5.jar ../tmp/cand.txt ../tmp/ref.txt -nBest -l en -norm -r 3" in meteor-1.5/')
