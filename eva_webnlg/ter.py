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
    for i, cand in enumerate(cands):
        cand_fpout.write(cand.lower() + '  (ID' + str(i) + ')\n')

    for i, line in enumerate(open(ref_path)):
        flist = line.strip().split('<ref_sep>')
        for ref in flist:
            if ref.strip() == "":
                continue
            ref_fpout.write(ref.strip().lower() + ' (ID' + str(i) + ")\n")

    ref_fpout.close()
    cand_fpout.close()

    print ('run "java -jar tercom.7.25.jar -h ../tmp/cand.txt -r ../tmp/ref.txt" in tercom-0.7.25/')
