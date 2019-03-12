# Expected: generate nbmodel.txt
import sys, os, re
import glob
import json
import collections
import string

stoplist = {'', '\n', 'chicago', 'is', 'of', 'the', 'i', 'have', 'at', 'in', 'my', 'a', 'the', 'was', 'when', 'for',
            'they',
            'had', 'there', 'to', 'and', 'it', 'be', 'would', 'this', 'you', 'hotel', 'on', 'were', 'that', 'staff',
            'from',
            'with', 'we', 'our', 'room', 'as', 'rooms', 'one', 'about', 'been', 'here', 'your', 'me', 'are', 'stay',
            'stayed',
            'an', 'us', 'location', 'did', 'service',
            'city','what','do','staying','experience','experiences'}
stoplist={'','\n'}

posdoc = 0
negdoc = 0
trudoc = 0
decdoc = 0
Tru_dic = {}
Dec_dic = {}
Pos_dic = {}
Neg_dic = {}
vocabulary = {}
Tru_total = 0
Dec_total = 0
Pos_total = 0
Neg_total = 0


def get_files(filelist, flag):
    # Return total number of tokens of POS/NEG and TRUE/DECEP
    global posdoc, negdoc, trudoc, decdoc, Tru_total, Pos_total, Neg_total, Dec_total
    global stoplist

    posnegmap = {1: Pos_total, 0: Neg_total}
    trudecmap = {1: Tru_total, 0: Dec_total}

    num_doc = len(filelist)
    if (flag == (1, 1)):
        posdoc += num_doc
        trudoc += num_doc
    elif flag == (1, 0):
        posdoc += num_doc
        decdoc += num_doc
    elif flag == (0, 1):
        negdoc += num_doc
        trudoc += num_doc
    else:
        negdoc += num_doc
        decdoc += num_doc
    pattern = r',|\.|/|;|\[|\]|<|>|\?|:|\{|\}|\~|!|@|#|\$|%|&|\(|\)|-|=|\_|\+|\\|"| '
    for file in filelist:
        f = open(file)
        alltext = f.read()
        token_list = re.split(pattern, alltext)
        for i ,ele in enumerate(token_list):
            token_list[i]=token_list[i].lower()
        token_list=list(set(token_list))
        for word in token_list:
            if(word.isdecimal()==True):
                continue
            # word = str.lower(word)
            if (word in stoplist):
                continue
            else:
                posnegmap[flag[0]] += 1
                trudecmap[flag[1]] += 1
                if (word in vocabulary):
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1

                if (flag[0] == 1):
                    Pos_total += 1
                    if (word in Pos_dic):
                        Pos_dic[word] += 1
                    else:
                        Pos_dic[word] = 1
                else:
                    Neg_total += 1
                    if (word in Neg_dic):
                        Neg_dic[word] += 1
                    else:
                        Neg_dic[word] = 1
                if (flag[1] == 1):
                    Tru_total += 1
                    if word in Tru_dic:
                        Tru_dic[word] += 1
                    else:
                        Tru_dic[word] = 1
                else:
                    Dec_total += 1
                    if word in Dec_dic:
                        Dec_dic[word] += 1
                    else:
                        Dec_dic[word] = 1
    return


def stat(train_class):
    pnflag = 0
    tdflag = 0

    for key in train_class:
        if (key.rfind("positive") != -1):
            pnflag = 1
        else:
            pnflag = 0
        if (key.rfind("truthful") != -1):
            tdflag = 1
        else:
            tdflag = 0
        print(len(train_class[key]))

        get_files(train_class[key], (pnflag, tdflag))

    print("END")

    # write into nbmodel.txt
    # optimization

    # keys = list(vocabulary.keys())
    # for keyvoca in keys:
    #     if vocabulary[keyvoca] >200 or vocabulary[keyvoca]<3:
    #         vocabulary.pop(keyvoca)
    #         if (keyvoca in Pos_dic):
    #             Pos_dic.pop(keyvoca)
    #         if (keyvoca in Neg_dic):
    #             Neg_dic.pop((keyvoca))
    #         if keyvoca in Tru_dic:
    #             Tru_dic.pop(keyvoca)
    #         if (keyvoca in Dec_dic):
    #             Dec_dic.pop(keyvoca)

    # End of optimization

    fw = open('nbmodel.txt', 'w')
    annotation = "First four lines are the number of positive doc,negtive doc, truthful doc and deceptive doc" \
                 "the following four lines are the total number of tokens for pos, neg ,truth and decep" \
                 "The last are four kinds of dict, for dict in pos neg truth and decep"
    fw.writelines(annotation + '\n')
    fw.writelines(str(posdoc) + '\n')
    fw.writelines(str(negdoc) + '\n')
    fw.writelines(str(trudoc) + '\n')
    fw.writelines(str(decdoc) + '\n')
    fw.writelines(str(Pos_total) + '\n')
    fw.writelines(str(Neg_total) + '\n')
    fw.writelines(str(Tru_total) + '\n')
    fw.writelines(str(Dec_total) + '\n')

    load_dict = json.dumps(Pos_dic)
    fw.writelines(load_dict + '\n')
    load_dict = json.dumps(Neg_dic)
    fw.writelines(load_dict + '\n')
    load_dict = json.dumps(Tru_dic)
    fw.writelines(load_dict + '\n')
    load_dict = json.dumps(Dec_dic)
    fw.writelines(load_dict + '\n')

    load_dict = json.dumps(vocabulary)
    fw.writelines(load_dict + '\n')

    fw.close()


if __name__ == "__main__":

    model_file = "nbmodel.txt"
    all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

    
    train_by_class = collections.defaultdict(list)

    for f in all_files:
        class1, class2, fold, fname = f.split('/')[-4:]
        train_by_class[class1 + class2].append(f)

    stat(train_by_class)


    











