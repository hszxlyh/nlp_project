import sys,os,re
import glob
import math
import json
import collections,string
import pickle
import numpy as np


stoplist={'','\n','chicago','is','of','the','i','have','at','in','my','a','the','was','when','for','they',
          'had','there','to','and','it','be','would','this','you','hotel','on','were','that','staff','from',
          'with','we','our','room','as','rooms','one','about','been','here','your','me','are','stay','stayed',
          'an','us','location','did','service'}

         # 'by','bed','view','people','husband','son','wife','daughter','sister','brother'}



word_index={}
vocabulary={}
voclength=0
pos_matrix=[0,0,0,0]
tru_matrix=[0,0,0,0]

def getscore():
    prec = pos_matrix[0] / (pos_matrix[0] + pos_matrix[1])
    recall = pos_matrix[0] / (pos_matrix[0] + pos_matrix[2])
    posF1 = 2 * prec * recall / (prec + recall)

    prec = pos_matrix[3] / (pos_matrix[3] + pos_matrix[2])
    recall = pos_matrix[3] / (pos_matrix[3] + pos_matrix[1])
    negF1 = 2 * prec * recall / (prec + recall)

    prec = tru_matrix[0] / (tru_matrix[0] + tru_matrix[1])
    recall = tru_matrix[0] / (tru_matrix[0] + tru_matrix[2])
    truF1 = 2 * prec * recall / (prec + recall)

    prec = tru_matrix[3] / (tru_matrix[3] + tru_matrix[2])
    recall = tru_matrix[3] / (tru_matrix[3] + tru_matrix[1])
    decF1 = 2 * prec * recall / (prec + recall)

    print("positive f1 score", posF1)
    print("negative f1 score", negF1)
    print("truthful f1 score", truF1)
    print("deceptive f1 score", decF1)
    print("mean f1 score", (posF1 + negF1 + truF1 + decF1) / 4)


def getmodel(file):
    global vocabulary,voclength
    global word_index
    f=open(file,'rb')
    vocabulary=pickle.load(f)
    voclength=len(vocabulary)
    # if('hilton' in vocabulary):
    #     print("hehe")
    word_index=pickle.load(f)

    w_posneg=pickle.load(f)
    b_posneg=pickle.load(f)
    w_trudec=pickle.load(f)
    b_trudec=pickle.load(f)
    return w_posneg,b_posneg,w_trudec,b_trudec

def vanilla_predict(w_posneg,b_posneg,w_trudec,b_trudec,test_class):
    global word_index

    fout=open('percepoutput.txt','w')
    mapdic1 = {1: 'truthful', -1: 'deceptive'}
    mapdic2 = {1: 'positive', -1: 'negative'}
    pattern = '\W+'

    pnflag=0
    tdflag=0
    predict_pn=0
    predict_td=0
    for key in test_class:
        if (key.rfind("positive") != -1):
            pnflag=1
        else:
            pnflag=-1
        if (key.rfind("truthful") != -1):
            tdflag=1
        else:
            tdflag=-1
        for pick_onefile in test_class[key]:
            single_list = np.zeros(voclength)
            f = open(pick_onefile)
            alltext = f.read()
            token_list = re.split(pattern, alltext)

            for word in token_list:
                if (word.isdecimal() == True):
                    continue
                word = str.lower(word)
                if (word in stoplist):
                    continue
                else:
                    if (word in vocabulary):
                        single_list[word_index[word]] += 1
            if(np.dot(single_list,w_posneg)+b_posneg >0):
                predict_pn=1
            else:
                predict_pn=-1
            if(np.dot(single_list,w_trudec)+b_trudec>0):
                predict_td=1
            else:
                predict_td=-1

            fout.writelines(mapdic1[predict_td] + ' ' + mapdic2[predict_pn] + ' ' + pick_onefile+ '\n')   
            #delete after submit-----
            if(pnflag==1):
                if(predict_pn==1):
                    pos_matrix[0]+=1
                else:
                    pos_matrix[2]+=1
            else:
                if(predict_pn==1):
                    pos_matrix[1]+=1
                else:
                    pos_matrix[3]+=1

            if (tdflag == 1):
                if (predict_td == 1):
                    tru_matrix[0] += 1
                else:
                    tru_matrix[2] += 1
            else:
                if (predict_td == 1):
                    tru_matrix[1] += 1
                else:
                    tru_matrix[3] += 1

def average_predict(w_posneg,b_posneg,w_trudec,b_trudec ,test_class):
    global vocabulary, voclength
    global word_index
    global pos_matrix,tru_matrix
    pos_matrix=[0,0,0,0]
    tru_matrix=[0,0,0,0]
    #
    # f = open('averagemodel.txt', 'rb')
    # w_posneg = pickle.load(f)
    # b_posneg = pickle.load(f)
    # w_trudec = pickle.load(f)
    # b_trudec = pickle.load(f)


    pattern = '\W+'

    pnflag = 0
    tdflag = 0
    predict_pn = 0
    predict_td = 0

    fout=open('percepoutput.txt','w')
    mapdic1 = {1: 'truthful', -1: 'deceptive'}
    mapdic2 = {1: 'positive', -1: 'negative'}
    for key in test_class:
        if (key.rfind("positive") != -1):
            pnflag = 1
        else:
            pnflag = -1
        if (key.rfind("truthful") != -1):
            tdflag = 1
        else:
            tdflag = -1
        for pick_onefile in test_class[key]:
            single_list = np.zeros(voclength)

            f = open(pick_onefile)
            alltext = f.read()
            token_list = re.split(pattern, alltext)

            for word in token_list:
                if (word.isdecimal() == True):
                    continue
                word = str.lower(word)
                if (word in stoplist):
                    continue
                else:
                    if (word in vocabulary):
                        single_list[word_index[word]] += 1

            if (np.dot(single_list, w_posneg) + b_posneg > 0):
                predict_pn = 1
            else:
                predict_pn = -1

            if (np.dot(single_list, w_trudec) + b_trudec > 0):
                predict_td = 1
            else:
                predict_td = -1

            fout.writelines(mapdic1[predict_td] + ' ' + mapdic2[predict_pn] + ' ' + pick_onefile+ '\n')
            # delete after submit-----
            if (pnflag == 1):
                if (predict_pn == 1):
                    pos_matrix[0] += 1
                else:
                    pos_matrix[2] += 1
            else:
                if (predict_pn == 1):
                    pos_matrix[1] += 1
                else:
                    pos_matrix[3] += 1

            if (tdflag == 1):
                if (predict_td == 1):
                    tru_matrix[0] += 1
                else:
                    tru_matrix[2] += 1
            else:
                if (predict_td == 1):
                    tru_matrix[1] += 1
                else:
                    tru_matrix[3] += 1




if __name__=="__main__":

    model_file = str(sys.argv[1])
    output_file = "percepoutput.txt"
    input_path = str(sys.argv[2])
    all_files = glob.glob(os.path.join(input_path, '*/*/*/*.txt'))

    test_by_class = collections.defaultdict(list)
    train_by_class = collections.defaultdict(list)

    for f in all_files:

        class1, class2, fold, fname = f.split('/')[-4:]
        test_by_class[class1 + class2].append(f)


    if(model_file.rfind("vanillamodel.txt") !=-1):
        w1,b1,w2,b2=getmodel(model_file)
        vanilla_predict(w1,b1,w2,b2,test_by_class)
    elif(model_file.rfind("averagemodel.txt")!=-1):
        w1,b1,w2,b2=getmodel(model_file)
        average_predict(w1,b1,w2,b2,test_by_class)
    else:
        print("ERROR!!!!")
        
            
            


