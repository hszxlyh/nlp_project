import sys,os,re
import glob
import math
import json
import collections,string

posdoc=0
negdoc=0
trudoc=0
decdoc=0
Tru_total=0
Dec_total=0
Pos_total=0
Neg_total=0
Tru_dic={}
Dec_dic={}
Pos_dic={}
Neg_dic={}

vocabulary={}
voc_length=0
#-----------
stoplist = {'', '\n', 'chicago', 'is', 'of', 'the', 'i', 'have', 'at', 'in', 'my', 'a', 'the', 'was', 'when', 'for',
            'they',
            'had', 'there', 'to', 'and', 'it', 'be', 'would', 'this', 'you', 'hotel', 'on', 'were', 'that', 'staff',
            'from',
            'with', 'we', 'our', 'room', 'as', 'rooms', 'one', 'about', 'been', 'here', 'your', 'me', 'are', 'stay',
            'stayed',
            'an', 'us', 'location', 'did', 'service',
            'city','what','do','staying','experience','experiences'}
#stoplist={'','\n'}



Pos_dev={}
Neg_dev={}
Tru_dev={}
Dec_dev={}
PNvoca=set()
TDvoca=set()


def getmodel():
    global voc_length,vocabulary
    global posdoc,negdoc,trudoc,decdoc
    global Pos_dic,Neg_dic,Tru_dic,Dec_dic

    f=open('nbmodel.txt')
    f.readline()#annotation
    posdoc=int(f.readline().strip('\n'))
    negdoc=int(f.readline().strip('\n'))
    trudoc=int(f.readline().strip('\n'))
    decdoc=int(f.readline().strip('\n'))

    Pos_total=int(f.readline().strip('\n'))
    Neg_total=int(f.readline().strip('\n'))
    Tru_total=int(f.readline().strip('\n'))
    Dec_total=int(f.readline().strip('\n'))

    js=f.readline()
    Pos_dic=json.loads(js)

    js = f.readline()
    Neg_dic = json.loads(js)

    js = f.readline()
    Tru_dic = json.loads(js)

    js = f.readline()
    Dec_dic = json.loads(js)

    js = f.readline()
    vocabulary = json.loads(js)
    voc_length=len(vocabulary)

    for term in vocabulary:
        if(term in Pos_dic):
            pass
        else:
            Pos_dic[term]=0
        if (term in Neg_dic):
            pass
        else:
            Neg_dic[term] = 0

        if (term in Tru_dic):
            pass
        else:
            Tru_dic[term] = 0
        if (term in Dec_dic):
            pass
        else:
            Dec_dic[term] = 0

    feature_selection()

def feature_selection():
    global posdoc,negdoc,trudoc,decdoc
    global TDvoca,PNvoca
    # positive or negative
    iglist=[]
    select_k=3800

    epsilon=1e-8


    for item in vocabulary:
        pw=(Pos_dic[item]+Neg_dic[item])/(posdoc+negdoc)
        pc1_w=Pos_dic[item]/(Pos_dic[item]+Neg_dic[item])
        pc2_w=Neg_dic[item]/(Pos_dic[item]+Neg_dic[item])

        pc1_wnot=(   (posdoc-Pos_dic[item]) /(posdoc-Pos_dic[item]+negdoc-Neg_dic[item])    )
        pc2_wnot= (   (negdoc-Neg_dic[item]) /(posdoc-Pos_dic[item]+negdoc-Neg_dic[item])    )
        # if(pc1_w==0 or pc2_w==0 or pc1_wnot==0 or pc2_wnot==0):
        #     continue

        infogain=pw*(pc1_w*math.log(pc1_w+epsilon)+pc2_w*math.log(pc2_w+epsilon)) + (1-pw)*(pc1_wnot*math.log(pc1_wnot+epsilon)+\
                                                                            pc2_wnot*math.log(pc2_wnot+epsilon))

        iglist.append((item,infogain))

    newlist=sorted(iglist,key=lambda x:x[1],reverse=True)
    print(len(newlist))
    newlist=[ele[0] for ele in newlist]
    PNvoca=set(newlist[:select_k])

    for item in vocabulary:
        pw=(Tru_dic[item]+Dec_dic[item])/(trudoc+decdoc)
        pc1_w=Tru_dic[item]/(Tru_dic[item]+Dec_dic[item])
        pc2_w=Dec_dic[item]/(Tru_dic[item]+Dec_dic[item])

        pc1_wnot=(   (trudoc-Tru_dic[item]) /(trudoc-Tru_dic[item]+decdoc-Dec_dic[item])    )
        pc2_wnot= (   (decdoc-Dec_dic[item]) /(trudoc-Dec_dic[item]+decdoc-Dec_dic[item])    )
        # if(pc1_w==0 or pc2_w==0 or pc1_wnot==0 or pc2_wnot==0):
        #     continue

        infogain=pw*(pc1_w*math.log(pc1_w+epsilon)+pc2_w*math.log(pc2_w+epsilon)) + (1-pw)*(pc1_wnot*math.log(pc1_wnot+epsilon)+\
                                                                            pc2_wnot*math.log(pc2_wnot+epsilon))

        iglist.append((item,infogain))

    newlist = sorted(iglist, key=lambda x: x[1], reverse=True)
    newlist = [ele[0] for ele in newlist]
    TDvoca = set(newlist[:select_k])


def classify(filelist,realposneg,realtrudec):
    global voc_length
    global posdoc, negdoc, trudoc, decdoc,Pos_total,Neg_total,Tru_total,Dec_total
    global Pos_dic,Neg_dic,Tru_dic,Dec_dic
    global PNvoca,TDvoca
    global stoplist

    fout=open('nboutput.txt','a')
    pattern = r',|\.|/|;|\[|\]|<|>|\?|:|\{|\}|\~|!|@|#|\$|%|&|\(|\)|-|=|\_|\+|\\|"| '
    mapdic1 = {1: 'truthful', 0: 'deceptive'}
    mapdic2 = {1: 'positive', 0: 'negative'}
    for file in filelist:

        fpredict=open(file)
        alltext = fpredict.read()
        token_list = re.split(pattern, alltext)
        fpredict.close()

        pro_pos=0
        pro_neg=0
        pro_tru=0
        pro_dec=0
        #predict_posneg
        token_set=set(token_list)
        token_list=list(token_set)
        for i,ele in enumerate(token_list):
            token_list[i]=token_list[i].lower()


        for item in PNvoca:
            if(item in token_list):
                pro_pos += math.log((Pos_dic[item] + 1) / (posdoc + 2))
                pro_neg += math.log((Neg_dic[item] + 1) / (negdoc + 2))
            else:
                pro_pos += math.log(1-((Pos_dic[item] + 1) / (posdoc + 2)))
                pro_neg += math.log(1-((Neg_dic[item] + 1) / (negdoc + 2)))

        pro_pos = pro_pos + math.log(posdoc / (posdoc + negdoc))
        pro_neg = pro_neg + math.log(negdoc / (posdoc + negdoc))
        if (pro_pos >= pro_neg):
            PNlabel = 1
        else:
            PNlabel = 0


        # Tru or deceptive
        for item in TDvoca:
            if(item in token_list):
                pro_tru += math.log((Tru_dic[item] + 1) / (trudoc + 2))
                pro_dec += math.log((Dec_dic[item] + 1) / (decdoc + 2))
            else:
                pro_tru += math.log(1-((Tru_dic[item] + 1) / (trudoc + 2)))
                pro_dec += math.log(1-((Dec_dic[item] + 1) / (decdoc + 2)))


        pro_tru = pro_tru + math.log(trudoc / (trudoc + decdoc))
        pro_dec = pro_dec + math.log(trudoc / (trudoc + decdoc))
        #print(pro_tru,pro_dec)

        if(pro_tru>=pro_dec):
            TDlabel=1
        else:
            TDlabel=0
            
        #write into the output text file  and get the chaos matrix

        fout.writelines(mapdic1[TDlabel]+' '+mapdic2[PNlabel]+' '+file+'\n')

    fout.close()

def predict(test_class):
    pnflag = 0
    tdflag = 0
    for key in test_class:
        if (key.rfind("positive") != -1):
            pnflag = 1
        else:
            pnflag = 0
        if (key.rfind("truthful") != -1):
            tdflag = 1
        else:
            tdflag = 0


        classify(test_class[key], pnflag, tdflag)






def testfunc():
    dica={'a':10,'b':20,'z':20}
    dicb={'c':99,'a':11,'d':66}
    dicb.update(dica)
    print(dicb)


if __name__=="__main__":

    model_file = "nbmodel.txt"
    all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt') ) 

    test_by_class = collections.defaultdict(list)
   

    for f in all_files:

        class1, class2, fold, fname = f.split('/')[-4:]
        test_by_class[class1 + class2].append(f)
 
    getmodel()
    predict(test_by_class)




























