import os,sys
import glob,pickle,json
import collections
import numpy as np

tag_map={}
word_map={}
number2tag={}
start_matrix=[]
transition_matrix=[]
emission_matrix=[]

def get_accuracy(standard,predict):
    right=0
    error=0
    f_stan=open(standard,'r',encoding='utf-8')
    f_pre=open(predict,'r',encoding='utf-8')
    while(1):
        st_sentence=f_stan.readline()
        pre_sentence=f_pre.readline()

        if(st_sentence==""):
            break
        st_list = st_sentence.split()
        pre_list= pre_sentence.split()
        #first_tag = split_tokens[0].rsplit('/', 1)[1]
        length=len(st_list)
        for i in range(0,length):
            st_tag=st_list[i].rsplit('/',1)[1]
            pre_tag=pre_list[i].rsplit('/', 1)[1]
            if(st_tag==pre_tag):
                right+=1
            else:
                error+=1
    return right/(right+error)
def get_para(model_file,read_file):
    global tag_map,word_map,start_matrix,transition_matrix,emission_matrix
    global number2tag
    start_dict={}
    tag_dict={}
    observe_dict={}

    f=open(model_file,'r',encoding='utf-8')

    js=f.readline().strip('\n')
    tag_map=json.loads(js)
    js = f.readline().strip('\n')
    word_map = json.loads(js)
    js = f.readline().strip('\n')
    start_dict = json.loads(js)
    js = f.readline().strip('\n')
    tag_dict = json.loads(js)
    js = f.readline().strip('\n')
    observe_dict = json.loads(js)

    length_tags=len(tag_map)
    length_words=len(word_map)

    count_tag={}


    for ta in tag_map:
        number2tag[tag_map[ta]]=ta
        count_tag[ta]=0


    start_matrix=np.zeros(length_tags)
    transition_matrix=np.zeros((length_tags,length_tags))
    emission_matrix=np.zeros((length_tags,length_words))

    for tag_k in start_dict:
        start_matrix[tag_map[tag_k]]=start_dict[tag_k]
    start_matrix=start_matrix/np.sum(start_matrix)


    for outkey in tag_dict:
        for inkey in tag_dict[outkey]:
            transition_matrix[tag_map[outkey]][tag_map[inkey]] =tag_dict[outkey][inkey]
    transition_matrix=transition_matrix+1


    temp_sum=transition_matrix.sum(axis=1)
    transition_matrix=transition_matrix/temp_sum[:,np.newaxis]


    for outkey in observe_dict:
        for inkey in observe_dict[outkey]:
            emission_matrix[tag_map[outkey]][word_map[inkey]]=observe_dict[outkey][inkey]
            count_tag[outkey]+=1
    temp_sum=emission_matrix.sum(axis=1)
    emission_matrix=emission_matrix/temp_sum[:,np.newaxis]

    most_k=5
    tag_with_sort=sorted(count_tag.items(),key = lambda x:x[1],reverse = True)
    tag_with_sort=tag_with_sort[:most_k]
    newtag={}

    for m in range(0,most_k):
        newtag[tag_with_sort[m][0]]=tag_map[tag_with_sort[m][0]]







    f_out=open("hmmoutput.txt",'w',encoding='utf-8')
    f_read=open(read_file,'r',encoding='utf-8')
    while(1):
        get_sentence=f_read.readline()
        if(get_sentence==""):
            break
        split_word=get_sentence.split()

        #Viterbi------------
        length=len(split_word)
        viterbi=np.zeros((length_tags,length))
        backpointer=np.zeros((length_tags,length))

        #Init First step
        if(split_word[0] not in word_map):
            for tag in tag_map:
                s=tag_map[tag]
                if tag in newtag:
                    if start_matrix[s] == 0:
                        viterbi[s, 0] = 0
                    else:
                        viterbi[s, 0] = np.log(start_matrix[s])
                else:
                    viterbi[s,0] = 0
        else:
            for tag in tag_map:
                s=tag_map[tag]
                value=start_matrix[s]*emission_matrix[s,word_map[split_word[0]]]
                if(value==0):
                    viterbi[s,0]=0
                else:
                    viterbi[s,0]= np.log(value)

        for time_step in range(1,length):
            step_word=split_word[time_step]
            if(step_word not in word_map):
                for each_state in tag_map:
                    s = tag_map[each_state]
                    if each_state in newtag:
                        max_viterbi = 0
                        argmax_tag = ""
                        for last_state in tag_map:
                            s_last = tag_map[last_state]
                            # new_viterbi_value =  * transition_matrix[s_last, s]
                            if viterbi[s_last, time_step - 1] == 0:
                                pass
                            elif max_viterbi == 0 or \
                                    viterbi[s_last, time_step - 1] + np.log(transition_matrix[s_last, s]) > max_viterbi:
                                max_viterbi = viterbi[s_last, time_step - 1] + np.log(transition_matrix[s_last, s])
                                argmax_tag = last_state
                        viterbi[s, time_step] = max_viterbi
                        backpointer[s, time_step] = tag_map[argmax_tag]
                    else:
                        viterbi[s,time_step]=0

            else:
                for each_state in tag_map:
                    max_viterbi = 0
                    argmax_tag = ""
                    s = tag_map[each_state]
                    for last_state in tag_map:
                        s_last = tag_map[last_state]
                        #new_viterbi_value =  * transition_matrix[s_last, s]
                        if viterbi[s_last, time_step - 1]==0:
                            pass
                        elif max_viterbi==0 or  \
                                viterbi[s_last,time_step-1]+np.log(transition_matrix[s_last, s])  >max_viterbi:
                            max_viterbi=viterbi[s_last,time_step-1]+np.log(transition_matrix[s_last, s])
                            argmax_tag=last_state
                    if emission_matrix[s,word_map[step_word]]==0:
                        viterbi[s,time_step]=0
                    else:
                        viterbi[s,time_step]=max_viterbi+np.log(emission_matrix[s,word_map[step_word]])
                    backpointer[s,time_step]=tag_map[argmax_tag]

        #---------------
        print(viterbi)
        print(tag_map)

        with open('zr.json', mode='a') as f:
            json.dump(viterbi.tolist(), fp=f)


        maxvalue=float('-inf')
        endtag=0
        for t in tag_map :
            if viterbi[tag_map[t],length-1]==0:
                continue
            if(viterbi[tag_map[t],length-1]>maxvalue):
                maxvalue=viterbi[tag_map[t],length-1]
                endtag=tag_map[t]

        i=length-1
        res=[]
        while(i>=0):
            combo=split_word[i]+"/"+number2tag[endtag]
            endtag=backpointer[int(endtag),i]
            res.append(combo)
            i-=1
        res.reverse()
        out=" ".join(res)
        f_out.writelines(out+"\n")

    f_out.close()
    f_read.close()


if __name__=="__main__":

    model="hmmmodel.txt"
    #target_file="it_isdt_dev_raw.txt"
    #standard_file = "it_isdt_dev_tagged.txt"
    #target_file="ja_gsd_dev_raw.txt"
    #standard_file="ja_gsd_dev_tagged.txt"
    target_file="add.txt"
    predict_file = "hmmoutput.txt"
    get_para(model,target_file)


    #accuracy=get_accuracy(standard_file,predict_file)
    #print(accuracy)



