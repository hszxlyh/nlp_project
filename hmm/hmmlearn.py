import os,sys
import glob
import collections
import pickle,json

start_dict={}
tag_dict={}
observe_dict={}
tag_count=0
word_count=0
tag_map={}
word_map={}

def readfiles(file):
    global start_dict,tag_dict,observe_dict
    global tag_count,word_count,tag_map,word_map

    f=open(file,'r',encoding='utf-8')
    while(1):
        sentence=f.readline()
        #print(sentence)
        if(sentence==""):
            break
        split_tokens=sentence.split()
        first_tag=split_tokens[0].rsplit('/',1)[1]
        if first_tag in start_dict:
            start_dict[first_tag]+=1
        else:
            start_dict[first_tag]=1
        lasttag=None
        for one_tagword in split_tokens:
            word,tag=one_tagword.rsplit('/',1)
            tag=str.upper(tag)
            if(tag not in known_tags):
                known_tags.add(tag)
                tag_map[tag]=tag_count
                tag_count+=1

                tag_dict[tag]={}
                observe_dict[tag]={}
            if word not in known_words:
                known_words.add(word)
                word_map[word]=word_count
                word_count+=1

            #observe_dict[tag][word]
            if lasttag==None:
                #only map the word and tag ----1_step
                if(word in observe_dict[tag]):
                    observe_dict[tag][word]+=1
                else:
                    observe_dict[tag][word]=1

                lasttag=tag
            else:
                if tag in tag_dict[lasttag]:
                    tag_dict[lasttag][tag]+=1
                else:
                    tag_dict[lasttag][tag]=1
                lasttag=tag
                if (word in observe_dict[tag]):
                    observe_dict[tag][word] += 1
                else:
                    observe_dict[tag][word] = 1



if __name__=="__main__":
    length_tag=0
    length_word=0
    known_tags = set()
    known_words = set()
    #all_files = glob.glob(os.path.join(os.getcwd() + '\\data', '*.txt'))
    file1="it_isdt_train_tagged.txt"
    file2="ja_gsd_train_tagged.txt"

    readfiles(file1)
    #compute number of tags and words
    for outkey in tag_dict:
        for innerkey in tag_dict[outkey]:
            known_tags.add(innerkey)
    length_tag=len(known_tags)
    for outkey in observe_dict:
        for innerkey in observe_dict[outkey]:
            known_words.add(innerkey)
    length_word=len(known_words)

    f_model=open("hmmmodel.txt",'w',encoding='utf-8')
    js=json.dumps(tag_map)
    f_model.writelines(js+'\n')
    js=json.dumps(word_map)
    f_model.writelines(js+'\n')
    js=json.dumps(start_dict)
    f_model.writelines(js+'\n')
    js=json.dumps(tag_dict)
    f_model.writelines(js+'\n')
    js=json.dumps(observe_dict)
    f_model.writelines(js+'\n')
    js = json.dumps(observe_dict['CC'])
    f_model.writelines(js + '\n')

    f_model.close()


































