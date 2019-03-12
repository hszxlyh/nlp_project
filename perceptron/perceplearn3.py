# Expected: generate nbmodel.txt
import sys, os, re
import glob, collections
import json
import pickle
import string
import math
import numpy as np

stoplist = {'', '\n', 'chicago', 'is', 'of', 'the', 'i', 'have', 'at', 'in', 'my', 'a', 'the', 'was', 'when', 'for',
            'they',
            'had', 'there', 'to', 'and', 'it', 'be', 'would', 'this', 'you', 'hotel', 'on', 'were', 'that', 'staff',
            'from',
            'with', 'we', 'our', 'room', 'as', 'rooms', 'one', 'about', 'been', 'here', 'your', 'me', 'are', 'stay',
            'stayed',
            'an', 'us', 'location', 'did', 'service'}
            #'by','bed','view','people','husband','son','wife','daughter','sister','brother'}

vocabulary = {}
word_index = {}
count_index = 0

# -----Feature selection part

PNvoca={}
TDvoca={}

# -------------------------Init end---------------


def get_vocabulary(filedict):
    # Return total number of tokens of POS/NEG and TRUE/DECEP
    global stoplist, count_index
    global vocabulary, word_index
    # pattern=r',|\.|/|;|\[|\]|<|>|\?|:|\{|\}|\~|!|@|#|\$|%|&|\(|\)|-|=|\_|\+|\\|"| '
    pattern = '\W+'
    for key in filedict:
        for pick_onefile in filedict[key]:
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
                        # pass
                        vocabulary[word] += 1
                    else:
                        # vocabulary.add(word)
                        vocabulary[word] = 1
                        word_index[word] = count_index
                        count_index += 1
                 
def train(train_class):
    max_iter = 200

    X_data = []
    pnflag = 0
    tdflag = 0
    y_posneg = []
    y_trudec = []

    voc_length = len(vocabulary)
    w_posneg = np.zeros(voc_length)
    w_trudec = np.zeros(voc_length)
    b_posneg = 0
    b_trudec = 0
    pattern = '\W+'

    # get all the X-datasets
    for key in train_class:
        if (key.rfind("positive") != -1):
            pnflag = 1
        else:
            pnflag = -1
        if (key.rfind("truthful") != -1):
            tdflag = 1
        else:
            tdflag = -1

        for pick_onefile in train_class[key]:
            single_list = np.zeros(voc_length)
            f = open(pick_onefile)
            alltext = f.read()
            token_list = re.split(pattern, alltext)
            y_posneg.append(pnflag)
            y_trudec.append(tdflag)
            for word in token_list:
                if (word.isdecimal() == True):
                    continue
                word = str.lower(word)
                if (word in stoplist):
                    continue
                else:
                    if (word in vocabulary):
                        single_list[word_index[word]] = +1
                    else:
                        pass

            X_data.append(single_list)

    # train for vanilla percep
    for epoch in range(0, max_iter):
        for idx in range(0, len(X_data)):
            if (y_posneg[idx] * (np.dot(X_data[idx], w_posneg) + b_posneg) <= 0):
                w_posneg += y_posneg[idx] * (X_data[idx])
                b_posneg += y_posneg[idx]
            if (y_trudec[idx] * (np.dot(X_data[idx], w_trudec) + b_trudec) <= 0):
                w_trudec += y_trudec[idx] * (X_data[idx])
                b_trudec += y_trudec[idx]
        if(epoch%2==0):
            Total_data = np.column_stack((X_data, y_posneg))
            Total_data = np.column_stack((Total_data, y_trudec))
            np.random.shuffle(Total_data)
            X_data = Total_data[:, :-2]
            y_posneg = Total_data[:, -2]
            y_trudec = Total_data[:, -1]
        

    fw = open('vanillamodel.txt', 'wb')
    pickle.dump(vocabulary, fw)
    pickle.dump(word_index, fw)
    pickle.dump(w_posneg, fw)
    pickle.dump(b_posneg, fw)
    pickle.dump(w_trudec, fw)
    pickle.dump(b_trudec, fw)
    fw.close()

    # train average------------
    w_posneg = np.zeros(voc_length)
    w_trudec = np.zeros(voc_length)
    b_posneg = 0
    b_trudec = 0
    u_posneg = np.zeros(voc_length)
    u_trudec = np.zeros(voc_length)
    beta_posneg = 0
    beta_trudec = 0
    count = 1
    for epoch in range(0, max_iter):
        for idx in range(0, len(X_data)):
            if (y_posneg[idx] * (np.dot(X_data[idx], w_posneg) + b_posneg) <= 0):
                w_posneg += y_posneg[idx] * (X_data[idx])
                b_posneg += y_posneg[idx]
                u_posneg += y_posneg[idx] * count * X_data[idx]
                beta_posneg += y_posneg[idx] * count
            if (y_trudec[idx] * (np.dot(X_data[idx], w_trudec) + b_trudec) <= 0):
                w_trudec += y_trudec[idx] * (X_data[idx])
                b_trudec += y_trudec[idx]
                u_trudec += y_trudec[idx] * count * X_data[idx]
                beta_trudec += y_trudec[idx] * count
            count += 1
        if(epoch%2==0):
            
            Total_data = np.column_stack((X_data, y_posneg))
            Total_data = np.column_stack((Total_data, y_trudec))
            np.random.shuffle(Total_data)
            X_data = Total_data[:, :-2]
            y_posneg = Total_data[:, -2]
            y_trudec = Total_data[:, -1]
        
      

    w_posneg = w_posneg - (1 / count) * u_posneg
    b_posneg = b_posneg - (1 / count) * beta_posneg
    w_trudec = w_trudec - (1 / count) * u_trudec
    b_trudec = b_trudec - (1 / count) * beta_trudec

    fw = open('averagemodel.txt', 'wb')
    pickle.dump(vocabulary, fw)
    pickle.dump(word_index, fw)
    pickle.dump(w_posneg, fw)
    pickle.dump(b_posneg, fw)
    pickle.dump(w_trudec, fw)
    pickle.dump(b_trudec, fw)
    fw.close()


if __name__ == "__main__":

    model_file = "vanillamodel.txt"
    avg_model_file = "averagemodel.txt" 
    input_path = str(sys.argv[1])
    
    all_files = glob.glob(os.path.join(input_path, '*/*/*/*.txt'))

    test_by_class = collections.defaultdict(list)
    train_by_class = collections.defaultdict(list)

    for f in all_files:

        class1, class2, fold, fname = f.split('/')[-4:]
        if fold == 'fold1':
            # True-clause will not enter in Vocareum as fold1 wont exist, but useful for your own code.
            test_by_class[class1 + class2].append(f)
        else:
            train_by_class[class1 + class2].append(f)

    get_vocabulary(train_by_class)
    train(train_by_class)




