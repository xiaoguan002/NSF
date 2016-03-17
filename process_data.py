import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import MySQLdb
import string

def build_data_cv(cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """

    try:
        conn=MySQLdb.connect('private')
        cur=conn.cursor()

        cur.execute('select AbstractNarration, OrganizationDivisionLongName from nsfmain')

        results_code=cur.fetchall()

        cur.close()
        conn.close()
    except MySQLdb.Error,e:
       print "Mysql Error %d: %s" % (e.args[0], e.args[1])


    revs = []

    vocab = defaultdict(float)
    for r in results_code:

        div = 0
        if r[1] == 'Division of Computer and Network Systems' or  r[1] == 'Division Of Computer and Network Systems':
            div = 0
        elif r[1] =='Div Of Information & Intelligent Systems' or r[1] =='Division of Information & Intelligent Systems':
            div = 1
        elif r[1] =='Division of Computing and Communication Foundations' or r[1] =='Division of Computer and Communication Foundations'or \
             r[1]== 'Div Of Computer & Communication Foundati':
            div = 2
        elif r[1] =='Division of Advanced CyberInfrastructure' or r[1] =='Div Of Advanced Cyberinfrastructure':
            div = 3
        else :
            continue

        rev = []
        if r[0] == '':
            continue


        rev.append(r[0].strip())
        orig_rev = text_to_word_sequence(" ".join(rev))
        if len(orig_rev) < 20:
            continue

        words = set(orig_rev)
        for word in words:
            vocab[word] += 1







        datum  = {"y":div,
                  "text": orig_rev,
                  "num_words": len(orig_rev),
                  "split": np.random.randint(0,cv)}

        #if datum['num_words'] > 400:
        #    continue
        revs.append(datum)

   # for i in range(1):
   #    print revs

    return revs, vocab


def base_filter():
    f = string.punctuation
    f = f.replace("'", '')
    f += '\t\n'
    return f

def text_to_word_sequence(text, filters=base_filter(), lower=True, split=" "):
    '''prune: sequence of characters to filter out
    '''
    if lower:
        text = text.lower()
    text = text.translate(string.maketrans(filters, split*len(filters)))
    seq = text.split(split)
    return [_f for _f in seq if _f]

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        print " Vocab_size:", vocab_size
        print " Vocab_dimension:", layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  


if __name__=="__main__":    
    w2v_file = 'F:/GoogleNews-vectors-negative300.bin'

    print "loading data...",        
    revs, vocab = build_data_cv(cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    min_l = np.min(pd.DataFrame(revs)["num_words"])
    mean_l = np.mean(pd.DataFrame(revs)["num_words"])
    '''
    for i in range(4):
        print revs[i]

    exit()
    '''
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "min sentence length: " + str(min_l)
    print "mean sentence length: " + str(mean_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, W2, word_idx_map, vocab, max_l], open("mr.p", "wb"))
    print "dataset created!"
    
