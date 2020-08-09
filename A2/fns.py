
import os
from pprint import pprint
from bs4 import BeautifulSoup
from nltk import WordNetLemmatizer
from nltk import regexp_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk import WordPunctTokenizer
import pandas as pd 
import numpy as np
import math

story_path = os.getcwd() + "/stories/"

def documentText(file_name):
    """
        Args:
            file_name (str) : file name 
        return:
            string (str) : file content
    """

    with open(story_path+'/'+file_name, 'rb') as f:
        file_content = str(f.read())
    
    return file_content

def getData(path):
    """
        Args:
            path (str): index.html file location
        Returns:
            dataframe:
    """
    
    loc = path + 'index.html'
    with open(loc, 'r') as f:
        data = str(f.read())
        
    sp = BeautifulSoup(data, "lxml")
    filename = []
    title = []

    for t in sp.find_all('tr'):
        flag = False

        for c in t.find_all('td'):
            if(not flag):    
                for a in c.find('a'):
                    filename.append(a)
                    flag = True  
            if(c.string != None):
                title.append(c.string.rstrip())
                flag = False
        break

    docs_frame = pd.DataFrame(columns = ["docid", "name", "title", "content"])
    # pprint(list(zip(filename, title)))

    f = 1
    for i in range(0, len(filename)):
        text = documentText(filename[i])
        docs_frame = docs_frame.append(pd.Series([f, filename[i], title[i], text], index = ["docid", "name", "title","content"]), ignore_index=True)
        f+=1

    return docs_frame


def tokenizeDocument(docs):
    """
        Args:
            docs (dict) : all documents
        return:
            final tokens (list) : terms for index

        0. Convert to lowercase
        1. Stop words removed
        2. Tokenize 
        3. Stemming
        4. Lemmatization
        5. Only words that starts with alphabet or digit. Front 0's removed.
    """

    stop_ws = set(stopwords.words('english'))

    ts = docs.split('\\n')
    docs = ' '.join(ts)
    ts = docs.split('\t')

    docs = ' '.join(ts)
  
    # Tokenization
    tokens = WordPunctTokenizer().tokenize(docs)

    # lowercase
    tokens_lowercase = [ w.lower() for w in tokens]
    
    #Remove Stop words
    tokens_stop  = [ w for w in tokens_lowercase if(w not in stop_ws)] 
    
    # Stemming 
    tokens_stem = [ PorterStemmer().stem(w) for w in tokens_stop]   # .wes. we

    # Lemmatization
    updated_tokens = [ WordNetLemmatizer().lemmatize(w) for w in tokens_stem]
     
    final_tokens = []
    
    for updated_token in updated_tokens:
        if(updated_token[0].isalpha()) and (len(updated_token) > 1):
            final_tokens.append(updated_token)
        else:
            if(updated_token.isnumeric()):
                final_tokens.append(str(int(updated_token)))
            else:
                if(updated_token[0].isdigit()):
                    updated_token = updated_token.lstrip('0')
                    final_tokens.append(updated_token)
        
    
    final_tokens = final_tokens[1:]  # remove b

    return final_tokens

def tokenizeInput(inp):
    """
        inp : Not a stop word
        
        0. Convert to lowercase
        1. Stemming
        2. Lemmatization
        3. Only words that starts with alphabet or digit.
    """
    
    # lowercase
    inp = inp.lower()
    
    # Stemming 
    inp_stem = PorterStemmer().stem(inp)
    
    # Lemmatization
    inp_lemma = WordNetLemmatizer().lemmatize(inp_stem)
    
    inp_lemma = inp_lemma.lstrip('0')
    
    #strip spaces 
    inp_lemma = inp_lemma.strip()
    
    return inp_lemma


#### Inverted Index ####

def makeIndex(token_docid):
    inverted_index = {}

    for element in token_docid:
        term = element[0]
        docid = element[1]
        if(term not in inverted_index.keys()):
            postings_list = [0,{}]  # df, key = docid, value = tf
            postings_list[0] = 1
            postings_list[1][docid] = 1 #tf 
            inverted_index[term] = postings_list
        else:
            plist = inverted_index[term] 
            if docid not in plist[1].keys():
               plist[1][docid] = 1 #tf 
               plist[0]+=1
            else:
                plist[1][docid]+=1

    return inverted_index



def queryTermsOR(qts, inverted_index):
    docs_found = []
    
    for qt in qts:
        if(qt in inverted_index.keys()):
            for doc_id in inverted_index[qt][1].keys():
                if(doc_id not in docs_found):
                    docs_found.append([qt, doc_id])
        
    return docs_found




def getResults(N, query_docs, title_index, text_index):
    score = []

    for qt_docid in query_docs:
        print(qt_docid)
        a = 1 
        b = 1
        if qt_docid[0] in title_index.keys():
            tf_title = 1 + math.log(title_index[qt_docid[0]][1][qt_docid[1]], 2)
            idf_title = math.log(N/title_index[qt_docid[0]][0], 2)
            a = 0.7 * (tf_title) * (idf_title)
        else:
            a = 0

        if qt_docid[0] in text_index.keys():     
            tf_all = 1 + math.log(text_index[qt_docid[0]][1][qt_docid[1]], 2)
            idf_all = math.log(N/text_index[qt_docid[0]][0], 2)

            tf_body = tf_all - tf_title
            idf_body = idf_all - idf_title
            b = 0.3 * (tf_body * idf_body)
        else:
            b = 0

        s = a + b
        score.append([s, qt_docid[0],qt_docid[1]])

    return score