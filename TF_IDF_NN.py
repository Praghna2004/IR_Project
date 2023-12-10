
# DISEASE DETECTION AND IDENTIFICATION BASED ON SYMPTOMS 

# importing nltk to download resources for stopwords and wordnet
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# importing all libraries
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split,cross_val_score
import math
import operator
import pickle
import re
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from statistics import mean
from nltk.corpus import wordnet 
import requests
from bs4 import BeautifulSoup
from itertools import combinations
from time import time
from collections import Counter
import operator
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import operator

warnings.resetwarnings()

# *Disease Symptom dataset** was created in a separate python program.

# *Dataset scrapping** was done using **NHP website** and **wikipedia data*


# Load Dataset scraped from NHP (https://www.nhp.gov.in/disease-a-z) & Wikipedia
# Scrapping and creation of dataset csv is done in a separate program

# Obtained DISEASE SYMPTOM dataset from kaggle and calculating TF-IDF values
df=pd.read_csv("Dataset\dis_sym_dataset_norm.csv")

## extract disease labels and store them in 'doc_lists'
doc_lists=list(df['label_dis'])
df=df.iloc[:,1:]
col_names=list(df.columns)
doc_lists=list(doc_lists)

#extract symptoms and store them in 'col_names'
N=len(df)
M=len(col_names)

# All symptoms IDF
idf={}
for col in col_names:
  temp=np.count_nonzero(df[col])
  idf[col]=np.log(N/temp)

# All disease,symptom TF
tf={}
for i in range(N):
  for col in col_names:
    key=(doc_lists[i],col)
    tf[key]=df.loc[i,col]

# All disease,symptom TF.IDF
tf_idf={}
for i in range(N):
  for col in col_names:
    key=(doc_lists[i],col)
    tf_idf[key]=float(idf[col])*float(tf[key])

# The matrix will contain the TF-IDF values for each disease-symptom pair.
# rows - diseases
# columns - symptoms
matrix = np.zeros((N, M),dtype='float32')
for i in tf_idf:
    sym = col_names.index(i[1])
    dis=doc_lists.index(i[0])
    matrix[dis][sym] = tf_idf[i]

# similarity calculation using cosine siilarity and TF-IDF values


def cos_Similarity(a, b):
    # Check if any element in the arrays is non-numeric
    if not np.issubdtype(a.dtype, np.number) or not np.issubdtype(b.dtype, np.number):
        return 0  # Return 0 if non-numeric elements are present

    # Check if the norm of any array is zero
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    else:
        temp = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return temp


# convert data to lower case
def conv_lower(data):
    return data.lower()

# tokenizing using regextokenizer
def regextokenizer_func(data):
    token = RegexpTokenizer(r'\w+')
    data = token.tokenize(data)
    return data

# generate query vector for tf_idf
def generate_vector(tokens):
    Q = np.zeros(M)
    counter = Counter(tokens)
    query_weights = {}
    for token in np.unique(tokens):
        tf = counter[token]
        try:
          idf_temp=idf[token]
        except:
          pass
        try:
            ind = col_names.index(token)
            Q[ind] = tf*idf_temp
        except:
            pass
    return Q

# function to calculate tf_idf_score
def tf_idf_score(k, query):
    query_weights = {}
    for key in tf_idf:
        if key[1] in query:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]
    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)
  
    l = []
    for i in query_weights[:k]:
        l.append(i)
    return l

# function to calculte Cosine Similarity 

def cosine_similarity(k, query):
    d_cosines = []
    # query_vector = np.array(query)  
    query_vector = generate_vector(query)  
    for d in matrix:
        d_cosines.append(cos_Similarity(query_vector, d))
    out = np.array(d_cosines).argsort()[-k:][::-1]
  
    final_display_disease={}
    for lt in set(out):
      final_display_disease[lt] = float(d_cosines[lt])
    return final_display_disease


# Retrieving Synonyms-symptoms list for a given input from thesaurus.com and Wordnet 
def QueryExpansion_Synonyms(term):
    QueryExpansion_Synonyms = []
    response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
    soup = BeautifulSoup(response.content,  "html.parser")
    try:
        container=soup.find('section', {'class': 'MainContentContainer'}) 
        row=container.find('div',{'class':'css-191l5o0-ClassicContentCard'})
        row = row.find_all('li')
        for x in row:
            QueryExpansion_Synonyms.append(x.get_text())
    except:
        None
    for syn in wordnet.synsets(term):
        QueryExpansion_Synonyms+=syn.lemma_names()
    return set(QueryExpansion_Synonyms)

splitter = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


# The Disease Combination dataset includes combinations for every disease that is present in the dataset because, in reality, it is frequently noted that a person does not necessarily have a condition when they have every symptom.

# *To tackle this problem, combinations are made with the symptoms for each disease.*

#  **This increases the size of the data exponentially and helps the model to predict the disease with much better accuracy.**

# *df_comb -> dataset generated by combining symptoms for each disease.*

# *df_norm -> dataset which contains a single row for each diseases with the symptoms for that corresponding disease.*

# **Dataset contains 261 diseases and their symptoms**



# Load Dataset scraped from NHP (https://www.nhp.gov.in/disease-a-z) & Wikipedia
# Scrapping and creation of dataset csv is done in a separate program

df_comb = pd.read_csv("Dataset\dis_sym_dataset_comb.csv")
df_norm = pd.read_csv("Dataset\dis_sym_dataset_norm.csv") 
Y = df_norm.iloc[:, 0:1]
X = df_norm.iloc[:, 1:]
# List of symptoms
dataset_symptoms = list(X.columns)
diseases = list(set(Y['label_dis']))
diseases.sort()

# Taking symptoms from user as input
# Preprocessing the input symtoms 

st.title("Disease detection System")
st.markdown('<h4 style="color:blue;">User Information Form</h4>', unsafe_allow_html=True)
p1,p2 = st.columns((4,4))
with p1:
    name = st.text_input("Enter your name:", key='name_input')

with p2:
    email = st.text_input("Enter your email:", key='email_input')


user_symptoms = st.text_input("* Enter symptoms you experienced separated by comma (,):",key = 123456789).lower().split(',')

# user_symptoms = str(input("\nPlease enter symptoms separated by comma(,):\n")).lower().split(',')
processed_user_symptoms=[]
for sym in user_symptoms:
    sym=sym.strip()
    sym=sym.replace('-',' ')
    sym=sym.replace("'",'')
    sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
    processed_user_symptoms.append(sym)

st.spinner("Analysing your symptoms")
# Finding all of the QueryExpansion_Synonyms for each user symptom and adding them to the pre-processed symptom string
user_symptoms = []
for user_sym in processed_user_symptoms:
    user_sym = user_sym.split()
    str_sym = set()
    for comb in range(1, len(user_sym)+1):
        for subset in combinations(user_sym, comb):
            subset=' '.join(subset)
            subset = QueryExpansion_Synonyms(subset) 
            str_sym.update(subset)
    str_sym.add(' '.join(user_sym))
    user_symptoms.append(' '.join(str_sym).replace('_',' '))

# query expansion is carried out by combining QueryExpansion_Synonyms discovered for every symptom that was initially input.
print("After query expansion done by using the symptoms entered")
print(user_symptoms)

st.write("* After query expansion using QueryExpansion_Synonyms:")
st.write(user_symptoms)

# Iterate through every symptom in the dataset and compare its similarity score to the user-inputted synonym string.
# If similarity>0.5 -> add the symptom to the final list

found_symptoms = set()
for idx, data_sym in enumerate(dataset_symptoms):
    data_sym_split=data_sym.split()
    for user_sym in user_symptoms:
        count=0
        for symp in data_sym_split:
            if symp in user_sym.split():
                count+=1
        if count/len(data_sym_split)>0.5:
            found_symptoms.add(data_sym)
found_symptoms = list(found_symptoms)

# Print all found symptoms
print("Top matching symptoms based on the user query!")
for idx, symp in enumerate(found_symptoms):
    print(idx,":",symp)


st.write("* Top matching symptoms based on the user query:")
for idx, symp in enumerate(found_symptoms):
    st.write(idx,":",symp)

# Ask the user to choose from a list of relevant symptoms that were detected in the dataset.

select_list = st.text_input("* Please select the relevant symptoms. Enter indices (separated-space):",key = "<uniquevalueofsomesort>").split()


dis_list = set()
final_symp = [] 
counter_list = []
for idx in select_list:
    symp=found_symptoms[int(idx)]
    final_symp.append(symp)
    dis_list.update(set(df_norm[df_norm[symp]==1]['label_dis']))
   
for dis in dis_list:
    row = df_norm.loc[df_norm['label_dis'] == dis].values.tolist()
    if row:
        row[0].pop(0)
        for idx,val in enumerate(row[0]):
            if isinstance(val,str) and not val.replace('.','',1).isdigit():
                st.warning(f"Skipping non-numeric value: {val}")
                continue

            numeric_val = float(val)
            if numeric_val != 0 and dataset_symptoms[idx] not in final_symp:
              counter_list.append(dataset_symptoms[idx])
             
dict_symp = dict(Counter(counter_list))
dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1),reverse=True)
#dict_symp_tup contains a list of tuples, where each tuple represents a co-occurring symptom along with its count, sorted by count in descending order.

# Iteratively, suggest top co-occuring symptoms to the user and ask to select the ones applicable 
found_symptoms=[]
count=0
uniq_val = 0
for tup in dict_symp_tup:
    count+=1
    found_symptoms.append(tup[0])
    if count%5==0 or count==len(dict_symp_tup):
        print("\nCommon co-occuring symptoms:")
        for idx,ele in enumerate(found_symptoms):
            print(idx,":",ele)
        st.write("* Common co-occuring symptoms are listed below: ")
        for idx,ele in enumerate(found_symptoms):
            st.write(idx,":",ele, key=f"symptom_{uniq_val}_{idx}")
            

        # select_list = input("Do you have have of these symptoms? If Yes, enter the indices (space-separated), 'no' to stop, '-1' to skip:\n").lower().split()


        select_list = st.text_input("* Do you have have of these symptoms? If Yes, enter the indices (space-separated), 'no' to stop, '-1' to skip:\n",key = f"input_{uniq_val}").lower().split()

       
        if select_list and select_list[0]=='no':
            break
      
        if select_list and select_list[0]=='-1':
            found_symptoms = [] 
            continue
    
        for idx in select_list:
            final_symp.append(found_symptoms[int(idx)])
    
        found_symptoms = []
        uniq_val+=1

# Final Symptom list


# Calculating TF-IDF and Cosine Similarity using matched symptoms
k = 5

print("Final list of Symptoms used for prediction are : ")
for val in final_symp:
    print(val)

st.write("* Final list of Symptoms used for prediction are : ")
for val in final_symp:
    st.write(val)

# Predicting the top 10 diseases based on symptoms given by user

# k=10
topk1=tf_idf_score(k,final_symp)
topk2=cosine_similarity(k,final_symp)

# display top k diseases predicted with TF-IDF

st.write(f"* Top {k} diseases predicted based on TF_IDF Matching :")
i = 0
topk1_index_mapping = {}
for key, score in topk1:
  st.write(f"{i}. Disease : {key} \t Score : {round(score, 2)}")
  topk1_index_mapping[i] = key
  i += 1



# display top k diseases predicted with cosine probablity
st.write(f"* Top {k} disease based on Cosine Similarity Matching :\n ")
topk2_sorted = dict(sorted(topk2.items(), key=lambda kv: kv[1], reverse=True))
j = 0
topk2_index_mapping = {}
for key in topk2_sorted:
  st.write(f"{j}. Disease : {diseases[key]} \t Score : {round(topk2_sorted[key], 2)}")
  topk2_index_mapping[j] = diseases[key]
  j += 1

finalitems = list(topk2_sorted.items())


relevant_ind = st.text_input(f"* Please select the relevant documents. Enter indices (separated-space):").split()
relevant_docs = list()
for i in relevant_ind:
    docsid = finalitems[int(i)][0]
    relevant_docs.append(str(docsid))

st.write("* Relevant Docs Id's :",relevant_docs)

relevant = 0
retrieved = 0
total_relevant = len(relevant_docs)
map_val = 0
recall = list()
precision = list()
print("\nEvaluation Metrics: (Precision, Recall) :\n")
st.write("* Evaluation Metrics: (Precision, Recall) :")

for key in topk2_sorted:
    try:
        retrieved += 1
        if str(key) in relevant_docs:
            relevant += 1
            map_val += round((relevant/retrieved), 2)
        recall.append(round((relevant/total_relevant),2))
        precision.append(round((relevant/retrieved), 2))
    except ZeroDivisionError:
            st.write("Error: Division by zero.")
            # st.write("Precision : ", round((relevant/retrieved), 2), " Recall : ", round((relevant/total_relevant),2), "\n")



# print("\nMean Average Precision(MAP) for the above query is: ", map_val/relevant)
plt.plot(recall,precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("PRECISION-RECALL CURVE")
# st.write("\nMean Average Precision(MAP) for the above query is: ", map_val/relevant)
st.pyplot(plt)
plt.show()


plt.close()

# Store relevance feedback information in "relevance_feedback.json" 

import json
item = dict()

for eachs in final_symp:
    item[eachs] = relevant_docs
print(item)
with open("relevance_feedback.json", "w", encoding='utf-8') as file:
    json.dump(item, file, indent=4)
    file.close()

# Extracting relevant queries based on the relevance feedback

import ast
f = open("relevance_feedback.json", 'r')
relevance_feedback = json.load(f)
f.close()
relevant_queries = list(relevance_feedback.keys())
print(relevant_queries)

final_similarity = {}
ranking = dict()
original = cosine_similarity(k,final_symp)
relevance_score = 0.1
for query in final_symp:
    for doc in topk2_sorted.keys():
        if(query in relevant_queries and str(doc) in relevance_feedback[query]):
            original[int(doc)] = original[int(doc)] + relevance_score
    final_similarity =  dict(sorted(original.items(), key = lambda x : x[1], reverse = True))
    
print(final_similarity)

st.success(f"Hey {name}! Here are our results:")
   

st.write(f"* Top {k} diseases after Relevance Feedback :\n ")
j = 0
topk2_index_mapping = {}
for key in final_similarity:
  st.write(f"{j}. Disease [{key}] : {diseases[key]} \t Score : {round(final_similarity[key], 2)}")
  print(f"{j}. Disease [{key}] : {diseases[key]} \t Score : {round(final_similarity[key], 2)}")
  topk2_index_mapping[j] = diseases[key]
  j += 1


st.success('Done! Prevention is better than cure')
