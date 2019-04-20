#! /usr/bin/python3
# -*- coding: utf-8 -*-

import nltk
import spacy
import os, sys, json,pickle,re
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import numpy as np
import collections

from gensim import corpora, models
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open("./data/review_yelp_all_12992.json",'r') as load_f:
    load_dict = json.load(load_f)

lemma = nltk.WordNetLemmatizer()
cachedstopwords = stopwords.words("english")


# 分词并过滤
def tokenize(review):
    # 剔除标点符号等干扰因素
	review = re.sub("[^a-zA-Z#]", " ",review)
	final = []
	# 分句
	sent_text = nltk.sent_tokenize(review)
	for sentence in sent_text:
        # 分词
		tokenized_text = nltk.word_tokenize(sentence)
		for word in tokenized_text:
            # 剔除stopword以及长度小于2的字符串
			if len(word)>2 and not word in cachedstopwords:
                # 全部转换为小写
				final.append(word.lower())
	return lemma_nltk(final)

# 获取词性
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# 词性过滤以及词干提取(spacy),text最大限制为1000000
def lemma_spacy(text,tags=['NOUN','ADJ']):
    nlp = spacy.load('en', disable=['parser', 'ner'])
    doc = nlp(" ".join(text))
    res = [token.lemma_ for token in doc if token.pos_ in tags]
    return res

def lemma_nltk(text,tags=["NN","NNS","NNP","NNPS","JJ","JJS","JJR","RB",'MD', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RP', 'RB', 'RBR', 'RBS']):
    taggedTxt = nltk.pos_tag(text)
    res = []
    for taggedWord in taggedTxt:
        if taggedWord[1] in tags:
            wordnet_pos = get_wordnet_pos(taggedWord[0]) or wordnet.NOUN
            lemmedWord = lemma.lemmatize(taggedWord[0],pos=wordnet_pos)
            # nltk无法还原best...
            if lemmedWord == "best":
                lemmedWord = "good"
            res.append(lemmedWord)
    return res


# 生成词频图
def freq_words(lem_words, terms = 20):
    fdist = nltk.FreqDist(lem_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()


# 生成并分析词频图
def plot_freq_words(lines = 1000000,top = 20):
    with open('reviewTxt.txt', 'r') as f:
        review = f.read(lines)
    words = tokenize(review)
    freq_words(words,terms=top)

# plot_freq_words()

def generate_filter_word():
    content = []
    for item in load_dict:
        rawTxt = load_dict[item]
        filterdWord = tokenize(rawTxt)
        content.append(filterdWord)
    with open('filterWords.txt', 'wb') as f:
        pickle.dump(content,f)
    return content

with open("./data/filterWords.txt",'rb') as f:
    content = pickle.load(f)
# 得到文档-单词矩阵 （直接利用统计词频得到特征）
dictionary = corpora.Dictionary(content)
# 将dictionary转化为一个词袋，得到文档-单词矩阵
texts = [dictionary.doc2bow(text) for text in content]
# 利用tf-idf来做为特征进行处理
texts_tf_idf = models.TfidfModel(texts)[texts]
lda = models.ldamodel.LdaModel(corpus=texts_tf_idf, id2word=dictionary, num_topics=4)
for index,topic in lda.print_topics(4):
    print(topic)

corpus_lda = lda[texts_tf_idf]
with open('feature4.txt', 'wb') as f:
    pickle.dump(corpus_lda,f)
