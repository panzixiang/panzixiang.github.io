---
layout: post
title: Word Embeddings
category: nlp
---

## Mathematical dimension reduction models

Word Representations suffer from the inherent **Curse of Dimensionality** due to its multidimensional representation in word vector space.

The idea is very simple — make a word vector representation, say in the form of multiple One-Hot vectors. Then deploy a Dimensionality Reduction algorithm such as Matrix Factorization using Singular Value Decomposition (SVD) to arrive at meaningful conclusions.

<!--more-->

### Co-occurrence aka Count based Methods:

A few of the co-occurrence or count based methods of generating Word Vectors are:

* Term Frequency — Inverse Frequency (TF-IDF) Vectorization
* Term-Document Frequency Vectorization

**Latent Semantic Analysis (LSA)** is an extension of SVD, which makes use of a “Document-Term Matrix”. LSA is built on the assumption of the Distributional Hypothesis that words that are close in meaning will occur in similar pieces of text.

The major drawback with such Dimensionality Reduction based methods are two folds —

* they are **computationally expensive**
* they **don't consider the global context** in which the same word occurs.

### NN Word Embedding models

#### Word2Vec

Each word is processed via an Input Layer, followed by a single Hidden Layer. Once the training is complete, the weights of the hidden layer will be used as a proxy representation for the input word.

#### FastText

Improves word2vec by treating each word as composed of n-grams. The whole vector as sum of the vector representation all the character n-grams. Fasttext can generate embedding for the words that does not appear in the training corpus (solves the OOV problem of word2vec). This can be done by adding the character n-gram of all the n-gram representations. 

At the cost of high memory requiremen since the word embedding is at the character/n-gram level, solved by hash binning and dropping low frequency counts

#### Glove

GloVe is a **count-based, unsupervised** learning model that uses co-occurrence (how frequently two words appear together) statistics at a Global level to model the vector representations of words

Word2vec relies only on local information of language. That is, the semantics learnt for a given word, is only affected by the surrounding words.

**The advantage of GloVe is that, unlike Word2vec, GloVe does not rely just on local statistics (local context information of words), but incorporates global statistics (word co-occurrence) to obtain word vectors.**

the appropriate starting point for word vector learning should be with **ratios of co-occurrence probabilities rather than the probabilities themselves**.

#### ELMo ("Embeddings from Language Model")

**Character-level tokens** are taken as the inputs to a **bi-directional LSTM** which produces word-level embeddings. Like BERT (but unlike the word embeddings produced by "Bag of Words" approaches, and earlier vector approaches such as Word2Vec and GloVe), ELMo embeddings are **context-sensitive**, producing different representations for words that share the same spelling but have different meanings (homonyms) such as "bank" in "river bank" and "bank balance". 

* The architecture above uses a character-level convolutional neural network (CNN) to represent words of a text string into raw word vectors
* These raw word vectors act as inputs to the first layer of biLM
* The forward pass contains information about a certain word and the context (other words) before that word
* The backward pass contains information about the word and the context after it
* This pair of information, from the forward and backward pass, forms the intermediate word vectors
* These intermediate word vectors are fed into the next layer of biLM
* The final representation (ELMo) is the weighted sum of the raw word vectors and the 2 intermediate word vectors

**ELMo vector assigned to a token or word is actually a function of the entire sentence containing that word. Therefore, the same word can have different word vectors under different contexts.**


#### Bidirectional Encoder Representations from Transformers (BERT)

BERT is at its core a transformer language model with a variable number of encoder layers and self-attention heads. **BERT does not use a decoder**.

BERT was pretrained on two tasks: language modelling (15% of tokens were **masked** and BERT was trained to predict them from context, MLM enables/enforces bidirectional learning from text by masking (hiding) a word in a sentence and forcing BERT to bidirectionally use the words on either side of the covered word to predict the masked word.) and next sentence prediction (BERT was trained to predict if a chosen next sentence was probable or not given the first sentence). As a result of the training process, BERT learns contextual embeddings for words.

BERT base is 110M (24M Embedding, 85M attention), BERT larege is 345M


## Benchmarks

**GLUE (General Language Understanding Evaluation) task set (consisting of 9 tasks)**
* CoLA: sentence grammatical or ungrammatical (Matthews correlation coefficient, correlation between predicted class vs true class regardless of class imbalance)
* SST-2: positive, negative neutral sentiment (accuracy)
* MRPC: Is sentence B a paraphrase of sentence A (binary accuracy/F1)
* STS-B: how similar are the two sentences (Pearson correlation coefficient)
* QQP: are the two questions similar (binary accuracy)
* MNLI-mm: does sentence A entail or contradict sentence B (binary accuracy)
* RTE: does sentence A entail sentence B (binary accuracy)
* QNLI: does sentence B contain the answer to sentence A (binary accuracy)
* WNLI: ambiguous pronoun replacement with noun (binary accuracy)



**SQuAD (Stanford Question Answering Dataset) v1.1 and v2.0**

SQuAD focuses on the task of question answering. It tests a model’s ability to read a passage of text and then answer questions about it. 23,215 individual paragraphs from 536 Wikipedia articles. 

* **Categories of answers.** Each answer was partitioned into one of the following categories: “date”, “other numeric”, “person”, “location”, “other entity”, “common noun phrase”, “adjective phrase”, “verb phrase”, “clause”, and “other”.
* **Reasoning required.**  different categories of reasoning required to answer thq questions: syntactic variation (synonymy, world knowledge), lexical variation, multiple sentence, ambiguity.


**SWAG (Situations With Adversarial Generations)**
Tests the task of grounded commonsense inference. 113k multiple choice questions about grounded situations. Each question is a video caption from LSMDC or ActivityNet Captions, with four answer choices about what might happen next in the scene.

#### Scoring
**Accuracy** (TP/total) bad metric if data is imbalanced
**Precision recall tradeoff:** a model that **identifies all of our positive cases** and that is at the same time **identifies only positive cases**.
**Precision** (TP/TP+FP)
**Recall** (TP/TP+FN)
**F1:** harmonic mean of precision and recall, good on imbalanced data