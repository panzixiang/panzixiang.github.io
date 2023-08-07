---
layout: post
title: Text Similarity
category: nlp
---

Text similarity has to determine how ‘close’ two pieces of text are both in surface closeness **lexical similarity** and meaning **semantic similarity**.

Since differences in word order often go hand in hand with differences in meaning (compare the ``dog bites the man`` with ``the man bites the dog``), we'd like our sentence embeddings to be sensitive to this variation.

## Big idea
The big idea is that you **represent documents as vectors of features**, and compare documents by measuring the distance between these features. 

<!--more-->

### Embeddings + similarity

List of embeddings
```
- Bag of Words (BoW)
- Term Frequency - Inverse Document Frequency (TF-IDF)
- Continuous BoW (CBOW) model and SkipGram model embedding(SkipGram)
- Pre-trained word embedding models : 
     ->  Word2Vec (by Google)
     ->  GloVe (by Stanford)
     ->  fastText (by Facebook)
- Poincarré embedding
- Node2Vec embedding based on Random Walk and Graph
```
There are two main difference between BoW or TF-IDF in keeping with word embeddings:

* BoW or TF-IDF create **one number per word** while word embeddings typically creates **one vector per word**.
* BoW or TF-IDF is good for **classification documents as a whole**, but word embeddings is good for **identifying contextual content**

#### Similarity measures
<details>
<summary>Bad ones</summary>

#### Bad ones
1. **Jaccard similarity** or **intersection over union** is defined as size of intersection divided by size of union of two sets. If two sentences have no common words and will have a Jaccard score of 0. This is a terrible distance score because the 2 sentences have very similar meanings. Here Jaccard similarity is neither able to capture semantic similarity nor lexical semantic of these two sentences.Moreover, this approach has an inherent flaw. That is, as the **size of the document increases, the number of common words tend to increase even if the documents talk about different topics**.

2. **K-means and Hierarchical Clustering Dendrogram**
   1) Use BoW, TFIDF to convert sentences into vectors.
   2) Use vector clustering of sentences as a proxy for similarity

    It can be noted that k-means (and minibatch k-means) are very sensitive to feature scaling and that in this case the IDF weighting helps improve the quality of the clustering. **Reducing the dimensionality of our document vectors by applying latent semantic analysis will be the solution**.

    However to overcome this big issue of dimensionality, there are measures such as V-measure and Adjusted Rand Index wich are information theoretic based evaluation scores: as they are only based on cluster assignments rather than distances, hence not affected by the curse of dimensionality.

    Note: as k-means is optimizing a non-convex objective function, it will likely end up in a local optimum. Several runs with independent random init might be necessary to get a good convergence.
</details>

#### Okay ones
1. **Cosine similarity**
    **CountVectorizer**:  It is used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text.
    The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance (due to the size of the document), chances are they may still be oriented closer together. The smaller the angle, higher the cosine similarity.
2. Improve Cosine similarity using **Smooth Inverse Frequency (SIF)**
    Taking the average of the word embeddings in a sentence (as we did just above) tends to give too much weight to words that are quite irrelevant, semantically speaking. Smooth Inverse Frequency tries to solve this problem in two ways:

    * Weighting: SIF takes the weighted average of the word embeddings in the sentence. Every word embedding is weighted by a/(a + p(w)), where a is a parameter that is typically set to 0.001 and p(w) is the estimated frequency of the word in a reference corpus.
    * Common component removal: SIF computes the principal component of the resulting embeddings for a set of sentences. It then subtracts from these sentence embeddings their projections on their first principal component. This should remove variation related to frequency and syntax that is less relevant semantically.
    SIF downgrades unimportant words such as but, just, etc., and keeps the information that contributes most to the semantics of the sentence.
3. Use Cosine similarity using **Latent Semantic Indexing (LSI)**
    LSI is a dimensional reduction technique part of latent semantic analysis(LSA). It is often assumed that **the underlying semantic space of a corpus is of a lower dimensionality than the number of unique tokens**. Therefore, LSA applies principal component analysis on our vector space and only keeps the directions in our vector space that contain the most variance (i.e. those directions in the space that change most rapidly, and thus are assumed to contain more information). This is influenced by the ``num_topics`` parameters we pass to the ``LsiModel`` constructor.
    LSI use the term frequency as term weights and query weights.
    Steps:
        1. Set term weights and construct the term-document matrix ``A`` and query matrix ``q``
        2. Decompose matrix ``A`` matrix and find the ``U``, ``S`` and ``V`` matrices, where ``A`` = ``U`` ``S`` ``(V)^T``
        3. Implement a Rank 2 Approximation by keeping the first two columns of ``U`` and ``V`` and the first two columns and rows of ``S``.
        4. Find the new document vector coordinates in this reduced 2-dimensional space. Rows of ``V`` holds eigenvector values.
        5. Find the new query vector coordinates in the reduced 2-dimensional space.These are the new coordinate of the query vector in two dimensions. Note how this matrix is now different from the original query matrix q given in Step 1.
        6. Rank documents in decreasing order of query-document cosine similarities
4. **Word movers distance (WMD)**
    Word Mover’s Distance solves this problem (2 sentences have no common words and will have a cosine distance of 0) by taking account of **the words' similarities in word embedding space**.
    WMD uses the word embeddings of the words in two texts to measure the minimum distance that the words in one text need to “travel” in semantic space to reach the words in the other text.
    WMD uses the word embeddings of the words in two texts to measure the **minimum distance that the words in one text need to "travel in semantic space to reach the words in the other text**.
5. **Latent Dirichlet Allocation (LDA)** with Jensen-Shannon distance
    Latent Dirichlet Allocation (LDA), is an unsupervised generative model that assigns topic distributions to documents.
    At a high level, the model assumes that **each document will contain several topics, so that there is topic overlap within a document**. The words in each document contribute to these topics. The topics may not be known a priori, and needn’t even be specified, but the number of topics must be specified a priori. Finally, there can be words overlap between topics, so several topics may share the same words.
    The model generates to latent (hidden) variables :

        A distribution over topics for each document (1)
        A distribution over words for each topics (2)

    After training, **each document will have a discrete distribution over all topics, and each topic will have a discrete distribution over all words**.

    Now we have **a topic distribution for a new unseen document**. The goal is to find the **most similar documents in the corpus**.

    To do that we compare the topic distribution of the new document to all the topic distributions of the documents in the corpus. We use the Jensen-Shannon distance metric to find the most similar documents.

    What the Jensen-Shannon distance tells us, is which documents are statisically “closer” (and therefore more similar), by comparing the divergence of their distributions.

    Jensen-Shannon is a method of measuring the similarity between two probability distributions. It is also known as information radius (IRad) or total divergence to the average

    Jensen-Shannon is symmetric, unlike Kullback-Leibler on which the formula is based. This is good, because we want the similarity between documents A and B to be the same as the similarity between B and A.

#### Statistical methods
1. **Variational Autoencoder (VAE)**
    The job of those models is to predict the input, given that same input. More specifically, let’s take a look at Autoencoder Neural Networks. This Autoencoder tries to learn to approximate the following identity function: $f_{W,b}(x) \approx x$
    This can be done by limiting the number of hidden units in the model. Those kind of autoencoders are called undercomplete. Autoencoder learns an compressed representation of the input

    The key problem will be to **obtain the projection of data in single dimension without losing information**. When this type of data is projected in latent space, a lot of information is lost and it is almost impossible to deform and project it to the original shape. No matter how much shifts and rotation are applied, original data cannot be recovered.

    So how does neural networks solves this problem ? The intuition is, **in the manifold space, deep neural networks has the property to bend the space in order to obtain a linear data fold view**. Autoencoder architectures applies this property in their hidden layers which allows them to learn low level representations in the latent view space.

    Autoencoders are trained in an **unsupervised manner** in order to learn the exteremely low level repersentations of the input data.

    A typical autoencoder architecture comprises of three main components:

    * Encoding Architecture : The encoder architecture comprises of series of layers with decreasing number of nodes and ultimately reduces to a latent view repersentation.
    * Latent View Repersentation : Latent view repersents the lowest level space in which the inputs are reduced and information is preserved.
    * Decoding Architecture : The decoding architecture is the mirro image of the encoding architecture but in which number of nodes in every layer increases and ultimately outputs the similar (almost) input.

    **It encodes data to latent (random) variables, and then decodes the latent variables to reconstruct the data.**

    Rather than directly outputting values for the latent state as we would in a standard autoencoder, the encoder model of a **VAE will output parameters describing a distribution for each dimension in the latent space.** Since we’re **assuming that our prior follows a normal distribution**, we’ll output two vectors describing the **mean and variance of the latent state distributions**. If we were to build a true multivariate Gaussian model, we’d need to define a **covariance matrix describing how each of the dimensions are correlated**. However, we’ll make a simplifying assumption that our covariance matrix only has nonzero values on the diagonal, allowing us to describe this information in a simple vector.

   **Our decoder model will then generate a latent vector by sampling from these defined distributions and proceed to develop a reconstruction of the original input.**

    In normal deterministic autoencoders the latent code does not learn the probability distribution of the data and therefore, it’s not suitable to generate new data.

    The VAE solves this problem since it explicitly defines a probability distribution on the latent code. In fact, **it learns the latent representations of the inputs not as single points but as soft ellipsoidal regions in the latent space, forcing the latent representations to fill the latent space rather than memorizing the inputs as punctual, isolated latent representations.**

    Using a VAE we are able to fit a parametric distribution (in this case gaussian). This is what differentiates a VAE from a conventional autoencoder which relies only on the reconstruction cost. This means that during run time, when we want to draw samples from the network all we have to do is generate random samples from the Normal Distribution and feed it to the encoder P(X|z) which will generate the samples.

    Our goal here is to use the **VAE to learn the hidden or latent representations of our textual data** — which is a matrix of Word embeddings. We will be using the VAE to map the data to the hidden or latent variables. We will then visualize these features to see if the model has learnt to differentiate between documents from different topics. We hope that similar documents are closer in the Euclidean space in keeping with their topics. **Similar documents are next to each other.**

1. **Pre-trained sentence encoders**

    The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.

    **Pre-trained sentence encoders aim to play the same role as word2vec and GloVe, but for sentence embeddings**: the embeddings they produce can be used in a variety of applications, such as text classification, paraphrase detection, etc. Typically they have been trained on a range of supervised and unsupervised tasks, in order to capture as much universal semantic information as possible.

    The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is variable length English text and the output is a 512 dimensional vector. We apply this model to the STS benchmark for semantic similarity, and the results can be seen in the example notebook made available. The universal-sentence-encoder model is trained with a deep averaging network (DAN) encoder.

    Several such encoders are available :

    * **InferSent (Facebook Research)** : BiLSTM with max pooling, trained on the SNLI dataset, 570k English sentence pairs labelled with one of three categories: entailment, contradiction or neutral.
    * **Google Sentence Encoder** : a simpler Deep Averaging Network (DAN) where input embeddings for words and bigrams are averaged together and passed through a feed-forward deep neural network.

1. **Siamese Manhattan LSTM (MaLSTM)**

    **Siamese networks are networks that have two or more identical sub-networks in them. Siamese networks seem to perform well on similarity tasks and have been used for tasks like sentence semantic similarity, recognizing forged signatures and many more.**

    The names MaLSTM and SiameseLSTM might leave an impression that there are some kind of new LSTM units proposed, but that is not the case.

    Siamese is the name of the general model architecture where the model consists of two identical subnetworks that compute some kind of representation vectors for two inputs and a distance measure is used to compute a score to estimate the similarity or difference of the inputs. In the attached figure, the LSTMa and LSTMb share parameters (weights) and have identical structure. The idea itself is not new and goes back to 1994.

    MaLSTM (Manhattan LSTM) just refers to the fact that they chose to use Manhattan distance to compare the final hidden states of two standard LSTM layers. Alternatives like cosine or Euclidean distance can also be used, but the authors state that: “Manhattan distance slightly outperforms other reasonable alternatives such as cosine similarity”.

    Also, you can trivially swap LSTM with GRU or some other alternative if you want.
    It projects data into a space in which similar items are contracted and dissimilar ones are dispersed over the learned space. It is computationally efficient since networks are sharing parameters.

    Siamese network tries to contract instances belonging to the same classes and disperse instances from different classes in the feature space.

1. **BERT with cosine distance**

    The language modeling tools such as ELMO, GPT-2 and BERT allow for **obtaining word vectors that morph knowing their place and surroundings.**
    The base case BERT model that we use here employs 12 layers (transformer blocks) and yields word vectors with p = 768. The word embeddings are pulled from the 11th layer.

   **BERT embeddings are contextual.** Each row show three sentences. The sentence in the middle expresses the same context as the sentence on its right, but different from the one on its left. All three sentences in the row have a word in common. The numbers show the computed cosine-similarity between the indicated word pairs. BERT embedding for the word in the middle is more similar to the same word on the right than the one on the left.

    **When classification is the larger objective, there is no need to build a BoW sentence/document vector from the BERT embeddings.** The [CLS] token at the start of the document contains a representation fine tuned for the specific classification objective. But for a clustering task we do need to work with the individual BERT word embeddings and perhaps with a BoW on top to yield a document vector we can use.