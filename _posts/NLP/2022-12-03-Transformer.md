---
layout: post
title: Transformer
category: nlp
---

# Transformer

The Transformer – a model that uses attention to boost the speed with which these models can be trained. The biggest benefit, comes from how The **Transformer lends itself to parallelization**.

Encoding component, a decoding component, and connections between them. The encoding component is a stack of encoders, each encoder is a  self-attention layer (a layer that helps the encoder look at other words in the input sentence as it encodes a specific word) + feed-forward layer. 

### Self-attention

As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word. Self-attention is the method the Transformer uses to bake the “understanding” of other relevant words into the one we’re currently processing.

The **first step** in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in this case, the embedding of each word). So for each word, we create a **Query vector, a Key vector, and a Value vector**. These vectors are created by multiplying the embedding by three matrices that we trained during the training process. Notice that these new vectors are smaller in dimension than the embedding vector. Their dimensionality is 64, while the embedding and encoder input/output vectors have dimensionality of 512. 

The **second step** in calculating self-attention is to calculate a **score**. The **score is calculated by taking the dot product of the query vector with the key vector** of the respective word we’re scoring. 

The **third and fourth steps** are to **divide the scores by 8** (the square root of the dimension of the key vectors used in the paper – 64. This leads to having more stable gradients. There could be other possible values here, but this is the default), then pass the result through a softmax operation. Softmax normalizes the scores so they’re all positive and add up to 1. This softmax score determines how much each word will be expressed at this position. Clearly the word at this position will have the highest softmax score, but sometimes it’s useful to attend to another word that is relevant to the current word.

The **fifth step** is to **multiply each value vector by the softmax score** (in preparation to sum them up). The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001, for example).

The **sixth step** is to **sum up the weighted value vectors**. This produces the output of the self-attention layer at this position (for the first word).

### “multi-headed” attention

1. It expands the model’s ability to focus on different positions. Yes, in the example above, z1 contains a little bit of every other encoding, but it could be dominated by the actual word itself. If we’re translating a sentence like “The animal didn’t cross the street because it was too tired”, it would be useful to know which word “it” refers to.

1. It gives the attention layer multiple “representation subspaces”. As we’ll see next, with multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices (the Transformer uses eight attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.


## Representing The Order of The Sequence Using Positional Encoding