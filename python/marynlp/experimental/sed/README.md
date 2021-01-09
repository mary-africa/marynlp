# SED NLP Library

## Introduction

The SED library is a specialized Natural Language Processing Library designed to cater to NLP tasks for Bantu Languages with Swahili as our first case-study. The library consists of packages for preprocessing, morphology analysis, as well as modeling of data for text-related Swahili NLP tasks. The algorithms are based off of extensive research carried out on the Swahili language including the String Edit Distance (SED) algorithm for morphology analysis which is the foundation on which most of the library is built on. Below are guidelines on how to make use of the various functionalities of the library and its components structured as follows:

The first section focuses on the morphology analyzer and its core functions as well as its role in downstream tasks. The following section details the tokenizer that makes use of the morphemes obtained from the analyzer to break down words in accordance to rules of the Swahili language. Section 3 focuses on the embeddings that utilize tokens from the tokenizer to encode information from words to vectors in a manner that is representative of the Swahili language. Finally section four describes the models used in downstream tasks such as sentiment analysis and how information from the embeddings is aggregated and used to learn and make inference from data. 

The development and production of this library has helped support just about 10 jobs. Many thanks to all who participated (go team Mary whoo-hoo!) in making this project possible. 

## Morphology Analyzer
At the heart of the SED library is the morphology analysis of the Swahili language, Swahili verbs in particular, which allows the algorithms to decipher the underlying structure of the language and make use of this information in downstream tasks such as language modelling and sentiment analysis.

The morphology analysis algorithm is based primarily on the String Edit Distance (ergo SED) otherwise known as the Levenshtein distance through which words are broken down to basic units of information (morphemes) based on their similarities. 

The algorithm is adapted from https://www.researchgate.net/publication/228566510_Refining_the_SED_heuristic_for_morpheme_discovery_Another_look_at_Swahili where the SED algorithm is utilized to design a heuristic for analyzing the morphology of the Swahili language. The analyzer goes beyond research from the paper to developing an algorithm that breaks down a set of words into morphemes. 

## Tokenizer
The SED tokenizer is built on top of the morphology analyzer utilizing the SED heuristic by making use of the morphemes obtained from analysis of sample words to break down words from corpora or datasets that are to be used in tasks such as language modelling sentiment analysis.

For many of the verb forms, the tokenizer is capable of effectively breaking them down, however, since the SED heuristic is less effective at analyzing words other than verbs, the tokenizer has been hard-coded with rules specific to the Swahili language to enable it to detect and properly break down various word types including nouns and noun-like verbs. 

## Embeddings
Owing to the agglutinative nature of the Swahili language, the embedding algorithm encodes information from text data through a two step procedure where morphemes of words, obtained from the tokenizer, are first encoded into a lower dimensional vector space, afterwhich the word embeddings are composed from the morpheme vectors using one of several composition functions.

The embedding algorithm can be used as either part of downstream tasks through which information from text data is encoded on-the-fly and the embedding layer trained along with the rest of the model, or as a standalone algorithm where embeddings are encoded separately from a larger corpus and used through transfer learning in downstream tasks.

Along with encoding information from text data, the embedding algorithm includes [word_similarity] and [word_analogy] methods that allow for checking how well information is encoded in the embedding space. 

## Sentiment Analysis Model
Part of the SED library are several NLP models that make use of the information extracted and encoded using the tokenizer and embeddings to further the task of Natural Language Understanding for Swahili. 

The first of such models is the Sentiment Analysis Model (SAM) that is trained and finetuned on about 2000 data points. The model uses a relatively simple Bidirectional Gated Recurrent Unit(GRU) along with the SED embeddings algorithm, fitted with a Recurrent Neural Network composition function. 

Results from SAM average around 0.82 accuaracy and F1-score with the a 0.7/0.3 train-test data split. The training data is further split to have a 0.7/0.3 train-validation split that was used to finetune the model and monitor overfitting.
