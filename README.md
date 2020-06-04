# `marynlp`
The first package to introduce swahili to NLP. Developed and maintained by [Inspired Ideas](http://inspiredideas.io/)


## About
`marynlp` is a package allows the use of NLP for swahili language including tasks.

## Package Overview

------
The package consists of the following components

|module|explanation|
|------|-----------|
|marynlp.data| Module contains anything that involves reading or manipulating data |
|marynlp.data.readers| Module contains different types of data readers |
|marynlp.data.transformers| Module contains objects that are responsible for transforming text from one form to another. This is handy when used with `marynlp.data.readers` as it helps in preprocessing the data as it is being red |
|marynlp.modules| Contains pretrained modules that are used for different NLP Tasks |

## 
 - Embeddings: These embeddings are included for both `static` embeddings. 
 - Named Entity Recognition Tagging
 - Parts-of-speech Tagging
 - Sentimental Analysis
 - Text Classification

## Installation

`marynlp` uses python `3.7.4` and can be setup using a `conda` environment. If you haven't created a virtual environment, open up your terminal and type in
```
conda create --name <env-name> python==3.7.4
```
then install the packages listen in the `requirements.txt` file using
```
pip install -r requirements.txt
```
