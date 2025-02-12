# Emotion Recognition using LLMs

## Overview
This repository contains code and resources for emotion recognition from text using Large Language Models (LLMs). 
The goal is to classify text into 13 different emotions using various LLMs and compare their performance with traditional NLP approaches like word embeddings.

## Models Used
* LLaMA (Meta)
* EmoLLaMA (Emotion-specialized LLaMA models)
* BERT
* BloomZ
* Vicuna

* Model sizes range from 1B to 13B parameters.
* Models were run locally using LM Studio

## Dataset
* 40,000 tweets labeled with 13 emotions
* Unbalanced dataset (some emotions appear more frequently than others)

## Evaluation Metrics
* Balanced Accuracy (primary metric)
* Accuracy, Precision, F1-score
* Confusion Matrices for performance analysis

## Results
* Best LLM: BERT-7B
* Larger models generally performed better within the same family
* No model achieved human-level performance
* Fusion of LLM outputs (Decision-Level Fusion) slightly improved results

## Requirements
* Python 3.10 or newer
* Required packages (install with pip):
    _pip install -r requirements.txt_

## Contributors
Created by Athena Pnevmatikaki as part of a thesis project in Emotion Recognition using NLP.
