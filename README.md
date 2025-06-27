# LLM-Based recommender system: prompting experiments

This repository contains all code and scripts used for the thesis experiments on large language model-based recommender systems, focusing on the evaluation of Zero-Shot, Few-Shot, and Chain-of-Thought (CoT) prompting across different scenarios and datasets.

## Features

- User filtering and cohort preparation scripts
- Prompt construction for each strategy (Zero-Shot, Few-Shot, CoT)
- Experiment orchestration for standard and cold-start conditions
- Metric computation (Recall@K, Precision@K, NDCG@K, Hit@K, Gini, ItemCV)
- Reproduction of all reported results and figures

## Data

Download the MovieLens 100K or 1M datasets from [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/).  
Data files are not included due to licensing.

## API credentials

API access keys/endpoints are not included for security. Obtain your own credentials from the LLM provider.

## Usage

Clone the repository and install requirements to get started.
