---
layout: post
title:  Re-Ranking After RAG
date: 2025-08-25 00:00:00
description: Making Every Context Token Count
tags: rag
categories:  agents
featured: true
---

Re-Rankers are functions that run after a RAG process returns results. Because of context constraints, RAG applications are critical to providing agents with the best context to complete their task. But the similarity search underlying RAG is far from perfect. Two chunks of text can be embedded similarly but still provide far different meanings: 

"The function will sort the user data by timestamp before displaying it."

vs

"The function will store the user data by timestamp before displaying it."

At a high-level our re-ranker fits into this simple RAG chatbot:
<img src="rerankers/RAG_Reranker.drawio.png" />


## Shifting From Recall To Precision 
The embeddings similarity search focuses on Recall - its goal is to maximize the largest percentage of the total number of relevant chunks in the database that are returned. In simpler terms, it wishes to maximize the True Positives. The re-ranker conversely optimizes for precision. Of the results returned to the agent, the re-ranker wants to maximize the % that are relevant. It aims to minimize False Positives. In doing so, re-rankers are critical to ensuring agents avoid unnecessary noise and [context rot](https://research.trychroma.com/context-rot). 

## The Downsides
Before diving into implementation, let's acknowledge some risks associated with adding a re-ranking function. First, re-rankers add substantial latency to the RAG application. The initial vector search can be optimized down to a single-millisecond operation. On the other hand, re-ranking models (especially the ones we explore below) can take in the 100's of ms up to several seconds. Additionally, if using re-ranker LLMs or even hosting a Cross-Encoder mentioned below, cost-per-query rises dramatically with a reranker included. For many applications, getting more precise results is worth the extra cost - and there are mitigation techniques as well. For instance, re-rankers can 'cascade' from fast/cheap to filter the intial results down and then 'slow/expensive' used to groom the final list. 

## Datasets

To train and optimize a re-ranker requires a ranking dataset. A pairwise ranking dataset (1 query 1 answer) is the simplest to generate so we'll focus on that here. 

#### Hand-Labeled
Hand-labeled datasets are accurate usually noise-less but who has the time? 

#### Artificially Labeled
Often you can create labels for a dataset artificially, by using metadata like tags, or topic modeling to then rate 'true' or 'false' for a given search query. 

#### LLM-Generated
LLM's can generate a ranking datset. In the simplest way you could prompt an LLM 'give me a search query that would result in this item'. But in more sophisticated use cases, we would provide the LLM with categories, tags and any other topic metadata for it to make a relevant but somewhat vaque query. 

#### Application-Generated
In a real-world application, re-ranking datasets can be generated from production data. In a RAG use case, we can retroactively label a results' relevance by checking whether it was actioned on by an agent (in a purely agentic workflow) or mentioned by the user in a subsequent query (in the case of a chatbot). 

## Common Models

### Cross-Encoder
A highly common model used in re-ranking tasks for RAG is the Cross Encoder. The Cross Encoder ouputs

### Ranking-Specialized LLMs

### BiEncoder



## Example: A Kaggle Assistant

## 

## Self-Improvment Mechanisms