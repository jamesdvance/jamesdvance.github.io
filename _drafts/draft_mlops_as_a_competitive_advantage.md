---
layout: post
title: MLOps As Competitive Advantage
date: 2026-04-29 00:00:00
description: Why process, not tools, separates the winners
tags: mlops
categories: mlops
---

AWS Sagemaker was released in November, 2017. Kubeflow was open-sourced in April 2018. Neflix open-sourced Metaflow in 2019, explaining the current available tools didn't give their data scientists enough flexibility. Between 2019 and 2026, [dozens](https://github.com/kelvins/awesome-mlops) of Software as a Service, Platform As a Service, and open-source MLOps tools have been released, each promising to fix some un-addressed flaw in the machine learning process. Then, in April 2026, nearly a decade after the release of Sagemaker, Flock Safety announced [traintrack](https://www.flocksafety.com/blog/beyond-kubeflow-why-we-built-traintrack), a bazel-based ml orchestrator. 

Why after a decade of solutions are companies continuing to re-invent the ml building process and continuing to invest in ml platform and mlops engineers? Companies are by and large rational, and only invest where they see real value. The truth is simple. 

> MLOps Is a Competitive Advantage
{:.callout}






~~~~~~~~~

## Intro

Why Do So many companies hire ML Platform Engineers, when all the major cloud providers offer an ML Platform, a major open source platform exists, and dozens of SaaS companies offer the same thing? Because 



### The Holy Iteration Cycle
* Anything that takes scientists out of the research iteration cycle is wasted
* Recreating features, pulling data, writing evaluation code is all undifferentiated work
* The more scientists have to build, the more things go wrong
* Think like a scientist to understand what to abstract and what to let them modify

That's why there's constantly new tools like Hydra

### What Should Be Automated and What Shouldn't Varies By Problem Case
* A tabular ML business problem might have a  very limited 
* A Language Model training might need an efficient process to make and version changes to massive text datasets
* A reinforcement learning problem might need particularly efficient simulation process, available to many scientists
* A computer vision dataset may need different versions of labels
* Efficiently using high-price compute
* Local / remote divide

## Show Me

I can yap all day about why companies should hire more people with my skillset, but let's consider some convincing examples. 

### Trading Firm 



### References

The blog where the guy gets to #1 overall Kaggle and realized experiment tracking and efficient organization were key

##### Disclaimer: Every word in this post was written by a human