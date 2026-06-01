---
layout: post
title:  Agents, Agents Everywhere
date: 2026-01-01 00:00:00
description: Levels of Orchestration for multi-purpose agents
tags: agents, orchestration, llmops
categories:  agents
featured: true
---

For this exploration, I'll use a multi-agent meal planning service. A person can bring up a chat window and generate a meal plan for a family of *n*, get an individualized daily food schedule optimized for nutrition and energy levels, create a shopping list, schedule a grocery pick-up, and incorporate eating out and changed plans. This is a handy example because I've already done some development in this space long before LLMs came along, but also because a single-agent workflow could theoretcally handle the process, while at the same time the space could support both parallel and collaborative workflows, so a multi-agent setup. 

### Level 0: No Orchestration: a hard-coded workflow
An agent has a prompt, a set of tools and some workflow graph that it follows. Given the (lack of reliability) in current agent workflows, this should be the norm for most deployments. 

### Level 1: Orchestrator Agent With Sub-Agents as Tools

The obvious routing preference is to have a 'lead agent'. 

Agent orchestration / routing is a piece of work around 
Random thoughts: 
* Managing context for agents in the form of state, memory detail, available tools, hooks and callbacks are a form of knapsack problem where we know we want to minimize context such that we we stay within effective bounds
* Routing is a problem that has to be done right. 
* Long-running tasks are 

Classifier Model Vs Agent 


[1]
* Plain rules engines are not effective agent orchestrators, because the underlying business variables are messy, probabalistic and always changing. Plus ML already exists 
* Uses a router agent to field incoming requests and determine where they need to go 
* Agent has tools 

[2]
* Use action schemas to solidify what can and can't be done 

[3] 
* Multi-agents ar handy in research where you can't predict the states in advance. It can't be done with a single hard-coded workflow 
* Research in particular requires going down rabit holes and spending longer than expected on certain tasks
* 

[4]
* Uses an NLP classifier to judge intent and route tasks

[5] 
* 

[6] 
* Shows the importance of incorporating real-time feedback and understanding when human intervention is necessary

## Appendix

### 1. Resources:
* [1. Multi Agent Architecture for Advertising at Spotify](https://engineering.atspotify.com/2026/2/our-multi-agent-architecture-for-smarter-advertising?utm_source=engineering.fyi)
* [2. How to Engineer Multi-Agent workflows (Github)](https://github.blog/ai-and-ml/generative-ai/multi-agent-workflows-often-fail-heres-how-to-engineer-ones-that-dont/)
* [3. How We Built Our Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
* [4. Agent Squad](https://github.com/2FastLabs/agent-squad)
* [5. Multi-Agent Collaboration via Evolving Orchestration](https://arxiv.org/pdf/2505.19591)
* [6. Backpressure is All You Need](https://www.lucasfcosta.com/blog/backpressure-is-all-you-need)
* [7. Multi-agent stack reddit thread](https://www.reddit.com/r/AI_Agents/comments/1sxdisu/whats_your_stack_for_building_multiagent_workflows/)

### 2. Definitions:
1. Adaptive Thinking:
2. Interleaved Thinking: 