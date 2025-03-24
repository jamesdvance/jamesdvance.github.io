## Build vs Buy Decisions in MLOps 

### Cloud Native
A lot of people want the ability to change cloud providers, in case pricing balances shift towards one cloud provider or another. 

### Setup Time
To Deploy Kubeflow on AWS, we must: 
1. Create an VPC
2. Create an EKS Cluster
3. Deploy Kubeflow on AWS

## Cost

### Sagemaker
Pricing is a key reason why AWS is kind enough to create a full-featured MLOps platform for its users. A 'Sagemaker' instance costs a premium against what a normal EC2-based instance costs. 

### Kubeflow (on managed Kubernetes)
An Elastic Kubernetes Engine (EKS) cluster on AWS costs $.10 per hour, or $2.40/day or $72/month *before* any servers are provisioned at additional cost. For a personal project, this is a hefty price tag. Whereas Sagemaker AI has a free tier, and promises 'you only pay for what you use'. However Amazon's EC2 P2-P4 GPU instances are available for integration with EKS, allowing GPU nodes at EC2 prices, instead of Sagemaker prices. 
Still, for personal projects, infrastructure-as-code can still improve this paradigm, as we can simply remove our resources with a simple call. Spot instances are even cheaper

## Toolkit

## Problem Fit
* Do you need to reference simulation? 

## Intangibles
* Cloud native vs agnostic
* Vender lock-in

### Tracking, Monitoring and Versioning

### What The Hell Is A Model Artifact? 
Sagemaker -  a 

### 

### Scalability 

### Maintenance L.O.E
In both cases, we want data scientists in the GUI, working on interesting problems (or they become an interesting problem). However, Sagemaker offers a well-documented experience with commensurate APIs. 

### Missing Features

### Self-Serve Ease

### Big Models

| -     | Kubeflow | Native Sagemaker | 
| ---   | ---      | ---              |
| setup time |  


#### Sagemaker Terraform

#### Kubeflow Terraform

#### Infrastructure As Code - CDK Vs Terraform in AWS
AWS CDK is a full-featured infrastructure-as-code offering from AWS. One massive advantage it has is that it integrates directly with Cloudformation, while Terraform does not. With Cloudformation, you can interact with your stacks in the AWS GUI to inspect failures, see latest changes and manually delete resources if necessary. A disadvantage of AWS CDK is its [API](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib-readme.html), which typically lags behind AWS features by a signficant time period (at the time of this writing, the AWS Sagemaker 'L2' constructs are still not available). 

Another advantage of Terraform is to be able to construct across domains, such as including New Relic monitoring and PagerDuty alerting along with your AWS setup. 


### Continuous Innovation
A crucial point about choosing a tool is ensuring one stays abreast of innovations. Given their advantageous pricing, AWS has every reason to maintain Sagemaker and keep it competitive. However, the Kubernetes ecosystem has the full support of the open source community. While Sagemaker recently introduced the unified Sagemaker AI experience, combining Lake Formation, DataZone and Sagemaker Studio into a unified tool, cutting-edge tools such as Jobset [5] continue to 

## Appendix

### References
1. [Spot Instances With EKS](https://aws.amazon.com/blogs/compute/cost-optimization-and-resilience-eks-with-spot-instances/)
2. [P3.2xlarge specs](https://aws-pricing.com/p3.2xlarge.html)
3. [Sagemaker Pricing](https://aws.amazon.com/sagemaker-ai/pricing/)
4. [EKS Pricing](https://aws.amazon.com/eks/pricing/)
5. [Jobset Unified Kubernetes API](https://jobset.sigs.k8s.io/)


### Code