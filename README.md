# Multi-Domain-Transfer-Learning

This repo includes the code of our approach of a Hierarchical Attention-based Transfer Learning model with Active Learning for Multi-Domain Sentiment Classification. First, a general model is trained on product reviews from 16 different domains. Then, this model is fine-tuned on data from a specific domain. We use an Active Learning Algorithm based on Entropy Sampling and Isolation Forest (for Outlier Detection) to reduce the amount of required labeled data. Thus, with 15% labeled data we achive the same performance as with full supervision.   
## Getting Started
For running the framework, this work recommends creating a new virtual environment which uses the python version 3.8.8
Afterwards, install the packages in the requirements.txt of the requirements_files directory to get started.

**For Bash:**  

python3 -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt

For running the models either we can do it directly from the command line or we can use the Run.py script.

**Datasets:**

We use the datasets from the paper of Pengfei Liu, Xipeng Qiu, and Xuanjing Huang. 2017. Adversarial Multi-task Learning for Text Classification. In Proceedings of the 55th Annual Meeting of theAssociation for Computational Linguistics (Volume 1: Long Papers). Associationfor Computational Linguistics, Vancouver, Canada, 1–10.

**Train the General Model**

First we need to train the general model. For that we will use the script: train_general_script.py. 

**Transfer Learning**


**Use the model with Active Learning**
