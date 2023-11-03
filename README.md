# Multi-Domain-Transfer-Learning

This repo includes the code of our approach of a Hierarchical Attention-based Transfer Learning model with Active Learning for Multi-Domain Sentiment Classification. First, a general model is trained on product reviews from 16 domains. Then, this model is fine-tuned on data from a specific domain. We use an Active Learning Algorithm based on Entropy Sampling and Isolation Forest (for Outlier Detection) to reduce the required labeled data. Thus, with 15% labeled data, we achieve the same performance as full supervision. The code is include in the paper: 
"Katerina Katsarou, Nabil Douss, and Kostas Stefanidis. 2023. REFORMIST: Hierarchical Attention Networks for Multi-Domain Sentiment Classification with Active Learning. In Proceedings of the 38th ACM/SIGAPP Symposium on Applied Computing (SAC '23). Association for Computing Machinery, New York, NY, USA, 919–928. https://doi.org/10.1145/3555776.3577689"

In multi-domain sentiment classification, the classifier is trained on the source domain that includes multiple domains and is tested on the target domain. It is essential to highlight that the domain in the target domain is one of the domains in the source domain. The primary assumption is that none of the domains has sufficient labeled data, which is a real-life scenario, and there is transferred knowledge among the domains. In real applications, domains are unbalanced. Some domains have much less labeled data than others, and manually labeling them would require domain experts and much time, which can induce tremendous costs. This work proposes the REFORMIST approach that uses transfer learning and is based on Hierarchical Attention with BiLSTMs while incorporating Fast-Text word embedding. The Transfer Learning approach in this work assumes that a lot of the available data is unlabeled by only selecting a portion of the domain-specific training set. Two approaches were followed for the data sampling. In the first one, the data is randomly sampled from the data pool, while the second method applies Active Learning to query the most informative observations. First, the general classifier is trained on all domains. Second, the general model transfers knowledge to the domain-specific classifiers, using the general model's trained weights as a starting point. Three different approaches were used, and the experiments showed that the sentence-level transfer learning approach yields the best results. In this approach, the transferred weights of word-level layers are not updated throughout the training, as opposed to the weights of sentence-level layers.
## Getting Started
For running the framework, this work recommends creating a new virtual environment which uses the python version 3.8.8
Afterward, you can install the packages in the requirements.txt of the requirements_files directory to get started.

**For Bash:**  

python3 -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt

For running the models, either we can do it directly from the command line or we can use the Run.py script.

## **Datasets:**

We use the datasets from the papers of Pengfei Liu, Xipeng Qiu, and Xuanjing Huang. 2017. Adversarial Multi-task Learning for Text Classification. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Vancouver, Canada, 1–10.

## **Train the General Model**

First, we need to train the general model. For that, we will use the script: train_general_script.py. With the option --bilstm=True, we select to use bilstm, and the option: --exclude is if we want to exclude a specific domain. Finally, the field "new_file_path" is the name of the trained model.

run train_general_script.py --bilstm=True --exclude="Domain_Name" -n “general_model”

## **Transfer Learning**

In The transfer_learning.py script, we can select the approach we want to use for the fine-tuning, and it can be either:
-  **-w**: word level,  where the word-level part of the Hierachical-Attention Network is retrainable, and we added a layer on top, whereas the sentence-level layers are frozen.
-  **-s**:sentence level, where we have sentence-level fine-tuning with an additional layer on the top, where the word-level part is frozen.
-  **-a**: word- and sentence-level fine tuning with adding one layer on top of both levels.
   
On the command line or the run.py script, we need to run this command:

    python transfer_learning.py -i /path/to/general_model -o output_file_name -a S

    *  -i:  the input model path
    *  -o:  the output file name
    *  -a: the fine-tuning approach (it can be either W, S, or A, as we mentioned above).



## **Use the model with Active Learning**

Next, we will proceed with transfer learning and the active learning algorithm. For that, we will run the active_learning.py script:


run active_learning.py -i ./path/to/general_model -o \"output_file_name\" -a S -m entropy --use-outlier-detection=True

   *  -i:  the input model path
   *  -o:  the output file name
   * -a: the fine-tuning approach (it can be either W, S, or A, as we mentioned above)
   * -m: the metric we want to use (entropy or uncertainty)
   * --use-outlier-detection: if we want to use outlier detection or not 
     
