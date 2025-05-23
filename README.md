# Graph-Based Parallel Multi-Objective Optimization of Skeletal Body Motion Data for Emotion Recognition with Knowledge-Distilled Classifier

### Recognizing human emotions from body motion is a critical challenge in affective computing, particularly in scenarios where facial expressions or speech are unavailable or unreliable. In this study, we propose a novel framework for emotion recognition from skeletal motion data using a graph-based and parallel multi-objective optimization approach. Skeletal motion sequences are represented as graphs, where nodes correspond to joints and edges capture anatomical connections, enabling the preservation of spatial structure and dynamic body patterns crucial for emotional expression. To improve both feature quality and model performance, we employ two evolutionary algorithms in parallel. A Genetic Algorithm (GA) is used to evolve the topology of the motion graphs, optimizing structural characteristics that influence expressiveness. Simultaneously, Particle Swarm Optimization (PSO) is applied to learn optimal joint-level weighting, enhancing the relevance of motion features in the frequency domain. This dual optimization process balances competing objectives, such as accuracy, graph complexity, and interpretability. After extracting graph-theoretic and frequency-domain features from the optimized representations, we train a high-performing Gradient Boosting classifier as a teacher model. To reduce computational cost while retaining predictive power, we distill this knowledge into a lightweight Decision Tree model using a hybrid of soft and hard labels. This knowledge-distilled classifier enables real-time and interpretable emotion recognition with minimal performance degradation. Experiments conducted on a multi-class skeletal emotion dataset show that our method significantly improves recognition accuracy and model efficiency compared to traditional pipelines. The proposed system offers a robust, interpretable, and scalable solution for emotion recognition in human–computer interaction, healthcare, and behavioral analysis applications.

![image](https://github.com/user-attachments/assets/cb9653b0-c858-4114-9283-019b33b0de36)
## Figure 1. The body skeleton mapped to a graph (a sample from our experiment)

![image](https://github.com/user-attachments/assets/3175f756-1993-4801-bca3-6b4536a2ba65)
## Figure 2. Samples of different emotions from the dataset (walking)

![image](https://github.com/user-attachments/assets/c2ffb19d-287d-40fd-abe2-eb10840191a9)
## Figure 3. The flow chart of the proposed method.

![image](https://github.com/user-attachments/assets/47888300-209b-4834-a3e7-1755a9f41391)
## Figure 4. The bar plot of the acquired results

#### Seyed Muhammad Hossein Mousavi
### Cyrus Intelligence Research Ltd
#### Tehran, Iran
### ORCID: 0000-0001-6906-2152 
## Contact: s.m.hossein.mousavi@cyrusai.ir and s.muhammad.hossein.mousavi@gmail.com
