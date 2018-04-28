# DeepSphere

DeepSphere is an unsupervised and end-to-end algorithm for discovering (nested) anomalies in dynamic networked systems. It is an unified method that can achieve two goals: (i) case-level anomaly detection, i.e. identifying whether a network is abnormal, (ii) nested level anomaly discovery, i.e. revealing which nodes/edges in the networks are anomalous, when anomalies occur and how they deviate from normal status. DeepSphere does not require any outlier-free (clean) or labeled data as input, it still can reconstruct normal patterns.


# Abstract

The increasing and flexible use of autonomous systems in many domains -- from intelligent transportation systems, information systems, to business transaction management -- has led to challenges in understanding the "normal" and "abnormal" behaviors of those systems. As the systems may be composed of internal states and relationships among sub-systems, it requires not only warning users to anomalous situations but also provides transparency about how the anomalies deviate from normalcy for more appropriate intervention. We propose a unified anomaly discovery framework DeepSphere that simultaneously meet the above two requirements -- identifying the anomalous cases and further exploring the cases' anomalous structure localized in spatial and temporal context. DeepSphere leverages deep autoencoders and hypersphere learning methods, having the capability of isolating anomaly pollution and reconstructing normal behaviors. DeepSphere does not rely on human annotated samples and can generalize to unseen data. Extensive experiments on both synthetic and real datasets demonstrate the consistent and robust performance of the proposed method.

# Dependency (packages)

_pickle, tensorflow, numpy, scikit-learn

# Usage

Two synthetic datasets -- "train.pkl" and "test.pkl" -- are provided. They are dictionaries which contain three components {'data':data, 'label':label, 'diff':diff}, where 'data' is a 4-dimensional tensor of shape (batch_size, time_steps, num_nodes, num_nodes), 'label' is a list of ground-truth case-level labels, and 'diff' stores all nested anomalies (including both coordinates and anomaly values).
