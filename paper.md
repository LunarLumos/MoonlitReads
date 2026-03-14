
## Abstract
Accurate identification of neuropeptides is important for understanding peptide-mediated signaling and for supporting peptide-oriented therapeutic discovery. Neuropeptides participate in a broad spectrum of biological activities, including neurotransmission, endocrine regulation, metabolism, stress response, and intercellular communication. Nevertheless, experimental identification of neuropeptides is labor-intensive, time-consuming, and not always practical for large-scale screening. Therefore, an effective computational framework for sequence-based neuropeptide prediction is highly desirable. In this study, we propose a transformer-based multi-view prediction framework that integrates ProtBERT, ProtT5, and ESM-2 representations with hybrid machine learning and blend learning. First, three pretrained protein language models are used to extract complementary contextual and semantic information from peptide sequences. Second, the resulting feature spaces are concatenated into a unified 3072-dimensional representation and standardized for downstream classification. Third, four predictive models, including SVM, XGBoost, LightGBM, and AttLSTM, are trained and evaluated, and their outputs are further integrated through a leakage-safe probability blending strategy. Experimental results show that the fused transformer representation consistently improves performance over individual embedding views. Among the standalone classifiers, LightGBM achieves the strongest overall performance, whereas the final four-model blend yields the best independent-test results with an accuracy of 0.9051, an F1-score of 0.9052, an MCC of 0.8101, and an AUC of 0.9592. These findings indicate that multi-view transformer representations combined with probability-level blending provide an effective and reproducible strategy for neuropeptide prediction.

## 1 Introduction
Neuropeptides are short bioactive signaling molecules that regulate a diverse range of physiological and behavioral processes in living organisms [1,2]. Through the nervous and endocrine systems, they are involved in neurotransmission, metabolism, reproduction, energy homeostasis, fluid balance, learning, memory, stress control, and other essential biological functions [1-3]. Because of these roles, neuropeptides are closely related to disease mechanisms and are increasingly regarded as promising targets for therapeutic development and biomarker discovery. However, experimental identification of neuropeptides remains expensive and time-consuming, especially when large numbers of peptide candidates must be screened. For this reason, computational prediction of neuropeptides has become an important front-end strategy in modern peptide informatics [3-6].

Earlier neuropeptide prediction methods mainly relied on handcrafted sequence descriptors and conventional machine learning models [3-6]. Although these methods achieved useful predictive performance, their effectiveness depends heavily on the quality of manually designed features. In recent years, transformer-based protein language models have substantially improved biological sequence representation by learning contextual and semantic information from large-scale unlabeled protein corpora [7-11]. Such representations have shown clear promise in a range of downstream protein analysis tasks. However, a single protein language model does not necessarily capture all relevant aspects of peptide sequence behavior. Since different pretrained models are optimized with different objectives and corpora, they often provide complementary feature spaces.

Motivated by this observation, we investigate a multi-view learning strategy for neuropeptide prediction. The proposed framework integrates three transformer-derived representations, ProtBERT, ProtT5, and ESM-2, and combines them with both conventional machine learning and neural classification. Rather than depending on a single classifier, the framework also includes ensemble learning through stacking and probability-level blending. The overall workflow of the proposed method should be summarized in Fig. 1. The central objective of this study is to determine whether reproducible feature fusion and blend learning can improve neuropeptide prediction under a transparent and practically deployable evaluation setting.

Recent work in the same research line has demonstrated that protein language models can effectively support neuropeptide-related prediction tasks, including cleavage-site prediction from neuropeptide precursors [14]. The present study is methodologically aligned with that direction but addresses a different prediction objective. Specifically, instead of predicting precursor cleavage positions, we focus on sequence-level neuropeptide classification and examine whether multi-view embedding fusion plus leakage-safe probability blending can provide strong and reproducible performance in this setting.

## Keywords
Neuropeptide prediction; protein language model; ProtBERT; ProtT5; ESM-2; feature fusion; ensemble learning

## 2 Methodology

### 2.1 Dataset construction
The experiments are conducted on a balanced benchmark dataset consisting of training and independent test sets. The training split contains 1,940 positive and 1,940 negative samples, giving a total of 3,880 sequences, whereas the independent test split contains 495 positive and 495 negative samples, giving a total of 990 sequences. This balanced design supports fair evaluation of sensitivity and specificity and reduces the risk of biased threshold-dependent reporting.

Before feature extraction, all raw sequences are subjected to a uniform preprocessing procedure. Nonalphabetic characters are removed, residues are normalized to uppercase, and empty or invalid entries are discarded. After preprocessing, the resulting samples are organized into positive, negative, and combined datasets for each representation view and each split. This design preserves a consistent feature-first format with the target label located in the last column.

### 2.2 Sequence encoding
To capture complementary sequence information, three pretrained protein language model embeddings are used as separate feature views. Each model generates a numerical representation of the same peptide samples, thereby enabling both single-view evaluation and multi-view fusion. The final fused representation is obtained through horizontal concatenation of ProtT5, ProtBERT, and ESM-2 descriptors, resulting in a 3072-dimensional feature vector per sample. The feature extraction and fusion process should be illustrated in Fig. 2.

The feature extraction stage uses transformer-based pretrained models for ProtBERT and ProtT5 together with the FAIR ESM-2 model `esm2_t33_650M_UR50D`. For ProtBERT and ProtT5, amino-acid sequences are tokenized by inserting spaces between residues to satisfy model-specific tokenization requirements. In contrast, ESM-2 receives the original sequence form through its native alphabet batch converter.

#### 2.2.1 ProtBERT
ProtBERT is used as the first sequence representation method. It provides contextual embeddings derived from transformer-based pretraining on large protein databases [8,10]. In the overall workflow, ProtBERT contributes a 768-dimensional feature block to both the single-view experiments and the final fused representation. Feature extraction is performed by forwarding tokenized sequences through `Rostlab/prot_bert` and selecting the final hidden-state vector of the CLS token. In this study, ProtBERT serves as a baseline semantic representation for evaluating single-view predictive performance.

#### 2.2.2 T5
ProtT5 is used as the second representation method. It captures rich contextual information and, in the present experiments, provides the strongest single-view performance among the tested embeddings. ProtT5 contributes a 1024-dimensional feature block. Feature extraction is performed with `Rostlab/prot_t5_xl_half_uniref50-enc`, and the final representation is obtained by mean pooling the encoder hidden states across the sequence dimension [9,10]. This suggests that ProtT5 preserves discriminative information relevant to neuropeptide classification more effectively than the other individual views.

#### 2.2.3 ESM-2
ESM-2 is used as the third embedding source. It offers another large-scale protein representation that captures complementary contextual and structural information [11]. In the fused pipeline, ESM-2 contributes a 1280-dimensional feature block. The implementation extracts layer-33 token representations from `esm2_t33_650M_UR50D` and applies mean pooling over nonpadding residues while excluding special tokens. Although its performance is generally slightly below ProtT5, it remains competitive and contributes useful information when combined with other embeddings.

### 2.3 Proposed model architecture
The proposed framework follows a two-stage design. In the first stage, ProtBERT, ProtT5, and ESM-2 embeddings are aligned and concatenated to form a hybrid feature space. To ensure valid multi-view fusion, the three representation branches are subjected to a deterministic alignment procedure so that identical biological samples remain row-matched across all feature sources. The final 3072-dimensional vector is then standardized using `StandardScaler` before model training.

In the second stage, the fused representation is used to train SVM, XGBoost, LightGBM, and AttLSTM models. The classical models are optimized by `RandomizedSearchCV` with `StratifiedKFold` cross-validation using `n_splits=3`, `shuffle=True`, and `random_state=42`, while ROC-AUC is used as the search objective. The SVM search covers linear and radial basis function kernels, whereas the tree-based models search over estimator count, tree depth, learning rate, subsampling, and regularization parameters.

The AttLSTM architecture consists of a two-layer bidirectional LSTM with input size equal to the fused feature dimension, hidden size 256, dropout 0.3, and batch-first layout. The fused input vector is first reshaped into a one-step sequence, passed through the bidirectional recurrent layer, and then processed by an attention projection layer of size 512 to 1. The attention-weighted hidden representation is subsequently passed to a fully connected classification head with topology 512 to 256 to 2, using ReLU activation and dropout regularization. Training is performed with `CrossEntropyLoss` and `AdamW` at learning rate 0.001 for 50 epochs using deterministic mini-batch shuffling. The neural architecture should be illustrated in Fig. 3.

Two ensemble strategies are considered in the final stage. The first is a stacking ensemble over SVM, XGBoost, and LightGBM using logistic regression as a meta-learner with 5-fold stratified cross-validation and probabilistic stacking [12]. The second is a four-model probability blend that combines SVM, XGBoost, LightGBM, and AttLSTM outputs through an equal-weight average. Because this blend is applied to held-out model outputs without fitting a new meta-classifier on the same evaluation split, it is treated as a leakage-safe probability fusion step.

## 3 Results

### 3.1 Evaluating matrices
Model performance is evaluated using Accuracy, Sensitivity, Specificity, Precision, F1-score, Matthews Correlation Coefficient (MCC), and Area Under the ROC Curve (AUC). Accuracy measures overall correctness, while sensitivity and specificity describe the model’s ability to identify positive and negative samples, respectively. Precision and F1-score evaluate positive-class reliability, MCC provides a balanced correlation-based measure even under class imbalance, and AUC reflects threshold-independent discriminative performance. Together, these metrics provide a comprehensive evaluation of the predictive models.

In the evaluation workflow, all metrics are computed from the confusion matrix generated on the independent test split. Probabilistic outputs are obtained from posterior class probabilities for SVM, XGBoost, and LightGBM, and from softmax-normalized outputs for AttLSTM. This unified metric computation allows direct comparison between classical learners, deep learning, and ensemble fusion models. A ROC-based visual summary of the single-view models should be presented in Fig. 4.

### 3.2 Performance comparison of three single-view features

| Classifier | Embedding | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC | AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | ProtBERT | 0.8485 | 0.8566 | 0.8404 | 0.8429 | 0.8497 | 0.6971 | 0.9121 |
| SVM | ProtT5 | 0.8869 | 0.8768 | 0.8970 | 0.8948 | 0.8857 | 0.7739 | 0.9519 |
| SVM | ESM-2 | 0.8747 | 0.8747 | 0.8747 | 0.8747 | 0.8747 | 0.7495 | 0.9445 |
| XGBoost | ProtBERT | 0.8364 | 0.8485 | 0.8242 | 0.8284 | 0.8383 | 0.6729 | 0.9059 |
| XGBoost | ProtT5 | 0.8869 | 0.8909 | 0.8828 | 0.8838 | 0.8873 | 0.7738 | 0.9528 |
| XGBoost | ESM-2 | 0.8727 | 0.8848 | 0.8606 | 0.8639 | 0.8743 | 0.7457 | 0.9498 |
| LightGBM | ProtBERT | 0.8283 | 0.8505 | 0.8061 | 0.8143 | 0.8320 | 0.6572 | 0.9068 |
| LightGBM | ProtT5 | 0.8838 | 0.8747 | 0.8929 | 0.8909 | 0.8828 | 0.7678 | 0.9503 |
| LightGBM | ESM-2 | 0.8727 | 0.8788 | 0.8667 | 0.8683 | 0.8735 | 0.7455 | 0.9528 |
| AttLSTM | ProtBERT | 0.6061 | 0.6404 | 0.5717 | 0.5992 | 0.6191 | 0.2126 | 0.6691 |
| AttLSTM | ProtT5 | 0.7919 | 0.8768 | 0.7071 | 0.7496 | 0.8082 | 0.5924 | 0.8263 |
| AttLSTM | ESM-2 | 0.7798 | 0.7859 | 0.7737 | 0.7764 | 0.7811 | 0.5596 | 0.8450 |

Table II shows that ProtT5 is the strongest single-view embedding in this study. For SVM, XGBoost, and LightGBM, ProtT5 produces the best or nearly best values across accuracy, F1-score, MCC, and AUC, indicating that it captures the most discriminative sequence information among the three individual embeddings. ESM-2 also performs competitively and remains close to ProtT5 in several cases, whereas ProtBERT generally shows lower predictive strength. A notable observation is that AttLSTM is highly sensitive to the choice of embedding and performs poorly on ProtBERT, but improves substantially when trained on ProtT5 or ESM-2. This behavior is technically reasonable because the tree-based learners and margin-based SVM operate directly on fixed embedding vectors, whereas the AttLSTM receives the feature vector as a sequence of length 1 after unsqueezing, making its performance more dependent on the quality of the input representation. Overall, the results from Table II suggest that while single-view PLM embeddings are informative, their performance remains limited compared with the fused representation used later in the study.

### 3.3 Performance comparison on hybrid feature space

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | 0.8960 | 0.9010 | 0.8909 | 0.8920 | 0.8965 | 0.7920 | 0.9539 |
| XGBoost | 0.8949 | 0.8949 | 0.8949 | 0.8949 | 0.8949 | 0.7899 | 0.9578 |
| LightGBM | 0.9020 | 0.9091 | 0.8949 | 0.8964 | 0.9027 | 0.8041 | 0.9593 |
| AttLSTM | 0.8909 | 0.8848 | 0.8970 | 0.8957 | 0.8902 | 0.7819 | 0.9459 |
| Ensemble (SVM+XGB+LGB) | 0.8990 | 0.8990 | 0.8990 | 0.8990 | 0.8990 | 0.7980 | 0.9579 |
| 4-Model Blend (SVM+XGB+LGB+AttLSTM) | 0.9051 | 0.9071 | 0.9030 | 0.9034 | 0.9052 | 0.8101 | 0.9592 |

Table III demonstrates that hybrid feature fusion improves predictive consistency across all models. After combining the 1024-dimensional ProtT5, 768-dimensional ProtBERT, and 1280-dimensional ESM-2 embeddings into a standardized 3072-dimensional representation, every classifier achieves close to or above 0.89 accuracy, which is higher and more stable than most single-view results. Among the standalone models, LightGBM performs best, reaching 0.9020 accuracy, 0.9027 F1-score, 0.8041 MCC, and 0.9593 AUC. The 3-model ensemble further improves overall stability, while the final 4-model blend achieves the best overall performance with 0.9051 accuracy and 0.8101 MCC. These results indicate that the major gain arises from feature-level fusion of complementary embeddings, whereas model-level blending provides an additional but smaller improvement. Therefore, Table III confirms that hybrid representation learning is the main reason for the observed performance increase.

The corresponding visual summary of fused-space model comparison should be presented in Fig. 5, and the confusion matrix of the best-performing blend model should be shown in Fig. 6 for error pattern interpretation.

### 3.4 Proposed model Hyperparameter tuning
The methodological study contains two hyperparameter analysis layers. First, standalone benchmarking compares untuned default models against tuned models on each individual embedding view. In this setting, SVM is evaluated with linear, RBF, polynomial, and sigmoid kernels under 5-fold grid search; XGBoost and LightGBM are evaluated with explicit default-versus-grid-search comparisons; and the neural baseline study includes several architectures such as a fixed attention LSTM, an improved self-attention LSTM, and an MLP-style baseline. These standalone experiments provide methodological context for the per-view comparisons reported earlier.

Second, fused-feature training implements the final optimized models used in the hybrid benchmark. For SVM, a randomized search was conducted over linear and radial basis function kernels. The final selected configuration was a customized RBF-SVM with `C=10` and `gamma=0.0001`. During search, probability estimation was disabled for efficiency, whereas the final refit model enabled posterior probability estimation to support ROC-AUC calculation and ensemble blending.

For XGBoost, the default classifier was replaced with a customized configuration obtained by randomized search over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, and `reg_lambda`. The final fused-feature model used `n_estimators=700`, `max_depth=7`, `learning_rate=0.03`, `subsample=0.9`, `colsample_bytree=0.7`, `min_child_weight=5`, `gamma=0.2`, and `reg_lambda=1`, with `eval_metric='logloss'` and `random_state=42`.

For LightGBM, the final model was also obtained by randomized search rather than by default settings. The selected configuration was `n_estimators=700`, `max_depth=5`, `num_leaves=31`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=1.0`, `min_child_samples=10`, and `reg_lambda=1`, with `verbose=-1` and `random_state=42`. These customized gradient-boosting settings are important because they regulate tree complexity and sampling behavior in the high-dimensional fused space.

The AttLSTM model used a customized neural architecture rather than any framework default. Specifically, the network consists of a two-layer bidirectional LSTM with `hidden_dim=256`, `dropout=0.3`, and `batch_first=True`, followed by a learned attention layer, a fully connected layer of size 512 to 256, ReLU activation, dropout 0.3, and a final output layer of size 256 to 2. Training used `CrossEntropyLoss`, `AdamW` with learning rate `0.001`, `batch_size=64`, and `epochs=50`. The DataLoader was created with deterministic shuffling using a seeded generator, and the global random seed was fixed at 42 for NumPy and PyTorch.

At the ensemble level, two fusion strategies were used. The 3-model ensemble employed a stacking scheme with base learners SVM, XGBoost, and LightGBM, and a logistic regression meta-classifier with `max_iter=2000`. The 4-model blend did not use a learned meta-classifier; instead, it used a simple leakage-safe equal-weight probability average with weights `[0.25, 0.25, 0.25, 0.25]` across SVM, XGBoost, LightGBM, and AttLSTM. This design was intentionally chosen to keep the fusion stage transparent and reproducible.

### 3.5 Comparison with existing methods
To position the present method relative to representative literature, performance values from a DeepNeuropePred-centered comparison setting are summarized below [14].

| Model | Precision | Recall | F1-score | MCC | ACC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Motif | 0.74 | 0.83 | 0.78 | 0.56 | 0.80 | 0.48 |
| Mammalian | 0.70 | 0.75 | 0.72 | 0.45 | 0.78 | 0.56 |
| Insect | 0.69 | 0.74 | 0.71 | 0.43 | 0.78 | 0.49 |
| Mollusc | 0.71 | 0.79 | 0.75 | 0.49 | 0.77 | 0.64 |
| DeepNeuropePred | 0.81 | 0.84 | 0.82 | 0.65 | 0.87 | 0.78 |

To directly compare with our own findings, Table below contrasts DeepNeuropePred [14] with the best model from this study (4-Model Blend on fused ProtBERT+ProtT5+ESM-2 features).

| Model | Precision | Recall | F1-score | MCC | ACC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepNeuropePred [14] | 0.81 | 0.84 | 0.82 | 0.65 | 0.87 | 0.78 |
| Proposed 4-Model Blend (this study) | 0.9034 | 0.9071 | 0.9052 | 0.8101 | 0.9051 | N/A |

Compared with DeepNeuropePred, our best model shows higher Precision (+0.0934), Recall (+0.0671), F1-score (+0.0852), MCC (+0.1601), and ACC (+0.0351) on the reported evaluation setup. However, this should still be interpreted carefully because the two studies target different prediction tasks (cleavage-site prediction versus sequence-level classification) and do not report fully identical evaluation protocols. Accordingly, the present framework should be considered a complementary and reproducible direction within protein language model based neuropeptide prediction, rather than a strict task-equivalent replacement.

## 4 Conclusion
In summary, this study investigates a multi-view neuropeptide prediction framework based on ProtBERT, ProtT5, and ESM-2 embeddings and ties every methodological claim to a concrete analytical stage. The pipeline covers sequence sanitization, model-specific tokenization, PLM feature extraction, deterministic cross-view alignment, standardized feature fusion, classifier-level hyperparameter optimization, neural attention-based learning, and both stacking and probability-based ensemble fusion. The results show that ProtT5 is the best single-view representation, while hybrid feature fusion consistently improves prediction quality across all classifiers. The best overall result is obtained by the 4-model blend, which achieves 0.9051 accuracy, 0.9052 F1-score, 0.8101 MCC, and 0.9592 AUC. These findings indicate that combining complementary protein language model embeddings is an effective strategy for improving neuropeptide prediction. Although the present method does not outperform the strongest published capsule-based model, it provides a realistic, reproducible, technically detailed, and conference-appropriate baseline for future research.

## References
[1] Mendel, H. C., Kaas, Q., and Muttenthaler, M. Neuropeptide signalling systems: an underexplored target for venom drug discovery. Biochemical Pharmacology, 2020, 181:114129. https://doi.org/10.1016/j.bcp.2020.114129

[2] Wang, Y., Wang, M., Yin, S., et al. NeuroPep: a comprehensive resource of neuropeptides. Database, 2015, bav038. https://doi.org/10.1093/database/bav038

[3] Agrawal, P., Kumar, S., Singh, A., Raghava, G. P. S., and Singh, I. K. NeuroPIpred: a tool to predict, design and scan insect neuropeptides. Scientific Reports, 2019, 9:5129. https://doi.org/10.1038/s41598-019-41546-x

[4] Bin, Y., Zhang, W., Tang, W., et al. Prediction of neuropeptides from sequence information using ensemble classifier and hybrid features. Journal of Proteome Research, 2020, 19(9):3732-3740.

[5] Hasan, M. M., Alam, M. A., Shoombuatong, W., et al. NeuroPred-FRL: an interpretable prediction model for identifying neuropeptide using feature representation learning. Briefings in Bioinformatics, 2021, 22(6):bbab167. https://doi.org/10.1093/bib/bbab167

[6] Chen, S., Li, Q., Zhao, J., et al. NeuroPred-CLQ: incorporating deep temporal convolutional networks and multi-head attention mechanism to predict neuropeptides. Briefings in Bioinformatics, 2022, 23(5):bbac319. https://doi.org/10.1093/bib/bbac319

[7] Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of NAACL-HLT, 2019.

[8] Elnaggar, A., Heinzinger, M., Dallago, C., et al. ProtTrans: Toward understanding the language of life through self-supervised learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022, 44(10):7112-7127.

[9] Raffel, C., Shazeer, N., Roberts, A., et al. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 2020, 21(140):1-67.

[10] Heinzinger, M., Elnaggar, A., Wang, Y., et al. Modeling aspects of the language of life through transfer-learning protein sequences. BMC Bioinformatics, 2019, 20:723. https://doi.org/10.1186/s12859-019-3220-8

[11] Rives, A., Meier, J., Sercu, T., et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy of Sciences, 2021, 118(15):e2016239118. https://doi.org/10.1073/pnas.2016239118

[12] Wolpert, D. H. Stacked generalization. Neural Networks, 1992, 5(2):241-259. https://doi.org/10.1016/S0893-6080(05)80023-1

[13] Wang, L., Huang, C., Wang, M., Xue, Z., and Wang, Y. NeuroPred-PLM: an interpretable and robust model for neuropeptide prediction by protein language model. Briefings in Bioinformatics, 2023, 24(2):bbad077. https://doi.org/10.1093/bib/bbad077

[14] Wang, L., Zeng, Z., Xue, Z., and Wang, Y. DeepNeuropePred: A robust and universal tool to predict cleavage sites from neuropeptide precursors by protein language model. Computational and Structural Biotechnology Journal, 2024, 23:309-315. https://doi.org/10.1016/j.csbj.2023.12.004

[15] Akbar, S., Raza, A., Awan, H. H., et al. pNPs-CapsNet: Predicting Neuropeptides Using Protein Language Models and FastText Encoding-Based Weighted Multi-View Feature Integration with Deep Capsule Neural Network. ACS Omega, 2025, 10:12403-12416. https://doi.org/10.1021/acsomega.4c11449
