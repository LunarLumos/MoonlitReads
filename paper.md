# Multi-Embedding Fusion for Peptide Classification: ProtBERT, T5, and ESM-2

## Abstract

Reliable peptide classification requires both expressive sequence representation and robust supervised learning. This study evaluates three pretrained protein language model embeddings, ProtBERT, T5, and ESM-2, using SVM, XGBoost, LightGBM, and AttLSTM. We then concatenate the three embeddings into a single fused feature space and train the same classifiers, followed by model-level fusion using a three-model ensemble and a four-model blend. Across single-embedding experiments, T5 provides the most consistent performance. On fused embeddings, all base models improve and become more balanced. The best overall result is obtained by the four-model blend (SVM+XGB+LGB+AttLSTM), with 0.9051 accuracy, 0.9052 F1, 0.8101 MCC, and 0.9592 AUC. Findings indicate that feature-level fusion is the primary source of improvement, while model-level fusion yields additional but moderate gains.

## Keywords

Peptide classification, protein language models, ProtBERT, T5, ESM-2, feature fusion, ensemble learning, bioinformatics

## 1. Introduction

Peptide classification is a central task in computational biology, supporting applications such as therapeutic peptide discovery and functional annotation. Traditional descriptor-based machine learning can be limited by manual feature engineering. Recent protein language models provide dense contextual embeddings that capture sequence-level semantics and biochemical patterns.

Different embedding models are trained with different objectives, so they encode complementary information. This motivates multi-embedding fusion, where several embeddings are combined into one representation for downstream learning. In parallel, classifier diversity can be exploited through model-level fusion.

This work evaluates both ideas under a consistent pipeline:

1. Single-embedding evaluation (ProtBERT, T5, ESM-2).
2. Feature-level fusion (ProtBERT+T5+ESM-2).
3. Model-level fusion (3-model ensemble and 4-model blend).

## 2. Materials and Methods

### 2.1 Dataset

The dataset includes a training split and an independent test split with balanced classes.

### Table I. Number of samples in training and independent test sets

| Dataset | Positive | Negative | Total |
| --- | ---: | ---: | ---: |
| Training | 1,940 | 1,940 | 3,880 |
| Test (Independent) | 495 | 495 | 990 |

### 2.2 Embeddings and Classifiers

Embeddings:

1. ProtBERT
2. T5
3. ESM-2

Classifiers:

1. SVM
2. XGBoost
3. LightGBM
4. AttLSTM

Model fusion on combined features:

1. Ensemble (SVM+XGB+LGB)
2. 4-Model Blend (SVM+XGB+LGB+AttLSTM)

### 2.3 Evaluation Metrics

Performance is reported using:

1. Accuracy
2. Sensitivity
3. Specificity
4. Precision
5. F1-score
6. MCC
7. AUC

AUC and MCC are emphasized as strong indicators of discriminative ability and balanced prediction quality.

## 3. Results

### 3.1 Single-Embedding Performance

### Table II. Comparison of classifiers using different embedding techniques

| Classifier | Embedding | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC | AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | ProtBERT | 0.8485 | 0.8566 | 0.8404 | 0.8429 | 0.8497 | 0.6971 | 0.9121 |
| SVM | T5 | 0.8869 | 0.8768 | 0.8970 | 0.8948 | 0.8857 | 0.7739 | 0.9519 |
| SVM | ESM-2 | 0.8747 | 0.8747 | 0.8747 | 0.8747 | 0.8747 | 0.7495 | 0.9445 |
| XGBoost | ProtBERT | 0.8364 | 0.8485 | 0.8242 | 0.8284 | 0.8383 | 0.6729 | 0.9059 |
| XGBoost | T5 | 0.8869 | 0.8909 | 0.8828 | 0.8838 | 0.8873 | 0.7738 | 0.9528 |
| XGBoost | ESM-2 | 0.8727 | 0.8848 | 0.8606 | 0.8639 | 0.8743 | 0.7457 | 0.9498 |
| LightGBM | ProtBERT | 0.8283 | 0.8505 | 0.8061 | 0.8143 | 0.8320 | 0.6572 | 0.9068 |
| LightGBM | T5 | 0.8838 | 0.8747 | 0.8929 | 0.8909 | 0.8828 | 0.7678 | 0.9503 |
| LightGBM | ESM-2 | 0.8727 | 0.8788 | 0.8667 | 0.8683 | 0.8735 | 0.7455 | 0.9528 |
| AttLSTM | ProtBERT | 0.6061 | 0.6404 | 0.5717 | 0.5992 | 0.6191 | 0.2126 | 0.6691 |
| AttLSTM | T5 | 0.7919 | 0.8768 | 0.7071 | 0.7496 | 0.8082 | 0.5924 | 0.8263 |
| AttLSTM | ESM-2 | 0.7798 | 0.7859 | 0.7737 | 0.7764 | 0.7811 | 0.5596 | 0.8450 |

T5 is the strongest standalone embedding in this benchmark. Tree-based models (SVM, XGBoost, LightGBM) are consistently stronger than AttLSTM in single-embedding settings.

### 3.2 Combined-Embedding Performance

### Table III. Performance on combined embeddings (ProtBERT+T5+ESM-2)

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | 0.8960 | 0.9010 | 0.8909 | 0.8920 | 0.8965 | 0.7920 | 0.9539 |
| XGBoost | 0.8949 | 0.8949 | 0.8949 | 0.8949 | 0.8949 | 0.7899 | 0.9578 |
| LightGBM | 0.9020 | 0.9091 | 0.8949 | 0.8964 | 0.9027 | 0.8041 | 0.9593 |
| AttLSTM | 0.8909 | 0.8848 | 0.8970 | 0.8957 | 0.8902 | 0.7819 | 0.9459 |
| Ensemble (SVM+XGB+LGB) | 0.8990 | 0.8990 | 0.8990 | 0.8990 | 0.8990 | 0.7980 | 0.9579 |
| 4-Model Blend (SVM+XGB+LGB+AttLSTM) | 0.9051 | 0.9071 | 0.9030 | 0.9034 | 0.9052 | 0.8101 | 0.9592 |

Key finding from Table III: feature-level fusion markedly improves model balance and overall quality. The best overall model is the 4-Model Blend by accuracy, F1, and MCC, while LightGBM is the strongest single classifier on combined embeddings.

### 3.3 Feature Selection Analysis

To investigate whether a compact subset of fused features can preserve predictive performance, we evaluated Recursive Feature Elimination (RFE) and SHAP-based feature selection using 200, 250, 300, and 400 selected features. In all cases, the downstream classifier was the 4-model blend operating on the reduced feature space.

### Table IV. Performance comparison using RFE and SHAP feature selection for the 4-model blend

| Method | No. of Features | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RFE | 200 | 0.8929 | 0.8939 | 0.8919 | 0.8923 | 0.8931 | 0.7859 | 0.9544 |
| RFE | 250 | 0.8909 | 0.8919 | 0.8899 | 0.8903 | 0.8911 | 0.7818 | 0.9563 |
| RFE | 300 | 0.8990 | 0.8990 | 0.8990 | 0.8986 | 0.8988 | 0.7980 | 0.9580 |
| RFE | 400 | 0.8990 | 0.9010 | 0.8970 | 0.8962 | 0.8986 | 0.7980 | 0.9580 |
| SHAP | 200 | 0.8949 | 0.8960 | 0.8939 | 0.8952 | 0.8956 | 0.7900 | 0.9567 |
| SHAP | 250 | 0.8889 | 0.8919 | 0.8859 | 0.8873 | 0.8896 | 0.7778 | 0.9580 |
| SHAP | 300 | 0.8909 | 0.8919 | 0.8899 | 0.8907 | 0.8913 | 0.7818 | 0.9579 |
| SHAP | 400 | 0.8949 | 0.8980 | 0.8919 | 0.8937 | 0.8958 | 0.7900 | 0.9559 |

Table IV shows that feature selection preserves most of the predictive power of the full fused representation. Among these settings, RFE with 300 and 400 selected features achieved the best accuracy of 0.8990, while SHAP with 200 selected features achieved a competitive accuracy of 0.8949. Although none of the reduced-feature settings surpassed the full-feature 4-model blend, the performance drop remained small despite a substantial reduction in dimensionality. This indicates that the fused embedding space contains a compact subset of highly informative features and supports the usefulness of feature selection as an efficiency-oriented extension rather than a replacement for the full model.

## 4. Discussion

Two practical conclusions emerge:

1. Feature-level fusion contributes the largest performance gain.
2. Model-level fusion contributes an additional, smaller improvement.

Compared with earlier embedding setups based on GloVe/FastText/Word2Vec, this framework provides stronger balanced metrics, especially in AUC, MCC, F1, and sensitivity. This is important for bioinformatics tasks where missing positives can be costly.

## 5. Conclusion

This study demonstrates that merging ProtBERT, T5, and ESM-2 embeddings is an effective strategy for peptide classification. Single-embedding experiments show T5 as the strongest standalone representation, while combined embeddings improve all classifiers and produce better balance between sensitivity and specificity. The best final performance is achieved by a 4-model blend, reaching 90.51% accuracy, 0.9052 F1, 0.8101 MCC, and 0.9592 AUC. Overall, the evidence supports multi-embedding fusion as the primary improvement mechanism, with model fusion as a complementary enhancement.

## 6. Limitations and Future Work

This work still has limitations:

1. Evaluation is based on one dataset split; external validation is needed.
2. Model-fusion gain is moderate, so complexity-performance tradeoff should be reported.
3. Biological interpretability of fused features is not yet analyzed.

Future work will include external benchmarks, statistical significance testing, ablation on embedding subsets, and interpretable analysis of learned representations.
