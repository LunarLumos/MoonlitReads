# Computational Prediction of Neuropeptide Using Transformer-Based Architecture and Blend Model

## Abstract
Accurate identification of neuropeptides is important for understanding peptide-mediated signaling and supporting peptide-based therapeutic discovery. Neuropeptides are involved in a wide range of biological activities, including neurotransmission, endocrine regulation, homeostasis, and intercellular communication. However, experimental identification of neuropeptides is time-consuming, costly, and not always suitable for large-scale screening. Therefore, developing an effective computational model for sequence-based neuropeptide prediction is highly desirable. In this paper, a novel predictor for computational prediction of neuropeptides is developed using transformer-based protein representations and a blend-model classification strategy. First, three pretrained transformer-derived embedding methods, namely ProtBERT, ProtT5, and ESM-2, are employed to capture complementary contextual and semantic information from peptide sequences. Second, the extracted feature spaces are integrated into a unified 3072-dimensional hybrid representation and standardized for downstream learning. Third, multiple classifiers, including SVM, XGBoost, LightGBM, and AttLSTM, are trained and evaluated, and their predictive outputs are further combined through a leakage-safe blend model. Experimental results show that the hybrid representation consistently improves performance over individual embedding views. Among the standalone classifiers, LightGBM achieves the strongest performance, whereas the final blend model yields the best overall results with an accuracy of 0.9051, F1-score of 0.9052, MCC of 0.8101, and AUC of 0.9592 on the independent test set. These findings indicate that transformer-based multi-view feature integration combined with probability-level blending is an effective strategy for neuropeptide prediction. The proposed framework provides a reproducible and practically useful computational resource for neuropeptide classification.

## 1 Introduction
Neuropeptides are short bioactive signaling molecules involved in neurotransmission, endocrine regulation, homeostasis, and peptide-based therapeutic discovery. Because wet-laboratory identification is expensive and time-consuming, computational prediction of neuropeptides has become an important front-end screening strategy. The predictive quality of such methods is largely determined by how sequence information is encoded and how effectively complementary discriminative cues are integrated.

Recent transformer-based protein language models have substantially improved biological sequence modeling by transferring contextual and semantic knowledge learned from large protein corpora. Nevertheless, distinct PLMs encode different aspects of biochemical context because they rely on different pretraining objectives, tokenization schemes, and model architectures. This creates a natural motivation for multi-view learning: instead of selecting a single embedding family, multiple transformer-derived feature spaces can be evaluated individually and then fused into a richer representation.

The methodological framework underlying this study follows that exact principle. It includes raw-sequence preprocessing and embedding extraction, tuned-versus-default benchmarking for each embedding view, deterministic multi-view alignment, fused-feature training, and two ensemble stages. Consequently, the paper is not based on a conceptual pipeline alone; it is grounded in an end-to-end reproducible workflow. The goal is to determine whether reproducible feature fusion and classifier fusion can improve neuropeptide prediction without relying on opaque or overly specialized optimization stages.

## Keywords
Neuropeptide prediction; protein language model; ProtBERT; ProtT5; ESM-2; feature fusion; ensemble learning

## 2 Methodology

### 2.1 Dataset construction
The experiments are conducted on a balanced benchmark dataset consisting of training and independent test sets. The training split contains 1,940 positive and 1,940 negative samples, giving a total of 3,880 sequences. The independent test split contains 495 positive and 495 negative samples, for a total of 990 sequences. This balanced design supports fair evaluation of sensitivity and specificity. To maintain reproducibility, data shuffling and preprocessing are performed with fixed random seeds.

At the preprocessing level, raw sequences are read from benchmark text files and processed by a sanitization routine that removes nonalphabetic characters and normalizes residues to uppercase. Invalid or empty sequences are skipped explicitly. The extracted features are then organized into positive, negative, and combined tabular datasets for each embedding view and each split, preserving a feature-first layout followed by a terminal target column.

### 2.2 Sequence encoding
To capture complementary information from peptide sequences, three pretrained protein language model embeddings are used as separate feature views. Each embedding generates a numerical representation of the same peptide samples, allowing comparison across models and fusion into a shared hybrid representation. In the implementation, the fused feature matrix is formed by horizontal concatenation of ProtT5, ProtBERT, and ESM-2 descriptors, resulting in a 3072-dimensional vector per sample.

The feature extraction process uses Hugging Face transformer models for ProtBERT and ProtT5 and the FAIR ESM-2 pretrained model `esm2_t33_650M_UR50D`. For ProtBERT and ProtT5, amino-acid sequences are tokenized by inserting spaces between residues to match model-specific tokenization requirements. ESM-2 receives the original sequence form through its alphabet batch converter.

#### 2.2.1 ProtBERT
ProtBERT is used as the first sequence representation method. It provides contextual embeddings derived from transformer-based pretraining on large protein databases. In the overall workflow, ProtBERT contributes a 768-dimensional feature block to both the single-view experiments and the final fused representation. Feature extraction is performed by forwarding tokenized sequences through `Rostlab/prot_bert` and selecting the final hidden-state vector of the CLS token. In this study, ProtBERT serves as a baseline semantic representation for evaluating single-view predictive performance.

#### 2.2.2 T5
ProtT5 is used as the second representation method. It captures rich contextual information and, in the present experiments, provides the strongest single-view performance among the tested embeddings. ProtT5 contributes a 1024-dimensional feature block. Feature extraction is performed with `Rostlab/prot_t5_xl_half_uniref50-enc`, and the final representation is obtained by mean pooling the encoder hidden states across the sequence dimension. This suggests that ProtT5 preserves discriminative information relevant to neuropeptide classification more effectively than the other individual views.

#### 2.2.3 ESM-2
ESM-2 is used as the third embedding source. It offers another large-scale protein representation that captures complementary contextual and structural information. In the fused pipeline, ESM-2 contributes a 1280-dimensional feature block. The implementation extracts layer-33 token representations from `esm2_t33_650M_UR50D` and applies mean pooling over nonpadding residues while excluding special tokens. Although its performance is generally slightly below ProtT5, it remains competitive and contributes useful information when combined with other embeddings.

### 2.3 Proposed model architecture
The proposed framework follows a two-stage design. In the first stage, ProtBERT, ProtT5, and ESM-2 embeddings are concatenated to form a hybrid feature space, and the resulting 3072-dimensional vector is standardized using `StandardScaler`. Before feature fusion, a deterministic alignment procedure applies a shared permutation within each split so that rows remain perfectly matched across ProtBERT, ProtT5, and ESM-2 views. After concatenation, the fused training set is shuffled once more with a fixed seed for downstream model training.

In the second stage, this fused feature representation is used to train SVM, XGBoost, LightGBM, and AttLSTM models. The classical models are optimized by `RandomizedSearchCV` with `StratifiedKFold` cross-validation using `n_splits=3`, `shuffle=True`, and `random_state=42`, while the search objective is ROC-AUC. The SVM search spans linear and RBF kernels; the tree-based models search over estimator count, depth, learning rate, subsampling, and regularization controls.

The AttLSTM architecture is implemented as a two-layer bidirectional LSTM with input size equal to the fused feature dimension, hidden size 256, dropout 0.3, and batch-first tensor layout. The input vector is unsqueezed into a sequence length of 1, propagated through the bidirectional recurrent block, and passed to an attention projection layer of size 512 to 1. The attention-weighted representation is then processed by a fully connected head with topology 512 to 256 to 2 using ReLU and dropout regularization. Training is performed with `CrossEntropyLoss` and `AdamW` at learning rate 0.001 for 50 epochs using deterministic mini-batch shuffling.

Two ensemble strategies are used. The first is a stacking ensemble over SVM, XGBoost, and LightGBM using logistic regression as a meta-learner with 5-fold stratified cross-validation and probabilistic stacking. The second is a four-model probability blend that combines SVM, XGBoost, LightGBM, and AttLSTM outputs through an equal-weight average. Because this blend operates directly on held-out probabilities without fitting a meta-classifier on the same evaluation set, it is described as a leakage-safe probability fusion step.

### 2.4 Recommended figure placement
To make the conference version visually complete, the following figures should be added in the indicated sections:

1. Figure 1 (end of Introduction): Overall workflow diagram.
	Suggested content: sequence input, three transformer embedding branches (ProtBERT/ProtT5/ESM-2), feature fusion, model training, blend output.
	Suggested caption: Overall framework for computational prediction of neuropeptides using transformer-based representations and blend-model classification.

2. Figure 2 (after Section 2.2): Embedding extraction and fusion schematic.
	Suggested content: dimension flow 768 + 1024 + 1280 -> 3072, then standardization and train/test split.
	Suggested caption: Multi-view embedding extraction and hybrid feature construction.

3. Figure 3 (after Section 2.3): AttLSTM architecture diagram.
	Suggested content: input vector, unsqueeze, BiLSTM (2 layers), attention block, FC head (512 -> 256 -> 2).
	Suggested caption: Attention-based bidirectional LSTM model used in the hybrid training stage.

4. Figure 4 (in Results, before Table II): Single-view ROC comparison.
	Suggested content: ROC curves for ProtBERT, ProtT5, and ESM-2 (best classifier per view or one fixed classifier across views).
	Suggested caption: ROC analysis for single-view transformer embeddings.

5. Figure 5 (in Results, before Table III): Hybrid-space model comparison chart.
	Suggested content: grouped bars for Accuracy, F1, MCC, AUC across SVM/XGBoost/LightGBM/AttLSTM/Ensemble/4-model blend.
	Suggested caption: Performance comparison of models on fused transformer feature space.

6. Figure 6 (after Section 3.3): Confusion matrix of best model.
	Suggested content: confusion matrix for 4-model blend on independent test set.
	Suggested caption: Confusion matrix of the final blend model on independent test data.

## 3 Results

### 3.1 Evaluating matrices
Model performance is evaluated using Accuracy, Sensitivity, Specificity, Precision, F1-score, Matthews Correlation Coefficient (MCC), and Area Under the ROC Curve (AUC). Accuracy measures overall correctness, while sensitivity and specificity describe the model’s ability to identify positive and negative samples, respectively. Precision and F1-score evaluate positive-class reliability, MCC provides a balanced correlation-based measure even under class imbalance, and AUC reflects threshold-independent discriminative performance. Together, these metrics provide a comprehensive evaluation of the predictive models.

In the evaluation workflow, all metrics are computed from the confusion matrix generated on the independent test split. Probabilistic outputs are obtained from posterior class probabilities for SVM, XGBoost, and LightGBM, and from softmax-normalized outputs for AttLSTM. This unified metric computation allows direct comparison between classical learners, deep learning, and ensemble fusion models. A visual ROC-based comparison should be provided in Figure 4.

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

### 3.2 Performance comparison on hybrid feature space

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | 0.8960 | 0.9010 | 0.8909 | 0.8920 | 0.8965 | 0.7920 | 0.9539 |
| XGBoost | 0.8949 | 0.8949 | 0.8949 | 0.8949 | 0.8949 | 0.7899 | 0.9578 |
| LightGBM | 0.9020 | 0.9091 | 0.8949 | 0.8964 | 0.9027 | 0.8041 | 0.9593 |
| AttLSTM | 0.8909 | 0.8848 | 0.8970 | 0.8957 | 0.8902 | 0.7819 | 0.9459 |
| Ensemble (SVM+XGB+LGB) | 0.8990 | 0.8990 | 0.8990 | 0.8990 | 0.8990 | 0.7980 | 0.9579 |
| 4-Model Blend (SVM+XGB+LGB+AttLSTM) | 0.9051 | 0.9071 | 0.9030 | 0.9034 | 0.9052 | 0.8101 | 0.9592 |

Table III demonstrates that hybrid feature fusion improves predictive consistency across all models. After combining the 1024-dimensional ProtT5, 768-dimensional ProtBERT, and 1280-dimensional ESM-2 embeddings into a standardized 3072-dimensional representation, every classifier achieves close to or above 0.89 accuracy, which is higher and more stable than most single-view results. Among the standalone models, LightGBM performs best, reaching 0.9020 accuracy, 0.9027 F1-score, 0.8041 MCC, and 0.9593 AUC. The 3-model ensemble further improves overall stability, while the final 4-model blend achieves the best overall performance with 0.9051 accuracy and 0.8101 MCC. These results indicate that the major gain arises from feature-level fusion of complementary embeddings, whereas model-level blending provides an additional but smaller improvement. Therefore, Table III confirms that hybrid representation learning is the main reason for the observed performance increase.

The corresponding visual summary should be included as Figure 5, and the best-model confusion matrix should be presented as Figure 6 for error pattern interpretation.

### 3.3 Proposed model Hyperparameter tuning
The methodological study contains two hyperparameter analysis layers. First, standalone benchmarking compares untuned default models against tuned models on each individual embedding view. In this setting, SVM is evaluated with linear, RBF, polynomial, and sigmoid kernels under 5-fold grid search; XGBoost and LightGBM are evaluated with explicit default-versus-grid-search comparisons; and the neural baseline study includes several architectures such as a fixed attention LSTM, an improved self-attention LSTM, and an MLP-style baseline. These standalone experiments provide methodological context for the per-view comparisons reported earlier.

Second, fused-feature training implements the final optimized models used in the hybrid benchmark. For SVM, a randomized search was conducted over linear and radial basis function kernels. The final selected configuration was a customized RBF-SVM with `C=10` and `gamma=0.0001`. During search, probability estimation was disabled for efficiency, whereas the final refit model enabled posterior probability estimation to support ROC-AUC calculation and ensemble blending.

For XGBoost, the default classifier was replaced with a customized configuration obtained by randomized search over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, and `reg_lambda`. The final fused-feature model used `n_estimators=700`, `max_depth=7`, `learning_rate=0.03`, `subsample=0.9`, `colsample_bytree=0.7`, `min_child_weight=5`, `gamma=0.2`, and `reg_lambda=1`, with `eval_metric='logloss'` and `random_state=42`.

For LightGBM, the final model was also obtained by randomized search rather than by default settings. The selected configuration was `n_estimators=700`, `max_depth=5`, `num_leaves=31`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=1.0`, `min_child_samples=10`, and `reg_lambda=1`, with `verbose=-1` and `random_state=42`. These customized gradient-boosting settings are important because they regulate tree complexity and sampling behavior in the high-dimensional fused space.

The AttLSTM model used a customized neural architecture rather than any framework default. Specifically, the network consists of a two-layer bidirectional LSTM with `hidden_dim=256`, `dropout=0.3`, and `batch_first=True`, followed by a learned attention layer, a fully connected layer of size 512 to 256, ReLU activation, dropout 0.3, and a final output layer of size 256 to 2. Training used `CrossEntropyLoss`, `AdamW` with learning rate `0.001`, `batch_size=64`, and `epochs=50`. The DataLoader was created with deterministic shuffling using a seeded generator, and the global random seed was fixed at 42 for NumPy and PyTorch.

At the ensemble level, two fusion strategies were used. The 3-model ensemble employed a stacking scheme with base learners SVM, XGBoost, and LightGBM, and a logistic regression meta-classifier with `max_iter=2000`. The 4-model blend did not use a learned meta-classifier; instead, it used a simple leakage-safe equal-weight probability average with weights `[0.25, 0.25, 0.25, 0.25]` across SVM, XGBoost, LightGBM, and AttLSTM. This design was intentionally chosen to keep the fusion stage transparent and reproducible.

### 3.4 Comparison with existing methods
When compared with existing neuropeptide prediction studies, the present framework should be interpreted as a realistic and reproducible baseline rather than a claim of state-of-the-art superiority. Some published studies report higher headline accuracy using more complex feature engineering, two-stage feature selection, or specialized deep architectures such as capsule networks. In contrast, the present study emphasizes methodological transparency, deterministic preprocessing, explicit feature extraction logic, and balanced performance across multiple metrics. The results show that the proposed multi-view framework is competitive and scientifically meaningful, even though it does not surpass the best reported results in the reference paper. Its main strength lies in demonstrating that a simpler fusion-based design can still achieve strong and stable performance under a fully specified and reproducible methodology.

## 4 Conclusion
In summary, this study investigates a multi-view neuropeptide prediction framework based on ProtBERT, ProtT5, and ESM-2 embeddings and ties every methodological claim to a concrete analytical stage. The pipeline covers sequence sanitization, model-specific tokenization, PLM feature extraction, deterministic cross-view alignment, standardized feature fusion, classifier-level hyperparameter optimization, neural attention-based learning, and both stacking and probability-based ensemble fusion. The results show that ProtT5 is the best single-view representation, while hybrid feature fusion consistently improves prediction quality across all classifiers. The best overall result is obtained by the 4-model blend, which achieves 0.9051 accuracy, 0.9052 F1-score, 0.8101 MCC, and 0.9592 AUC. These findings indicate that combining complementary protein language model embeddings is an effective strategy for improving neuropeptide prediction. Although the present method does not outperform the strongest published capsule-based model, it provides a realistic, reproducible, technically detailed, and conference-appropriate baseline for future research.

## References
References should include benchmark neuropeptide prediction studies, protein language model sources for ProtBERT, ProtT5, and ESM-2, transformer-based bioinformatics surveys, and the published neuropeptide paper used for methodological positioning. In the final submitted version, all implementation-linked claims about tokenization, pooling, model architecture, and ensemble design should be supported by the relevant methodological citations.
