# Text Classification Project: Comparative Analysis of Deep Learning and Classical Machine Learning Models

## Overview

This project explores multiple approaches for text classification on a domain-specific dataset, focusing on both **Deep Learning models** and **Classical Machine Learning models**. The goal is to evaluate and compare the performance of these models on both full-length texts and summarized versions of the texts.

---

## Dataset

The original dataset used for training and testing can be downloaded from the following source:  
[CORE Corpus - TurkuNLP GitHub](https://github.com/TurkuNLP/CORE-corpus)

- Training and test data CSV files have been preprocessed to remove unwanted classes (`OTHER`, `IP`).  
- Labels are encoded numerically for model compatibility.  
- Both full-text and summarized-text datasets are used for experimentation.

---

## Project Structure

The project contains the following main scripts and resources:

```

/Project\_D/
│
├── deep\_learning/
│   ├── bert\_base\_model.ipynb
│   ├── roberta\_base\_model.ipynb
│   ├── bert\_summarize\_model.ipynb
│   ├── roberta\_summarize\_model.ipynb
│   └── loss\_f1\_curves/             # Contains loss\_f1\_curve.png files for each model
│
├── machine\_learning\_summarize/
│   ├── machine\_learning\_summarize.ipynb
│   ├── confusion\_matrix.png       # Confusion matrix for classical ML voting model on summarized data
│   ├── classification\_report.png  # Classification report for classical ML voting model on summarized data
│
├── machine\_learning/
│   ├── confusion\_matrix.png       # Confusion matrix for classical ML voting model on full text data
│   ├── classification\_report.png  # Classification report for classical ML voting model on full text data
│
└── README.md

```

---

## Models and Approaches

### Deep Learning Models

- **BERT Base** (Full text)  
- **RoBERTa Base** (Full text) — *Best performing model*  
- **BERT Summarize** (Summarized text)  
- **RoBERTa Summarize** (Summarized text)  

These models are fine-tuned transformer architectures utilizing the HuggingFace library. Training and evaluation include plotting of loss and F1 score curves saved in the `loss_f1_curves` folder.

---

### Classical Machine Learning Models

- Logistic Regression (with cross-validation)  
- Random Forest Classifier  
- LightGBM Classifier  
- Final Ensemble Voting Classifier (soft voting combining the above models)

These models use TF-IDF vectorization (character-level n-grams 3-5) and Chi-squared feature selection (k=10,000 features) applied through sklearn Pipelines.  

---

## Performance Summary

| Model                       | Data Type         | Accuracy  |
|-----------------------------|-------------------|-----------|
| RoBERTa Base                | Full text         | **0.7444** |
| BERT Base                  | Full text         | 0.7318    |
| RoBERTa Summarize          | Summarized text   | 0.7164    |
| BERT Summarize             | Summarized text   | 0.7132    |
| Classical ML Voting        | Full text         | 0.7203    |
| Classical ML Voting        | Summarized text   | 0.6580    |

---

## Visualizations and Reports

### Loss and F1 Score Curves for Deep Learning Models

Loss and F1 score curves during training provide insights into model convergence and overfitting behavior.  
**Location:**  
`deep_learning/loss_f1_curves/`  
Example:  
- `bert_base_loss_f1_curve.png`  
- `roberta_base_loss_f1_curve.png`

### Confusion Matrix

Below is the confusion matrix of the **best performing model: RoBERTa Base** on full text data, illustrating class-wise prediction accuracy and common confusions.

![Confusion Matrix - RoBERTa Base](deep_learning/roberta_base_confusion_matrix.png)

> *Note: This confusion matrix demonstrates the highest overall accuracy among all models.*

Confusion matrices for other models are available in their respective folders.

### Classification Reports

Detailed classification reports including precision, recall, F1-score for all classes are saved as PNG images for all models.  
Examples:  
- `machine_learning/classification_report.png`  
- `machine_learning_summarize/classification_report.png`

---

## How to Run

1. **Download the dataset** from the CORE corpus repository:  
   https://github.com/TurkuNLP/CORE-corpus

2. **Preprocess the data** as shown in the notebooks (removing classes, encoding labels).

3. **Train and evaluate models** using the provided notebooks in the `/deep_learning/` and `/machine_learning_summarize/` folders.

4. **Use saved models and reports** to reproduce results and visualize metrics.

---

## Conclusions

- Transformer-based models, particularly **RoBERTa Base**, consistently outperform classical machine learning models.  
- Summarization of text reduces model accuracy, but may be useful for resource-limited scenarios.  
- Classical ML models with TF-IDF and feature selection provide a decent baseline and faster training times.  
- Ensemble methods improve robustness for classical models but still lag behind deep learning in accuracy.

---

## References

- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, 2018  
- Liu et al., RoBERTa: A Robustly Optimized BERT Pretraining Approach, 2019  
- CORE Corpus by TurkuNLP: https://github.com/TurkuNLP/CORE-corpus  
- Scikit-learn, LightGBM documentation  

---

```
