# Topic-Modeling-Sentiment-Analysis-in-Financial-Text
1. Project Overview

This assignment implements a full NLP pipeline for financial text understanding, including:

Exploratory Data Analysis (EDA)

Topic Modeling using Latent Dirichlet Allocation (LDA)

Three Sentiment Analysis Systems:

FinBERT (domain-specific transformer)

Local LLM (Phi-2) — Zero-Shot

RAG-Enhanced LLM (FAISS + SBERT + Phi-2 Few-Shot)

Comparative evaluation across models

Conditional Fine-Tuning Rule Implementation

Reproducibility and structured pipeline execution

The pipeline adheres to all rules and constraints outlined in Assignment 3.

2. Folder Structure
/
│── i221158_a3.ipynb        
│── i221158_a3_report.pdf   
│── dataset/
│     └── Sentences_AllAgree.txt   
│── README.txt            

4. How to Run the Notebook
Step 1 — Open Notebook in Google Colab

Upload the file i221158_a3.ipynb to Google Colab.

Step 2 — Upload Dataset

Create a folder named:

/content/dataset/


Upload the file:

Sentences_AllAgree.txt


The final dataset path must be:

/content/dataset/Sentences_AllAgree.txt

Step 3 — Run the Notebook Sequentially

Simply run all cells from top to bottom.
The notebook automatically:

Installs all required libraries

Configures the environment

Sets seeds for reproducibility

Loads and preprocesses the dataset

Performs LDA topic modeling

Runs FinBERT, Phi-2 Zero-Shot, and RAG models

Generates confusion matrices

Performs comparative analysis

Applies the fine-tuning rule

No manual setup is required beyond uploading the dataset.

4. Dependencies

All dependencies are installed automatically by the first cell in the notebook.

The following libraries are used:

pandas

numpy

torch

nltk

gensim

faiss (CPU or GPU version)

scikit-learn

sentence-transformers

transformers

accelerate

bitsandbytes (if available)

matplotlib

seaborn

The notebook detects missing libraries and installs them on the fly.

5. Key Outputs Generated

The notebook produces:

✔ EDA Outputs

Sentiment distribution

Cleaned token samples

✔ LDA Topic Modeling

Multi-topic coherence comparison

Topic keywords

Topic assignments

✔ Sentiment Analysis

FinBERT Results (Full 453 samples)

Local LLM Zero-Shot Results (50 samples)

RAG-Enhanced Few-Shot Results (50 samples)

Each model outputs:

Accuracy

Precision

Recall

F1-score

Confusion matrix plot

✔ Conditional Fine-Tuning Decision

FinBERT achieved 96.91% accuracy, so fine-tuning was skipped, as per assignment rules.

6. Reproducibility

The notebook ensures reproducibility by:

Setting all seeds (random, numpy, torch)

Using deterministic splits

Using fixed max length and batch sizes

Reporting exact metrics

Using fixed prompt templates in LLM and RAG

7. Notes & Limitations

Local LLM and RAG evaluations are done on 50 samples due to model size and GPU time limits in Colab.

All core pipeline components are implemented exactly as required by the assignment.

RAG performance may fluctuate depending on Colab GPU availability.

8. Contact

For any issue with reproduction or pipelines, feel free to contact:
i221158@nu.edu.pk
