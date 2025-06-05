"""
    Univariate filtering
"""

from utils_data_processing import preprocess_imdb
from sklearn.metrics import make_scorer, accuracy_score
import os
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from constants import NUM_JOBS, NGRAM_RANGE, MIN_DF


def sparsity_scorer(clf: Pipeline) -> float:
    """
    Define a sparsity score for the pipeline

    The sparsisity score can for instance be computed
    from the fraction of ketpt words in the vocabulary.
    Optionally and if relevant, it may also include the sparsity of the
    linear estimator.

    The score is a float in [0, 1]
    a pipeline with a 0 score is not sparse at all
    a pipeline with a 1 score is fully sparse

    EDIT THIS FUNCTION
    """
    f = clf["vectorizer"].get_feature_names_out()
    n_features = len(f)
    n_features_selected = clf["selector"].k
    sparsity = n_features_selected / n_features
    return sparsity

def evaluate_k(k: int, C=1.0):
    """
    Evaluate the model with a given k for SelectKBest
    """
    clf = Pipeline([
        ("vectorizer", CountVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF)),
        ("scaler", MaxAbsScaler()),
        ("selector", SelectKBest(score_func=chi2, k=k)),
        ("classifier", LogisticRegression(C=C, solver="liblinear"))
    ])

    clf.fit(traindata.data, traindata.target)
    
    acc_train = clf.score(traindata.data, traindata.target)
    acc_test = clf.score(testdata.data, testdata.target)
    sparsity = sparsity_scorer(clf)
    return acc_train, acc_test, sparsity


if __name__ == "__main__":
    traindata, _, testdata = preprocess_imdb(num_jobs=NUM_JOBS)
    """
    Add lines here.
    """

    k = [1, 100, 1000, 10000, 20000, 40000, 60000, 80000]
    
    acc_train_list = []
    acc_test_list = []
    sparsity_list = []

    for k_value in k:
        acc_train, acc_test, sparsity = evaluate_k(k_value)
        print(f"k={k_value}, Train Accuracy: {acc_train:.3f}, Test Accuracy: {acc_test:.3f}, Sparsity: {sparsity:.3f}")
        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)
        sparsity_list.append(sparsity)
