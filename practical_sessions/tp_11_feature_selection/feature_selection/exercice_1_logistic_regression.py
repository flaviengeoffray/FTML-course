"""
    Learn a baseline estimator.

    We build a Pipeline that contains
   - a one-hot encoding of the data
   - a scaling of the data
   - a logistic regression

   The one-hot encoding part has some important parameters, about which
   you can find more info in the doc.
   -  ngram range: choice of the length of the ngrams that are used in the
      CountVectorizer. A possible choice is to use
   the value ngram_range = (1, 2), but you may experiment with other values.
   -  min_df: minimum number of documents or document frequency for a word to be 
   kept in the dicitonary.
"""

from utils_data_processing import preprocess_imdb
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from constants import NUM_JOBS, NGRAM_RANGE, MIN_DF


def save_vocabulary(clf: Pipeline, file_name: str) -> None:
    """
    Save the vocabulary to a .txt file
    Extract the feature space size.
    """
    vectorizer = clf["vectorizer"]
    path = "vocabularies"
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, file_name), "w") as f:
        for word in vectorizer.get_feature_names_out():
            f.write(f"{word}\n")


if __name__ == "__main__":
    """
    preprocess_imdb() returns scikit bunches
    For instance,
    - traindata.data contains the list of all source texts.
    - traindata.target contains the list of all binary targets (positive or
      negative review)
    """
    traindata, _, testdata = preprocess_imdb(num_jobs=NUM_JOBS)

    # define the pipeline
    clf = Pipeline([
        ("vectorizer", CountVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF)),
        ("scaler", MaxAbsScaler()),
        ("classifier", LogisticRegression(solver="liblinear"))
    ])

    clf.fit(traindata.data, traindata.target)

    file_name = f"vocabulary_{NGRAM_RANGE[0]}_{NGRAM_RANGE[1]}_min_df_{MIN_DF}.txt"
    save_vocabulary(clf, file_name)

    acc_train = clf.score(traindata.data, traindata.target)
    acc_test = clf.score(testdata.data, testdata.target)
    print(f"Accuracy on training set: {acc_train:.3f}")
    print(f"Accuracy on test set: {acc_test:.3f}")

