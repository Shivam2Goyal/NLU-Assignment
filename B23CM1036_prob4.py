import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def load_dataset(csv_path: str):
    data = pd.read_csv(csv_path)
    texts = data["text"]
    labels = data["label"]
    return texts, labels


def build_tfidf(train_corpus, test_corpus):
    extractor = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    train_matrix = extractor.fit_transform(train_corpus)
    test_matrix = extractor.transform(test_corpus)

    return train_matrix, test_matrix


def evaluate_classifier(name, estimator, X_tr, y_tr, X_te, y_te):
    estimator.fit(X_tr, y_tr)
    predictions = estimator.predict(X_te)

    accuracy = accuracy_score(y_te, predictions)
    f1 = f1_score(y_te, predictions, average="weighted")

    print(name)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}\n")


def main():
    # Read preprocessed splits
    train_texts, train_labels = load_dataset("train_processed.csv")
    test_texts, test_labels = load_dataset("test_processed.csv")

    # TF-IDF representation
    X_train, X_test = build_tfidf(train_texts, test_texts)

    # Model collection
    models = [
        ("Naive Bayes", MultinomialNB()),
        ("Linear SVM", LinearSVC()),
        ("Random Forest", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ]

    print("Sports vs Politics Text Classification\n")

    for model_name, model in models:
        evaluate_classifier(model_name, model, X_train, train_labels, X_test, test_labels)


if __name__ == "__main__":
    main()
