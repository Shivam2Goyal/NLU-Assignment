import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Loading the CSVs
def get_data(file_name):
    df = pd.read_csv(file_name)
    # Using 'text' and 'label' columns as per the project spec
    return df["text"], df["label"]

def main():
    print("--- Sports / Politics Classification ---")
    
    # 1. Loading the pre-split CSVs
    train_x, train_y = get_data("train_processed.csv")
    test_x, test_y = get_data("test_processed.csv")

    # 2. Setup TF-IDF 
    # Added max_features because sometimes the vocabulary gets way too big and slows down the Random Forest.
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2), # Using bigrams to catch phrases like "prime minister"
        min_df=3,           # Ignoring words that only show up once or twice
        max_features=5000   
    )

    # Transform the text into numbers - Vectorizing
    X_train = tfidf.fit_transform(train_x)
    X_test = tfidf.transform(test_x)

    # 3. Stating the models to test
    clf_list = [
        ("Naive Bayes", MultinomialNB()),
        ("Linear SVM", LinearSVC()),
        ("Random Forest", RandomForestClassifier(n_estimators=100, n_jobs=-1))
    ]

    # 4. Training and Evaluation
    for name, clf in clf_list:
        clf.fit(X_train, train_y)
        
        # Get predictions
        y_preds = clf.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(test_y, y_preds)
        f1 = f1_score(test_y, y_preds, average="weighted")
        
        print(f"Results for {name}:")
        print(f" -> Accuracy: {round(acc, 4)}")
        print(f" -> F1-Score: {round(f1, 4)}")

if __name__ == "__main__":
    main()
