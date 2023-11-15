import pyphen
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from a01_word_difficulty import WordDifficulty
from classified_words import difficult_words, easy_words


class WordDifficultyML:
    def __init__(self):
        self.word_difficulty = WordDifficulty()
        self.clf = None
        self.scaler = None

    def get_syllable_count(self, word):
        dic = pyphen.Pyphen(lang='en_US')
        return int(len(dic.inserted(word).split('-')))

    def set_model(self):
        # Create an empty DataFrame
        df = pd.DataFrame(index=difficult_words,
                          columns=['words', 'wordnet_words', 'movie_reviews', 'reuters', 'brown', 'gutenberg',
                                   'webtext', 'nps_chat', 'inaugural', "syllable_count", "classification"])

        # Populate the DataFrame using your function
        for word in easy_words:
            for column, value in self.word_difficulty.eval_word(word).items():
                # Update the DataFrame using the at method
                df.at[word, column] = value
            df.at[word, "syllable_count"] = self.get_syllable_count(word)
            df.at[word, "classification"] = 1
        for word in difficult_words:
            for column, value in self.word_difficulty.eval_word(word).items():
                # Update the DataFrame using the at method
                df.at[word, column] = value
            df.at[word, "syllable_count"] = self.get_syllable_count(word)
            df.at[word, "classification"] = 2

        # set the classification columns as category
        df['classification'] = df['classification'].astype('category')

        # Separate features and target variable
        X = df[['words', 'wordnet_words', 'movie_reviews', 'reuters', 'brown', 'gutenberg', 'webtext', 'nps_chat',
                'inaugural', 'syllable_count']]
        y = df['classification']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create a model (you can choose a different model if needed)
        self.clf = LogisticRegression(random_state=42)
        # self.clf = GradientBoostingClassifier(random_state=42)

        # Train the model
        self.clf.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = self.clf.predict(X_test_scaled)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        # Print classification report for more detailed evaluation
        print(classification_report(y_test, y_pred))

        # Eval new word

    def is_valid_word(self, word):
        word = word.strip().lower()
        if word in self.word_difficulty.stopwords:
            return False
        if not word.isalpha():
            return False
        if len(word) < 3:
            return False
        return True

    def eval_word(self, word):
        new_word_features = self.word_difficulty.eval_word(word)
        new_word_features["syllable_count"] = self.get_syllable_count(word)
        print(new_word_features)

        # Convert the dictionary to a DataFrame with a single row
        new_word_df = pd.DataFrame([new_word_features])

        # Standardize the features using the same scaler used during training
        new_word_features_scaled = self.scaler.transform(new_word_df)

        # Predict the probability scores using the trained model
        probability_scores = self.clf.predict_proba(new_word_features_scaled)[0]

        # Get the predicted class based on the maximum probability score
        predicted_class = "easy" if int(probability_scores.argmax()) == 0 else "difficult"

        probability_scores = [round(score, 2) for score in probability_scores]

        print(f"The probability scores for {word} are: {probability_scores} --> {predicted_class}")
        return probability_scores
