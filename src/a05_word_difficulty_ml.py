import pickle
import pyphen
import pandas as pd
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from a01_word_difficulty import WordDifficulty
from classified_words import difficult_words, easy_words


class WordDifficultyML:
    """A machine learning class for word difficulty prediction.

    Attributes:
    - word_difficulty: An instance of the WordDifficulty class for word evaluation.
    - lemmatizer: An instance of the WordNetLemmatizer for lemmatizing words.
    - model: The machine learning model for word difficulty prediction.
    - scaler: The scaler used for standardizing features in the model.

    Methods:
    - get_syllable_count(word): Returns the syllable count of a given word.
    - set_model(): Sets up and trains the machine learning model.
    - is_valid_word(word): Checks if a word is a valid candidate for evaluation.
    - to_base_form(word): Converts a word to its base or root form.
    - eval_word(word): Evaluates the difficulty of a word using the trained model.
    - save_model(filename): Saves the trained model and scaler to a file.
    - load_model(filename): Loads a previously saved model and scaler from a file.
    """

    def __init__(self):
        self.word_difficulty = WordDifficulty()
        self.lemmatizer = WordNetLemmatizer()
        self.model = None
        self.scaler = None

    def get_syllable_count(self, word: str) -> int:
        """Returns the syllable count of a given word.

        Parameters:
        - word (str): The word to count syllables for.

        Returns:
        - int: The syllable count.
        """
        dic = pyphen.Pyphen(lang='en_US')
        return int(len(dic.inserted(word).split('-')))

    def set_model(self):
        """Sets up and trains the machine learning model for word difficulty prediction.
        """
        # 1. Create an empty DataFrame
        df = pd.DataFrame(index=difficult_words,
                          columns=['words', 'wordnet_words', 'movie_reviews', 'reuters', 'brown', 'gutenberg',
                                   'webtext', 'nps_chat', 'inaugural', "syllable_count", "classification"])

        # 2. Populate the DataFrame using your function
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

        # 3. set the classification columns as category
        df['classification'] = df['classification'].astype('category')

        # 4. Separate features and target variable
        X = df[['words', 'wordnet_words', 'movie_reviews', 'reuters', 'brown', 'gutenberg', 'webtext', 'nps_chat',
                'inaugural', 'syllable_count']]
        y = df['classification']

        # 5. Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Standardize the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 7. Create a model (you can choose a different model if needed)
        self.model = LogisticRegression(random_state=42)

        # 8. Train the model
        self.model.fit(X_train_scaled, y_train)

        # 9. Make predictions on the test set
        y_pred = self.model.predict(X_test_scaled)

        # 10. Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        # Print classification report for more detailed evaluation
        print(classification_report(y_test, y_pred))

    def is_valid_word(self, word: str) -> bool:
        """Checks if a word is a valid candidate for evaluation.

        Parameters:
        - word (str): The word to check.

        Returns:
        - bool: True if the word is a valid candidate, False otherwise.
        """
        word = word.strip().lower()
        if not word.isalpha():
            return False
        if word in self.word_difficulty.stopwords:
            return True
        if word in self.word_difficulty.wordnet_words:
            return True
        if word in self.word_difficulty.names:
            return False
        return False

    def to_base_form(self, word: str) -> str:
        """Converts a word to its base or root form using lemmatization.

        Parameters:
        - word (str): The word to lemmatize.

        Returns:
        - str: The base or root form of the word.
        """
        return self.lemmatizer .lemmatize(word)  # reduce words to their base or root form

    def eval_word(self, word: str) -> float:
        """Evaluates the difficulty of a word using the trained model.

        Parameters:
        - word (str): The word to evaluate.

        Returns:
        - float: The difficulty-probability score for the word.
        """
        new_word_features = self.word_difficulty.eval_word(word)
        new_word_features["syllable_count"] = self.get_syllable_count(word)
        print(new_word_features)

        # Convert the dictionary to a DataFrame with a single row
        new_word_df = pd.DataFrame([new_word_features])

        # Standardize the features using the same scaler used during training
        new_word_features_scaled = self.scaler.transform(new_word_df)

        # Predict the probability scores using the trained model
        probability_scores = self.model.predict_proba(new_word_features_scaled)[0]
        difficult_probability = round(probability_scores[1], 2)
        predicted_class = "difficult" if difficult_probability > 0.55 else "easy"

        if difficult_probability >= 0.9 and not self.is_valid_word(word):
            print(f"It seems that '{word}` is not a word (got {difficult_probability})")

            word_base = self.to_base_form(word)
            if word_base != word:
                return self.eval_word(word_base)
            return -1.0

        print(f"The difficulty-probability score for `{word}` is: {difficult_probability} ({predicted_class})")
        return difficult_probability

    def save_model(self, filename: str):
        """Saves the trained model and scaler to a file.

        Parameters:
        - filename (str): The name of the file to save the model to.
        """
        if self.model is not None and self.scaler is not None:
            with open(filename, 'wb') as file:
                model_data = {'model': self.model, 'scaler': self.scaler}
                pickle.dump(model_data, file)
            print(f"Model saved to {filename}")
        else:
            print("Cannot save model. Please train the model first.")

    def load_model(self, filename: str):
        """Loads a previously saved model and scaler from a file.

        Parameters:
        - filename (str): The name of the file to load the model from.
        """
        try:
            with open(filename, 'rb') as file:
                model_data = pickle.load(file)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"Model file '{filename}' not found. Please check the file path.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
