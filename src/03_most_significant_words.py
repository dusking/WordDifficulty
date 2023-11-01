import os
from pathlib import Path

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from opensubtitlescom import OpenSubtitles

from contractions import contractions_dict

SUBTITLES_COM_APP_NAME = os.environ.get("SUBTITLES_COM_APP_NAME")
SUBTITLES_COM_API_KEY = os.environ.get("SUBTITLES_COM_API_KEY")
SUBTITLES_COM_USERNAME = os.environ.get("SUBTITLES_COM_USERNAME")
SUBTITLES_COM_PASSWORD = os.environ.get("SUBTITLES_COM_PASSWORD")


class MostSignificantWords:
    def __init__(self):
        """Initialize the MostSignificantWords class.

        Initializes class variables for handling subtitles and natural language processing.
        """
        self.srt_folder = Path("srt")
        self.lemmatizer = WordNetLemmatizer()
        self.opensubtitles = OpenSubtitles(SUBTITLES_COM_API_KEY, f"{SUBTITLES_COM_APP_NAME} v1.0.0")
        self.opensubtitles.login(SUBTITLES_COM_USERNAME, SUBTITLES_COM_PASSWORD)
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def download_corpora(self):
        """Download essential NLTK corpora for natural language processing tasks.

        This function downloads the following NLTK corpora:
        - 'stopwords': Common stopwords in English.
        - 'punkt': for sentence tokenization.

        Usage:
        download_corpora()
        """
        nltk.download('stopwords')
        nltk.download('punkt')

    def get_tfidf_significant_words(self, features) -> list:
        """Get the most significant words using TF-IDF for a list of features.

        Args:
            features (list): List of feature dictionaries.

        Returns:
            list: List of significant words for each feature.
        """
        transcripts = []
        for feature in features:
            title = feature["title"]
            if "season_number" in feature:
                title += f"_s{feature['season_number']}_e{feature['episode_number']}"
            transcript = self.get_cleaned_transcript(**feature)
            transcripts.append({"title": title, "transcript": transcript})
        return self.evaluate_most_important_words(transcripts)

    def get_cleaned_transcript(self, title, year=None, season_number=None, episode_number=None) -> str:
        """Clean and preprocess the transcript for a given title and episode.

        Args:
           title (str): Title of the episode.
           year (int): Year of the episode.
           season_number (int): Season number.
           episode_number (int): Episode number.

        Returns:
           str: Cleaned and preprocessed transcript.
       """
        transcript = ""
        subtitles = self.get_subtitles(title, year, season_number, episode_number)
        for line in subtitles:
            words = []
            for word in line.words():
                for word_i in contractions_dict.get(word, word).split():
                    if word_i in self.stopwords:
                        continue
                    words.append(word_i)
            transcript += " ".join(words) + " "
        return transcript

    def get_subtitles(self, title, year=None, season_number=None, episode_number=None) -> list:
        """Get subtitles for a given title, season, and episode.

        Args:
           title (str): Title of the episode.
           year (int): Year of the episode.
           season_number (int): Season number.
           episode_number (int): Episode number.

        Returns:
           list: List of subtitle lines.
       """
        suffix = ".srt"
        episode_str = f"_s{season_number}_e{episode_number}" if season_number else ""
        filename = title.lower().replace(" ", "_") + episode_str + suffix
        filepath = self.srt_folder / filename
        if not filepath.exists():
            print(f"Retrieving {title} subtitles")
            response = self.opensubtitles.search(query=title, year=year, season_number=season_number,
                                                 episode_number=episode_number, languages="en")
            if not response.data:
                print(f"Missing subtitles for {title}")
                return []
            self.opensubtitles.download_and_save(response.data[0], filename=filepath)
        with open(str(filepath)) as f:
            srt = f.read()
        return self.opensubtitles.parse_srt(srt)

    def evaluate_most_important_words(self, movie_transcripts) -> list:
        """Evaluate the most important words using TF-IDF for a list of movie transcripts.

        Args:
           movie_transcripts (list): List of movie transcripts with titles.

        Returns:
           list: List of important words for each movie transcript.
        """
        def custom_tokenizer(text):
            words = nltk.word_tokenize(text)

            # Apply stemming
            stemmer = PorterStemmer()
            base_form_stemmed_words = []
            for word in words:
                word = stemmer.stem(word)  # handle steamid words like "he's"
                if "'" in word:
                    # skip words like I'm - since they will be break into I and 'm
                    continue
                base_form_stemmed_words.append(word)
            return base_form_stemmed_words

        # 1. Tokenize the movies transcripts
        tokens_vec = []
        for movie_transcript in movie_transcripts:
            tokens = nltk.word_tokenize(movie_transcript["transcript"])
            tokens_vec.append(" ".join(tokens))

        # 2. Create a TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)

        # 3. Fit and transform the transcript
        tfidf_matrix = tfidf_vectorizer.fit_transform(tokens_vec)

        # 4. Get feature names (words) and their corresponding TF-IDF scores
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()

        # 5. Evaluate the most important words per film
        important_words_per_film = []
        for i, score in enumerate(tfidf_scores):
            # Sort words by TF-IDF scores and select the top 30
            important_words = [word for word, score in
                               sorted(zip(feature_names, score), key=lambda x: x[1], reverse=True)[:30]]
            important_words_str = ", ".join(important_words)
            film_name = movie_transcripts[i]["title"]
            important_words_per_film.append({"film": film_name, "words": important_words_str})

        return important_words_per_film
