import os
import re
from pathlib import Path
from collections import Counter

import pandas as pd
from opensubtitlescom import OpenSubtitles

from word_difficulty import WordDifficulty

SUBTITLES_COM_APP_NAME = os.environ.get("SUBTITLES_COM_APP_NAME")
SUBTITLES_COM_API_KEY = os.environ.get("SUBTITLES_COM_API_KEY")
SUBTITLES_COM_USERNAME = os.environ.get("SUBTITLES_COM_USERNAME")
SUBTITLES_COM_PASSWORD = os.environ.get("SUBTITLES_COM_PASSWORD")


class MovieVocabularyAnalysis:
    """Estimate the difficulty of words in movie or TV show subtitles using Python.

    This class provides methods to download subtitles, extract and analyze word difficulty,
    and summarize the word difficulty statistics.

    Attributes:
        srt_folder (Path): The folder to store downloaded subtitle files.
        word_difficulty (WordDifficulty): An instance of the WordDifficulty class for word assessment.
        os (OpenSubtitles): An instance of the OpenSubtitles class for downloading subtitles.
    """
    def __init__(self):
        """Initialize the MovieWordDifficulty class.

        Initializes the srt_folder, word_difficulty, and os attributes.
        """
        self.srt_folder = Path("srt")
        self.word_difficulty = WordDifficulty()
        self.opensubtitles = OpenSubtitles(SUBTITLES_COM_API_KEY, f"{SUBTITLES_COM_APP_NAME} v1.0.0")
        self.opensubtitles.login(SUBTITLES_COM_USERNAME, SUBTITLES_COM_PASSWORD)

    def assess_difficulty(self, title, year=None, season_number=None, episode_number=None, attempts=3) -> dict:
        """Assess the word difficulty of a movie or TV show transcript.

        Args:
            title (str): The title of the movie or TV show.
            year (int): The year of the movie release (if applicable).
            season_number (int): The season number (if applicable).
            episode_number (int): The episode number (if applicable).
            attempts (int): Maximum download retries for valid subtitles (default is 3).

        Returns:
            dict: A dictionary with word difficulty statistics, including counts and percentages.
        """
        index = 0
        while index < attempts:
            filename = self.download_subtitles(title, year, season_number, episode_number, index)
            if not filename:
                return {}
            df = self.analyze_words_from_srt_file(filename)
            stat = self.summarize_words_classification(df)
            if self.is_valid_subtitles_file(stat):
                return stat
            index += 1
        return {}

    def download_subtitles(self, title, year=None, season_number=None, episode_number=None, index=None) -> str:
        """Download subtitles for a movie or TV show.

       Args:
           title (str): The title of the movie or TV show.
           year (int): The year of the movie release (if applicable).
           season_number (int): The season number (if applicable).
           episode_number (int): The episode number (if applicable).
           index (int): The index of the subtitle to download - from available options (if applicable).

       Returns:
           str: The file path to the downloaded subtitle file.
       """
        index = index or 0
        suffix = f"_{index}.srt" if index else ".srt"
        episode_str = f"_s{season_number}_e{episode_number}" if season_number else ""
        filename = title.lower().replace(" ", "_") + episode_str + suffix
        filepath = self.srt_folder / filename
        if not filepath.exists():
            print(f"Retrieving {title} subtitles")
            response = self.opensubtitles.search(query=title, year=year, season_number=season_number,
                                                 episode_number=episode_number, languages="en")
            if not response.data:
                print(f"Missing subtitles for {title}")
                return ""
            self.opensubtitles.download_and_save(response.data[index], filename=filepath)
        return str(filepath)

    def analyze_words_from_srt_file(self, filename) -> pd.DataFrame:
        """Extract words from a subtitle file.

        Args:
            filename (str): The file path of the subtitle file.

        Returns:
            DataFrame: A DataFrame containing words in the subtitle file with their count and classification.
        """
        with open(filename) as f:
            srt = f.read()
        subtitles = self.opensubtitles.parse_srt(srt)
        words = Counter()
        for line in subtitles:
            for word in self.extract_words_from_line(line.content):
                words[word] += 1
        df = pd.DataFrame(list(words.items()), columns=['Word', 'Count'])

        # Add a column for word difficulty classification.
        df['Classification'] = df['Word'].apply(self.word_difficulty.evaluate_word_difficulty)

        return df

    def extract_words_from_line(self, line) -> list:
        """Extract words from a subtitle line.

        Args:
            line (str): The subtitle line.

        Returns:
            list: A list of words extracted from the subtitle line.
        """
        # Remove HTML tags (may be used for text formatting)
        line = re.sub(r'<.*?>', '', line)

        # Removes any non-alphabet characters
        clean_text = re.sub(r'[^a-zA-Z]', ' ', line)

        # Extract word
        return [w.strip().lower() for w in clean_text.split()]

    def summarize_words_classification(self, df) -> dict:
        """Summarize word classification statistics based on their counts.

        Args:
            df (pd.DataFrame): A DataFrame containing word counts and classifications.

        Returns:
            dict: A dictionary with word classification statistics.
                Keys are "easy," "moderate," and "difficult," and values are the respective word counts
                and percentages in the subtitles.
        """
        grouped_df = df.groupby('Classification')["Count"].sum()
        stat = grouped_df.to_dict()
        total = sum(stat.values())
        for key, value in stat.items():
            stat[key] = f"{value} ({round((value / total) * 100, 1)}%)"
        return stat

    def is_valid_subtitles_file(self, stat) -> bool:
        """Check the validity of the subtitles file based on the percentage of unclassified words.

        Args:
            stat (dict): A dictionary with word classification statistics.

        Returns:
            bool: True if the subtitles file is valid, False if it's potentially problematic.
        """
        def extract_percentage(stat_value):
            return float(re.search(r'\((.*?)\)', stat_value).group(1).rstrip('%'))

        valid_subtitles_threshold = 5
        unclassified_words_percentage = extract_percentage(stat["unclassified"])
        if unclassified_words_percentage < valid_subtitles_threshold:
            return True
        print(f"Potentially problematic subtitles file, unclassified words: {unclassified_words_percentage}%")
        return False
