import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE


from a03_most_significant_words import MostSignificantWords


class TSne:
    def __init__(self):
        self.tfidf = MostSignificantWords()

    def run(self):
        def name(**feature):
            season = feature.get("season_number", "")
            episode = f" S{season}" if season else ""
            return feature.get("title").title() + episode

        features = [
            dict(title="Black Mirror", season_number=1, episode_number=1, color="black"),
            dict(title="Black Mirror", season_number=1, episode_number=2, color="black"),
            dict(title="Black Mirror", season_number=1, episode_number=3, color="black"),
            dict(title="Black Mirror", season_number=2, episode_number=1, color="black"),
            dict(title="Black Mirror", season_number=2, episode_number=2, color="black"),
            dict(title="Black Mirror", season_number=2, episode_number=3, color="black"),

            dict(title="Two and a Half Men", season_number=1, episode_number=1, color="red"),
            dict(title="Two and a Half Men", season_number=1, episode_number=2, color="red"),
            dict(title="Two and a Half Men", season_number=1, episode_number=3, color="red"),
            dict(title="Two and a Half Men", season_number=2, episode_number=1, color="red"),
            dict(title="Two and a Half Men", season_number=2, episode_number=2, color="red"),
            dict(title="Two and a Half Men", season_number=2, episode_number=3, color="red"),

            dict(title="How I Met Your Mother", season_number=1, episode_number=1, color="pink"),
            dict(title="How I Met Your Mother", season_number=1, episode_number=2, color="pink"),
            dict(title="How I Met Your Mother", season_number=1, episode_number=3, color="pink"),
            dict(title="How I Met Your Mother", season_number=2, episode_number=1, color="pink"),
            dict(title="How I Met Your Mother", season_number=2, episode_number=2, color="pink"),
            dict(title="How I Met Your Mother", season_number=2, episode_number=3, color="pink"),

            # James Bond
            dict(title="Licence To Kill", year=1989, color="green"),
            dict(title="GoldenEye", year=1995, color="green"),
            dict(title="Tomorrow Never Dies", year=1997, color="green"),
            dict(title="The World Is Not Enough", year=1999, color="green"),
            dict(title="Die Another Day", year=2002, color="green"),
            dict(title="Casino Royale", year=2006, color="green"),

            # Start Wars
            dict(title="The Clone Wars", year=2008, color="blue"),
            dict(title="The Phantom Menace", year=1999, color="blue"),
            dict(title="Attack of the Clones", year=2002, color="blue"),
            dict(title="Revenge of the Sith", year=2005, color="blue"),
            dict(title="The Force Awakens", year=2015, color="blue"),

            dict(title="Friends", season_number=1, episode_number=1, color="purple"),
            dict(title="Friends", season_number=1, episode_number=2, color="purple"),
            dict(title="Friends", season_number=1, episode_number=3, color="purple"),
            dict(title="Friends", season_number=1, episode_number=4, color="purple"),
            dict(title="Friends", season_number=2, episode_number=1, color="purple"),
            dict(title="Friends", season_number=2, episode_number=2, color="purple"),
            dict(title="Friends", season_number=2, episode_number=3, color="purple"),
            dict(title="Friends", season_number=2, episode_number=4, color="purple"),

            dict(title="Agent Carter", season_number=1, episode_number=1, color="orange"),
            dict(title="Agent Carter", season_number=1, episode_number=2, color="orange"),
            dict(title="Agent Carter", season_number=1, episode_number=3, color="orange"),
            dict(title="Agent Carter", season_number=1, episode_number=4, color="orange"),
            dict(title="Agent Carter", season_number=2, episode_number=1, color="orange"),
            dict(title="Agent Carter", season_number=2, episode_number=2, color="orange"),
            dict(title="Agent Carter", season_number=2, episode_number=3, color="orange"),
            dict(title="Agent Carter", season_number=2, episode_number=4, color="orange"),
        ]

        names = [name(**item) for item in features]
        colors = [item["color"] for item in features]
        tfidf_matrix = self.tfidf.get_tfidf_matrix_for_features(features)
        return self.visualize_tfidf_with_tsne_clusters(tfidf_matrix, names, colors)

    def visualize_tfidf_with_tsne_clusters(self, tfidf_matrix, movie_names, colors, n_components=2, perplexity=10):
        """
        Visualize TF-IDF data with t-SNE and add colored clusters.

        Args:
        - tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF matrix for the documents.
        - movie_names (list): List of movie names.
        - colors (list): List of colors corresponding to each data point.
        - n_components (int): Number of t-SNE components (default is 2 for 2D).
        - perplexity (float): Perplexity parameter for t-SNE (controls balance between local and global relationships).

        Returns:
        - None (displays the t-SNE plot with clusters).
        """

        # Run t-SNE on the TF-IDF features
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        tsne_results = tsne.fit_transform(tfidf_matrix.toarray())

        # Create a scatter plot with data points
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors)

        # List of cluster labels for each data point.
        color_to_label = {color: i for i, color in enumerate(set(colors))}
        cluster_labels = [color_to_label[color] for color in colors]

        # Add colored circles for clusters
        cluster_ids = np.unique(cluster_labels)
        for cluster_id in cluster_ids:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_center = np.mean(tsne_results[cluster_indices], axis=0)
            cluster_color = colors[cluster_indices[0]]
            cluster_size = len(cluster_indices)
            circle = mpatches.Circle(cluster_center, radius=cluster_size, color=cluster_color, alpha=0.3)
            plt.gca().add_patch(circle)

        # Label the points with movie names
        for i, movie in enumerate(movie_names):
            plt.annotate(movie, (tsne_results[i, 0], tsne_results[i, 1]))

        plt.title("t-SNE Visualization of Movie Transcripts")
        plt.show()

