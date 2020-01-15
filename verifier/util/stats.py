from verifier import config
from verifier.preprocessing.samples_database import SamplesDatabase


def print_sample_stats():
    db = SamplesDatabase.get()

    total = len(db.read(None, "title"))
    crossplatform = len(db.filter(('cross_platform', '==', True)))
    en = len(db.filter(('lang', '==', 'en'), ('description_raw', 'len>', 30)))

    downloads = len(db.filter(('downloads', '>=', config.Clustering.min_downloads_visualize)))

    print("# total = ", total)
    print("# cross platform = ", crossplatform, "   ", 100*crossplatform/total, "%")
    print("# en > 30 = ", en, "   ", 100*en/total, "%")
    print("# downloads >= 4e6 = ", downloads, "   ", 100*downloads/total, "%")


if __name__ == "__main__":
    print_sample_stats()