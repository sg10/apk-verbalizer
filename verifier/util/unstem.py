from collections import Counter


class UnStemmer:

    _instance = None
    enabled = False

    @staticmethod
    def get():
        if UnStemmer._instance is None:
            UnStemmer._instance = UnStemmer()

        return UnStemmer._instance

    def __init__(self):
        self.stemmed2original = {}

    def add(self, original_token, stemmed_token):
        original_token = original_token.lower()
        stemmed_token = stemmed_token.lower()

        self.stemmed2original[stemmed_token] = self.stemmed2original.get(stemmed_token, Counter({}))
        self.stemmed2original[stemmed_token][original_token] = self.stemmed2original[stemmed_token].get(original_token, 0) + 1

    def resolve(self, stemmed_token):
        e = self.stemmed2original.get(stemmed_token, None)
        if e is None:
            return stemmed_token

        return e.most_common(1)[0][0]

    def truncate(self):
        dict_truncated = {}

        for stemmed_token in self.stemmed2original.keys():
            original_token, max_count = self.stemmed2original.get(stemmed_token).most_common(1)[0][0]
            if max_count > 1:
                dict_truncated[stemmed_token] = original_token

        return dict_truncated
