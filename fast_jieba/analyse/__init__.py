from fast_jieba.fast_jieba import TextRank, TFIDF

_TextRank = TextRank()


def textrank(text, topK=20, withWeight=False, allowPos=()):
    if withWeight:
        return _TextRank.extract_tags(text, topK, allowPos)
    else:
        return [item[0] for item in _TextRank.extract_tags(text, topK, allowPos)]


_TFIDF = TFIDF()


def extract_tags(text, topK=20, withWeight=False, allowPos=()):
    if withWeight:
        return _TFIDF.extract_tags(text, topK, allowPos)
    else:
        return [item[0] for item in _TFIDF.extract_tags(text, topK, allowPos)]


def set_stopwords(stopwords):
    _TFIDF.set_stopwords(stopwords)
    _TextRank.set_stopwords(stopwords)


def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f:
            add_stop_word(line.strip())


def add_stopword(word):
    _TFIDF.add_stopword(word)
    _TextRank.add_stopword(word)


def remove_stopword(word):
    _TFIDF.remove_stopword(word)
    _TextRank.remove_stopword(word)


