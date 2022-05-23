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
    _TFIDF.set_stop_words(stopwords)
    _TextRank.set_stop_words(stopwords)


def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f:
            add_stop_word(line.strip())


def add_stop_word(word):
    _TFIDF.add_stop_word(word)
    _TextRank.add_stop_word(word)


def remove_stop_word(word):
    _TFIDF.remove_stop_word(word)
    _TextRank.remove_stop_word(word)


