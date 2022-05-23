from typing import List, Tuple


def textrank(text: str, topK: int = 20, withWeight: bool = False, allowPos: Tuple[str, ...]=()):
    ...


def extract_tags(text, topK:int = 20, withWeight:bool = False, allowPos: Tuple[str, ...]=()):
    ...


def set_stopwords(stopwords: List[str]):
    ...


def load_stopwords(stopwords_file: str):
    ...


def add_stop_word(word: str):
    ...


def remove_stop_word(word: str):
    ...


