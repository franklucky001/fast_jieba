use std::collections::{BTreeSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::{Deref, DerefMut};
use pyo3::prelude::*;
use rayon::prelude::*;
use jieba_rs;
use jieba_rs::KeywordExtract;
use once_cell::sync::{Lazy};

static mut INNER_JIEBA: Lazy<jieba_rs::Jieba> = Lazy::new(|| jieba_rs::Jieba::new());

#[pyclass(subclass)]
pub struct Jieba(&'static mut jieba_rs::Jieba);

#[pyclass]
pub struct TFIDF(jieba_rs::TFIDF<'static>);

#[pyclass]
pub struct TextRank(jieba_rs::TextRank<'static>);

#[pymethods]
impl Jieba {

    #[new]
    fn new() -> Self {
        unsafe {
            Self(INNER_JIEBA.deref_mut())
        }
    }

    #[args(freq = "None", tag = "None")]
    #[pyo3(text_signature = "($self, word, freq, tag)")]
    fn add_word(&mut self, py: Python, word: & str, freq: Option<usize>, tag: Option<&str>) -> usize{
        py.allow_threads(move || self.0.add_word(word, freq, tag))
    }

    #[pyo3(text_signature = "($self, segment)")]
    fn suggest_freq(&mut self, py: Python, segment: & str) -> usize{
        py.allow_threads(move || self.0.suggest_freq(segment))
    }

    #[pyo3(text_signature = "($self, path)")]
    fn load_dict(&mut self, py: Python, path: String){
        py.allow_threads(move || {
            let mut dict_str = String::new();
            File::open(path)
                .expect("file not found")
                .read_to_string(&mut dict_str)
                .expect("read dict failed");
            let mut reader = BufReader::new(dict_str.as_bytes());
            self.0.load_dict(&mut reader).expect("load dict failed")
        })
    }
    ///cut the input sentence, eq python library jieba.cut
    #[args(hmm="true")]
    #[pyo3(text_signature = "($self, text, hmm)")]
    fn cut<'a>(&self, py: Python, text: &'a str, hmm: bool) -> Vec<&'a str>{
        py.allow_threads(move || self.0.cut(text, hmm))
    }

    #[args(hmm="true")]
    #[pyo3(text_signature = "($self, texts, hmm)")]
    fn batch_cut<'a>(&self, py: Python, texts: Vec<&'a str>, hmm: bool) -> Vec<Vec<&'a str>>{
        py.allow_threads(move || texts.into_par_iter().map(|text|self.0.cut(text, hmm)).collect())
    }

    ///cut the input sentence, return all possible words, eq python library jieba.cut_all
    #[pyo3(text_signature = "(&self, text)")]
    fn cut_all<'a>(&self, py: Python, text:&'a str) -> Vec<&'a str>{
        py.allow_threads(move || self.0.cut_all(text))
    }

    #[pyo3(text_signature = "(&self, text)")]
    fn batch_cut_all<'a>(&self, py: Python, texts: Vec<&'a str>) -> Vec<Vec<&'a str>>{
        py.allow_threads(move || texts.into_par_iter().map(|text|self.0.cut_all(text)).collect())
    }

    ///cut for search the input sentence, eq python library jieba.cut_for_search
    #[args(hmm="true")]
    #[pyo3(text_signature = "($self, text, hmm)")]
    fn cut_for_search<'a>(&self, py: Python, text: &'a str, hmm: bool) -> Vec<&'a str>{
        py.allow_threads(move || self.0.cut_for_search(text, hmm))
    }

    #[args(hmm="true")]
    #[pyo3(text_signature = "($self, text, hmm)")]
    fn batch_cut_for_search<'a>(&self, py: Python, texts: Vec<&'a str>, hmm: bool) -> Vec<Vec<&'a str>>{
        py.allow_threads(move || texts.into_par_iter().map(|text|self.0.cut_for_search(text, hmm)).collect())
    }

    ///tag the input sentence, return words and tags
    #[args(hmm="true")]
    #[pyo3(text_signature = "($self, text, hmm)")]
    fn tagging<'a>(&'a self, py: Python, text: &'a str, hmm: bool) -> Vec<(&'a str, &'a str)>{
        py.allow_threads(move || self.0
            .tag(text, hmm)
            .into_iter()
            .map(|item|(item.word, item.tag))
            .collect()
        )
    }

    #[args(hmm="true")]
    #[pyo3(text_signature = "($self, text, hmm)")]
    fn batch_tagging<'a>(&'a self, py: Python, texts: Vec<&'a str>, hmm: bool) -> Vec<Vec<(&'a str, &'a str)>>{
        py.allow_threads(move || texts.into_par_iter().map(|text|{
            self.0.tag(text, hmm)
                .into_par_iter()
                .map(|item|(item.word, item.tag))
                .collect()
        }).collect())
    }

    #[args(mode = "\"default\"", hmm = "true")]
    #[pyo3(text_signature = "($self, text, mode, hmm)")]
    fn tokenize<'a>(
        &self,
        py: Python,
        text: &'a str,
        mode: &str,
        hmm: bool,
    ) -> Vec<(&'a str, usize, usize)> {
        let tokenize_mode = if mode.to_lowercase() == "search" {
            jieba_rs::TokenizeMode::Search
        } else {
            jieba_rs::TokenizeMode::Default
        };
        py.allow_threads(move || {
            self.0
                .tokenize(text, tokenize_mode, hmm)
                .into_iter()
                .map(|t| (t.word, t.start, t.end))
                .collect()
        })
    }

    #[args(mode = "\"default\"", hmm = "true")]
    #[pyo3(text_signature = "($self, text, mode, hmm)")]
    fn batch_tokenize<'a>(
        &self,
        py: Python,
        texts: Vec<&'a str>,
        mode: &str,
        hmm: bool,
    ) -> Vec<Vec<(&'a str, usize, usize)>>{
        py.allow_threads(move || texts.into_par_iter().map(|text|{
            let tokenize_mode = if mode.to_lowercase() == "search" {
                jieba_rs::TokenizeMode::Search
            } else {
                jieba_rs::TokenizeMode::Default
            };
            self.0
                .tokenize(text, tokenize_mode, hmm)
                .into_par_iter()
                .map(|t|(t.word, t.start, t.end))
                .collect()
        }).collect())
    }
}

#[pymethods]
impl TFIDF {

    #[new]
    fn new()-> Self{
        unsafe {
            Self(jieba_rs::TFIDF::new_with_jieba(INNER_JIEBA.deref()))
        }
    }

    #[pyo3(text_signature = "($self)")]
    fn add_stopword(&mut self, py: Python, word: String) -> bool{
        py.allow_threads(move || self.0.add_stop_word(word))
    }

    #[pyo3(text_signature = "($self)")]
    fn remove_stopword(&mut self, py: Python, word: & str) -> bool{
        py.allow_threads(move || self.0.remove_stop_word(word))
    }

    #[pyo3(text_signature = "($self, words)")]
    fn set_stopwords(&mut self, py: Python, words: BTreeSet<String>){
        py.allow_threads(move || self.0.set_stop_words(words))
    }

    #[pyo3(text_signature = "($self, path)")]
    fn set_idf_path(&mut self, py: Python, path: String) {
        let mut reader = BufReader::new(File::open(path).expect("file not found"));
        py.allow_threads(move || {
            self.0.load_dict(&mut reader).expect("load idf file failed")
        })
    }

    #[args(topK = "20", allowPos = "Vec::new()")]
    #[pyo3(text_signature = "($self, text, top_k, allow_pos)")]
    fn extract_tags(&self, py: Python, text: & str, top_k: usize, allow_pos: Vec<String>) -> Vec<(String, f64)>{
        py.allow_threads(move || {
            self.0.extract_tags(text, top_k, allow_pos)
                .into_iter()
                .map(|item|(item.keyword, item.weight))
                .collect()
            }
        )
    }
}

#[pymethods]
impl TextRank{

    #[new]
    fn new() ->Self{
        unsafe {
            Self(jieba_rs::TextRank::new_with_jieba(INNER_JIEBA.deref()))
        }
    }
    #[pyo3(text_signature = "($self)")]
    fn add_stopword(&mut self, py: Python, word: String) -> bool{
        py.allow_threads(move || self.0.add_stop_word(word))
    }

    #[pyo3(text_signature = "($self)")]
    fn remove_stopword(&mut self, py: Python, word: & str) -> bool{
        py.allow_threads(move || self.0.remove_stop_word(word))
    }

    #[pyo3(text_signature = "($self, words)")]
    fn set_stopwords(&mut self, py: Python, words: BTreeSet<String>){
        py.allow_threads(move || self.0.set_stop_words(words))
    }

    #[args(topK = "20", withWeight = "false", allowPos = "Vec::new()")]
    #[pyo3(text_signature = "($self, text, top_k, allow_pos)")]
    fn extract_tags(&self, py: Python, text: & str, top_k: usize, allow_pos: Vec<String>) -> Vec<(String, f64)>{
        py.allow_threads(move || self.0.extract_tags(text, top_k, allow_pos)
            .into_iter()
            .map(|item|(item.keyword, item.weight))
            .collect()
        )
    }
}

#[pymodule]
fn fast_jieba(_py: Python, m: & PyModule) -> PyResult<()>{
    m.add_class::<Jieba>()?;
    m.add_class::<TFIDF>()?;
    m.add_class::<TextRank>()?;
    Ok(())
}
