import math
import re
import numpy as np

# A basic set of English stopwords to avoid large external dependencies like NLTK datasets
STOPWORDS = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}

def preprocess_text(text):
    """
    Step 1: Lowercase, remove punctuation, tokenize, and remove stopwords.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize by splitting on whitespace
    tokens = text.split()
    # Remove stopwords
    return [word for word in tokens if word not in STOPWORDS]

def compute_tf(document_tokens):
    """
    Step 2: Computes Term Frequency (TF) for a given document.
    TF = (Number of times term T appears in document) / (Total words in document)
    """
    tf_dict = {}
    total_words = len(document_tokens)
    if total_words == 0: return tf_dict
    
    word_counts = {}
    for word in document_tokens:
        word_counts[word] = word_counts.get(word, 0) + 1
        
    for word, count in word_counts.items():
        tf_dict[word] = count / float(total_words)
    return tf_dict

def compute_idf(documents_list):
    """
    Step 3: Computes Inverse Document Frequency (IDF) for a list of documents.
    Uses smoothed IDF: log_10((1 + N) / (1 + df)) + 1
    This matches industry standards (like Scikit-Learn) and prevents zero-division errors.
    """
    N = len(documents_list)
    idf_dict = {}
    doc_count = {}
    
    # Count how many documents contain each word
    for doc_tokens in documents_list:
        unique_words = set(doc_tokens)
        for word in unique_words:
            doc_count[word] = doc_count.get(word, 0) + 1
            
    for word, count in doc_count.items():
        # Smoothed IDF formula
        idf_dict[word] = math.log10((1 + N) / float(1 + count)) + 1
    return idf_dict

def compute_tfidf(tf_dict, idf_dict):
    """
    Step 4: Computes TF-IDF by multiplying TF and IDF values.
    """
    tfidf_dict = {}
    for word, tf in tf_dict.items():
        tfidf_dict[word] = tf * idf_dict.get(word, 0.0)
    return tfidf_dict

def build_vocabulary(documents_list):
    """
    Helper: Creates a sorted list of all unique words across all documents (Corpus).
    """
    vocab = set()
    for doc in documents_list:
        vocab.update(doc)
    return sorted(list(vocab))

def vectorize(tfidf_dict, vocab):
    """
    Helper: Converts a TF-IDF dictionary to an ordered numerical vector based on the vocabulary.
    """
    return [tfidf_dict.get(word, 0.0) for word in vocab]

def cosine_similarity(vec1, vec2):
    """
    Step 5: Computes cosine similarity between two vectors.
    Cosine Similarity = (A . B) / (||A|| * ||B||)
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Prevent division by zero if a vector is completely empty
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)
