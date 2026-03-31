from core import preprocess_text, compute_tf, compute_idf, compute_tfidf, cosine_similarity, build_vocabulary, vectorize
import pandas as pd

def main():
    print("="*60)
    print(" TF-IDF Information Retrieval System (Command Line Edition)")
    print("="*60)
    
    # Sample Collection of Documents
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a field of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "The lazy dog sleeps all day in the sun.",
        "Artificial intelligence and machine learning are revolutionizing technology."
    ]
    
    print("\n--- Document Collection ---")
    for i, d in enumerate(docs):
        print(f"Doc {i+1}: {d}")
        
    # Step 1: Preprocess documents
    processed_docs = [preprocess_text(d) for d in docs]
    
    # Step 2: Compute Term Frequencies (TF)
    tf_docs = [compute_tf(d) for d in processed_docs]
    
    # Step 3: Compute Inverse Document Frequency (IDF)
    idf = compute_idf(processed_docs)
    
    # Step 4: Compute TF-IDF
    tfidf_docs = [compute_tfidf(tf, idf) for tf in tf_docs]
    
    # Build vocabulary to form matrices
    vocab = build_vocabulary(processed_docs)
    
    # Printing Intermediate Steps as requested
    print("\n[Intermediate Step] Vocabulary Built:")
    print(vocab)
    
    print("\n[Intermediate Step] IDF Values per term:")
    for word, val in idf.items():
        print(f"{word}: {val:.4f}")
        
    print("\n[Intermediate Step] TF-IDF Matrix:")
    matrix_data = []
    for tfidf in tfidf_docs:
        matrix_data.append(vectorize(tfidf, vocab))
    
    # Using Pandas strictly for a neat tabular display of the Matrix
    df_tfidf = pd.DataFrame(matrix_data, columns=vocab, index=[f"Doc {i+1}" for i in range(len(docs))])
    print(df_tfidf.to_string())
    
    # ==============================
    # Query Phase
    # ==============================
    query = input("\nEnter your search query: ")
    if not query.strip():
        query = "machine learning artificial intelligence"
        print(f"No query entered! Using default query: '{query}'")
        
    # Preprocess Query
    processed_query = preprocess_text(query)
    print(f"\nProcessed Query Tokens: {processed_query}")
    
    # Compute TF and TF-IDF for Query
    query_tf = compute_tf(processed_query)
    query_tfidf = compute_tfidf(query_tf, idf)
    print("\n[Intermediate Step] Query TF-IDF dictionary:")
    print(query_tfidf)
    
    # Vectorize Query against corpus vocabulary
    query_vector = vectorize(query_tfidf, vocab)
    
    # Step 5: Rank Documents using Cosine Similarity
    similarities = []
    for i, doc_vector in enumerate(matrix_data):
        sim = cosine_similarity(query_vector, doc_vector)
        similarities.append((i, sim))
        
    # Sort by similarity, highest first
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*40)
    print(" RANKED RESULTS (Highest to Lowest)")
    print("="*40)
    
    if similarities[0][1] == 0:
        print("No matches found for your query. Try different words.")
    else:
        for rank, (doc_idx, sim) in enumerate(similarities):
            # Show documents that have at least some similarity
            if sim > 0:
                print(f"Rank {rank+1} | Score: {sim:.4f} | Doc {doc_idx+1}: {docs[doc_idx]}")

if __name__ == "__main__":
    main()
