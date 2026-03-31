import streamlit as st
import pandas as pd
import re
from core import preprocess_text, compute_tf, compute_idf, compute_tfidf, cosine_similarity, build_vocabulary, vectorize

st.set_page_config(page_title="TF-IDF Search Engine", layout="wide", page_icon="🔍")

st.title("🔍 TF-IDF Based Information Retrieval System")
st.markdown("A mini-project demonstrating a custom search engine functioning on pure TF-IDF and Cosine Similarity, ideal for a B.Tech university project. Upload documents, enter a query, and watch the math unfold!")

# Default document corpus
default_docs = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a field of artificial intelligence.",
    "Natural language processing enables computers to understand human language.",
    "The lazy dog sleeps all day in the sun.",
    "Artificial intelligence and machine learning are revolutionizing technology.",
    "Python is a great programming language for data science and AI."
]

tab1, tab2, tab3 = st.tabs(["📄 1. Documents & Data", "🧮 2. Under the Hood (Math)", "🔍 3. Search Engine"])

with tab1:
    st.header("Document Collection")
    st.markdown("Upload a text file (one document per line) or manually edit the default documents below.")
    
    uploaded_file = st.file_uploader("Upload a text document (.txt)", type="txt")
    
    if uploaded_file is not None:
        file_contents = uploaded_file.getvalue().decode("utf-8")
        default_text_val = file_contents
    else:
        default_text_val = "\n".join(default_docs)
        
    docs_text = st.text_area("Edit Documents (Enter one document per line):", value=default_text_val, height=250)
    docs = [d.strip() for d in docs_text.split("\n") if d.strip()]
    if not docs:
        st.error("Please enter at least one document.")
        st.stop()

# ================================
# Background Computation Process
# ================================
processed_docs = [preprocess_text(d) for d in docs]
tf_docs = [compute_tf(d) for d in processed_docs]
idf = compute_idf(processed_docs)
tfidf_docs = [compute_tfidf(tf, idf) for tf in tf_docs]
vocab = build_vocabulary(processed_docs)

# Create Matrix
matrix_data = [vectorize(tfidf, vocab) for tfidf in tfidf_docs]
df_tfidf = pd.DataFrame(matrix_data, columns=vocab, index=[f"Doc {i+1}" for i in range(len(docs))])

with tab2:
    st.header("Intermediate Steps (Math!)")
    
    st.subheader("TF-IDF Matrix")
    st.dataframe(df_tfidf, use_container_width=True)
    
    # Export File Feature
    csv = df_tfidf.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download TF-IDF Matrix as CSV",
        data=csv,
        file_name='tfidf_matrix.csv',
        mime='text/csv',
    )
    
    with st.expander("View Vocabulary and IDF values"):
        st.markdown("**Corpus Vocabulary:**")
        st.write(vocab)
        st.markdown("**Smoothed IDF Values:**")
        idf_df = pd.DataFrame(list(idf.items()), columns=["Term", "IDF Score"]).set_index("Term")
        st.dataframe(idf_df.T)

with tab3:
    st.header("Run Search Query")
    query = st.text_input("Enter your search query string:", value="machine learning intelligence")
    
    if query:
        st.subheader("Results")
        processed_query = preprocess_text(query)
        st.write(f"**Processed Query Tokens:** `{processed_query}`")
        
        # Calculate Query Vector
        query_tf = compute_tf(processed_query)
        query_tfidf = compute_tfidf(query_tf, idf)
        query_vector = vectorize(query_tfidf, vocab)
        
        with st.expander("View Query Vector Details"):
            st.write("**Query Term Frequency:**", query_tf)
            st.write("**Query TF-IDF Representation:**", query_tfidf)
    
        # Calculate Similarities
        similarities: list[tuple[int, float]] = []
        for i, doc_vector in enumerate(matrix_data):
            sim = cosine_similarity(query_vector, doc_vector)
            similarities.append((int(i), float(sim)))
            
        # Ranking
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if similarities[0][1] == 0:
            st.warning("No matching documents found. Try checking for different keywords.")
        else:
            for rank, (doc_idx, sim) in enumerate(similarities):
                rank_int = int(rank)
                doc_idx_int = int(doc_idx)
                if sim > 0:
                    original_text = docs[doc_idx_int]
                    
                    # Extra Feature: Highlight query terms in result
                    highlighted_text = original_text
                    for token in processed_query:
                        # Regex replacement retaining the original casing
                        highlighted_text = re.sub(
                            f'(?i)\\b({token})\\b', 
                            r'<mark style="background-color: #ffd166; color: black; border-radius: 3px; padding: 0 2px;">\1</mark>', 
                            highlighted_text
                        )
                    
                    st.markdown(f"#### Rank {rank_int+1} (Doc {doc_idx_int+1})")
                    
                    # Clip float to 0.0 - 1.0 safely for st.progress
                    safe_sim = float(min(max(sim, 0.0), 1.0))
                    st.progress(safe_sim)
                    
                    st.markdown(f"**Similarity Score:** `{sim:.4f}`")
                    st.markdown(f"> {highlighted_text}", unsafe_allow_html=True)
                    st.write("---")
