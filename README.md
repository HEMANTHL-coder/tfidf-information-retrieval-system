# TF-IDF Based Information Retrieval System

This is a complete mini-project for a **TF-IDF based Information Retrieval (Search Engine)**, designed and built optimally for a university/B.Tech submission.

## 📝 What is TF-IDF?
**TF-IDF** stands for **Term Frequency - Inverse Document Frequency**. It is a cornerstone algorithm in Natural Language Processing and Information Retrieval. It statistically measures how important a word is to a specific document inside a collection (corpus) of documents.

### The Math Explained:
1. **Term Frequency (TF):** Measures how frequently a term appears in a document.
   `TF = (Number of occurrences of a term in Document) / (Total number of words in Document)`
   
2. **Inverse Document Frequency (IDF):** Measures how much information the word provides across the entire corpus. If a word is very common (like "the", "is"), it will have a low IDF. If it's rare, it will have a high IDF.
   `IDF = log10(Total number of documents / Number of documents containing the term)`
   
3. **TF-IDF:**
   `TF-IDF Score = TF * IDF`

By pairing these two metrics, TF-IDF effectively rewards words that heavily categorize a specific text and penalizes uninformative filler words!

## ❓ Why use it in Information Retrieval?
In a Search Engine, we want to match a user's query to the most relevant documents in a database. Using explicit word counts allows bias toward unusually long documents. TF-IDF acts as a normalized weighting scheme ensuring we isolate meaningful keywords!

**Advantages:**
- Simple, predictable, and easy to code from scratch without black-box methods.
- Highly effective for basic informational and exact text matching algorithms.
- Computationally inexpensive compared to modern Neural/Deep Learning models.

**Limitations:**
- Cannot capture semantics (i.e. synonyms like "car" and "automobile" are treated as completely different entities).
- Ignores word order or syntactic context ("Bag-of-Words" strict assumption).

---

## 🛠 Project Architecture
- `core.py`: The heart of the logic. Includes pure Python methods handling String Preprocessing, calculating vectors, and Cosine Similarity equations strictly built with arrays and math dictionaries.
- `app.py`: A very attractive frontend implemented via Streamlit allowing visual interpretation of the underlying matrices.
- `cli.py`: A pure-Python command prompt interface if you prefer reading prints in Terminal.

---

## 🚀 How to Run the Project

### Installing Requirements
You will need Python installed on your PC. It relies solely on Standard Python logic but uses `numpy`, `pandas`, and `streamlit` for fast vector representation and user interfaces.
Run this command in the project folder:
```bash
pip install -r requirements.txt
```

### Option 1: Web Interface (Recommended for showcase)
A Streamlit dashboard was created to showcase "Intermediate values", matrix representations, and word highlighting required for Extra Marks.
```bash
streamlit run app.py
```
*Note: A browser window will automatically open with the UI.*

### Option 2: Command Line Testing
To run the pure scripted version:
```bash
python cli.py
```

---

## 📊 Sample I/O Output

**Input Query:** `"machine learning"`

**Command Line Output Output:**
```
Processed Query Tokens: ['machine', 'learning']

========================================
 RANKED RESULTS (Highest to Lowest)
========================================
Rank 1 | Score: 0.5369 | Doc 5: Artificial intelligence and machine learning are revolutionizing technology.
Rank 2 | Score: 0.4903 | Doc 2: Machine learning is a field of artificial intelligence.
```<img width="865" height="758" alt="Screenshot 2026-03-31 105341" src="https://github.com/user-attachments/assets/bd3de1a6-6240-4659-8867-3d5f8568de8b" />

*(The UI behaves comparably, showcasing yellow text highlighters cleanly onto exact matching words found over the query.)*
<img width="829" height="740" alt="Screenshot 2026-03-31 105405" src="https://github.com/user-attachments/assets/a5c00a7d-512e-4944-8d26-9cc810f1c8f5" />
<img width="849" height="835" alt="Screenshot 2026-03-31 105424" src="https://github.com/user-attachments/assets/114a6726-05ff-467d-a10f-4ed5826ac13d" />
## 📸 Project Demo
![TF-IDF UI](screenshot.png)

Screenshot 2026-03-31 105405.png
Screenshot 2026-03-31 105424.png
