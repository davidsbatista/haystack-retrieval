# Retrieving with Haystack 2.x

## Classical Retrieval Techniques

### Sentence-Window Retrieval

During indexing, documents are broken into smaller chunks, or sentences. To provide full context a sentence window 
retriever fetches a number of neighboring sentences before and after each relevant fetched document chunk. Size of the 
window is a parameter that can be set by the user.

### Auto-Merging Retrieval

A retriever which returns parent documents of the matched leaf nodes documents, based on a threshold setting.

The AutoMergingRetriever assumes you have a hierarchical tree structure of documents, where the leaf nodes
are indexed in a document store. See the HierarchicalDocumentSplitter for more information on how to create
such a structure. During retrieval, if the number of matched leaf documents below the same parent is
higher than a defined threshold, the retriever will return the parent document instead of the individual leaf
documents.

The rational is, given that a paragraph is split into multiple chunks represented as leaf documents, and if for
a given query, multiple chunks are matched, the whole paragraph might be more informative than the individual
chunks alone.

### Maximum Marginal Relevance

Aims to balance the relevance and diversity of the retrieved documents.

Most modem IR search engines produce a ranked list of retrieved documents ordered by declining relevance to the user’s query. 

We motivated the need for “relevant novelty” as a potentially superior criterion. A first approximation to 
measuring relevant novelty is to measure relevance and novelty independently and provide a linear combination as the metric. 

We call the linear combination “marginal relevance” - i.e. a document has high marginal relevance if it is both 
relevant to the query and contains minimal similarity to previously selected documents. We strive to maximize-marginal 
relevance in retrieval and summarization, hence we label our method “maximal marginal relevanci” (MMR).

Paper: https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf

### Hybrid Search Retrieval

Vector search which captures the semantic meaning of the query and keyword search which identifies exact matches 
for specific terms. Hybrid search combines the strengths of vector search and keyword search to enhance retrieval accuracy. 

In fields like medicine, many terms and concepts are not semantically understood but are instead specific keywords such 
as medication names, anatomical terms, disease names, and diagnoses.

Pure vector search might miss these critical keywords, whereas keyword-based search ensures that specific, important 
terms are considered. By integrating both methods, hybrid search allows for a more comprehensive retrieval process.

These search, vector and keyword, methods run in parallel, and the results are then merged and ranked according to a 
weighted system. For instance, using Weaviate, you can adjust the alpha parameter to balance the importance of vector 
versus keyword search results, creating a combined, ranked list of documents. This balances precision and recall, 
improving overall retrieval quality but requires careful tuning of weighting parameters.

## LLMs-based Retrieval Techniques

### Multi-query

The Multi-query technique (Langchain, 2023) enhances document retrieval by expanding a single user query into 
multiple similar queries with the assistance of an LLM. 

This process involves generating N alternative questions that echo the intent of the original query but from different 
angles, thereby capturing a broader spectrum of  potential answers. Each query, including the original, is then vectorized 
and subjected to its own retrieval process, which increases the chances of fetching a higher volume of relevant information 
from the document repository. 

To manage the resultant expanded dataset, a re-ranker is often employed, utilizing machine learning models to sift 
through the retrieved chunks and prioritize those most relevant in regards to the initial query. See Figure 3 for an 
overview of how Multi-query RAG system workflow.

https://blog.langchain.dev/query-transformations/

### Hypothetical Document Embeddings - HyDE 

Hypothetical Document Embeddings (HyDE) is a technique proposed in the paper “ Precise Zero-Shot Dense Retrieval 
without Relevance Labels” which improves retrieval by generating “fake” hypothetical documents based on a given query, 
and then uses those “fake” documents embeddings to retrieve similar documents from the same embedding space.

Paper: https://aclanthology.org/2023.acl-long.99/ -  - - HyDE: Hypothetical Document Embeddings for Zero-Shot Dense Retrieva

### Document Summary Index

The Document Summary Index method enhances RAG systems by indexing document summaries for efficient retrieval, while 
providing LLMs with full text documents for response generation (Liu, 2023a). 

This decoupling strategy optimizes  retrieval speed and accuracy through summary-based indexing and supports comprehensive 
response synthesis by  utilizing the original text.


During build-time: 
- We ingest each document, and use a LLM to extract a summary from each document. 
- We also split the document up into text chunks (nodes). 
- Both the summary and the nodes are stored within our Document Store abstraction. 
- We maintain a mapping from the summary to the source document/nodes.

During query-time: 
 
- We retrieve relevant documents to the query based on their summaries, using the following approaches:

  - LLM-based Retrieval: We present sets of document summaries to the LLM, and ask the LLM to determine which documents are relevant + their relevance score.
  - Embedding-based Retrieval: We retrieve relevant documents based on summary embedding similarity (with a top-k cutoff).

Note that this approach of retrieval for document summaries (even with the embedding-based approach) is different than 
embedding-based retrieval over text chunks. The retrieval classes for the document summary index retrieve all nodes for 
any selected document, instead of returning relevant chunks at the node-level.

Storing summaries for a document also enables LLM-based retrieval. Instead of feeding the entire document to the LLM in 
the beginning, we can first have the LLM inspect the concise document summary to see if it’s relevant to the query at all.
This leverages the reasoning capabilities of LLM’s which are more advanced than embedding-based lookup, but avoids the cost/latency of feeding the entire document to the LLM

https://www.llamaindex.ai/blog/a-new-document-summary-index-for-llm-powered-qa-systems-9a32ece2f9ec


## Results

The following table shows the semantic similarity of the answers retrieved by the different techniques.

| Technique                                 | Semantic Answer Similarity |
|-------------------------------------------|----------------------------|
| Sentence-Window Retrieval                 | 0.700                      |
| Auto-Merging Retrieval                    | 0.505                      |
| Baseline RAG + Maximum Marginal Relevance | 0.670                      |
| Hybrid Search                             |                            |
| Multi-Query                               | 0.620                      |
| Hypothetical Document Embeddings - HyDE   | 0.693                      |
| Document Summary Index                    | 0.731                      |


### Dataset
  - ARAGOG dataset

### Parameters
  - embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
  - chunk_size = 15
  - split_by = "sentence"
  - top_k = 3

### Sentence-Window-Retrieval
  - window_size= 3
  - top_k = 3

### Multi-query
  - top_k = 3
  - n_variations = 3

### Baseline RAG + Maximum Marginal Relevance
  - top_k = 3
  - lambda_threshold = 0.5

### Hypothetical Document Embeddings - HyDE
  - top_k = 3
  - nr_completions = 5