# Retrieving with Haystack 2.x

## Classical Retrieval Techniques

- __Sentence-Window Retrieval__ uses a sliding window to split the document into chunks and retrieve the most relevant
- __Auto-Merging Retrieval__ retrieves the most relevant chunks of a document and merges them into a single
- __Maximum Marginal Relevance__ ranks documents by selecting first those relevant to the query and dissimilar to the already retrieved. [[1](#1)]
- __Hybrid Search Retrieval__ combines multiple retrieval strategies.

## LLMs-based Retrieval Techniques

- __Multi-query__ retrieves documents based on multiple queries generated from the original query.
- __Hypothetical Document Embeddings - HyDE__ enhances retrieval by creating and using “fake” hypothetical document based on a query to find similar documents. [[2](#2)] 
- __Document Summary Index__ leverages document summaries for retrieval and uses full text documents for response generation. [[3](#3)]

## Experimental Results

The following table shows the semantic similarity of the answers retrieved by the different techniques over the [ARAGOG 
dataset](https://github.com/predlico/ARAGOG). The results are obtained by comparing the retrieved answers with the ground truth answers using the Semantic
Similarity metric.

| Technique                                 | Semantic Answer Similarity |
|-------------------------------------------|----------------------------|
| Sentence-Window Retrieval                 | 0.700                      |
| Auto-Merging Retrieval                    | 0.505                      |
| Baseline RAG + Maximum Marginal Relevance | 0.670                      |
| Hybrid Search                             | 0.699                      |
| Multi-Query                               | 0.620                      |
| Hypothetical Document Embeddings - HyDE   | 0.693                      |
| Document Summary Index                    | 0.731                      |

## References

1. <a name="1"></a>[The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)
2. <a name="2"></a>[Hypothetical Document Embeddings - HyDE](https://aclanthology.org/2023.acl-long.99/)
3. <a name="3"></a>[A New Document Summary Index for LLM-Powered QA Systems](https://www.llamaindex.ai/blog/a-new-document-summary-index-for-llm-powered-qa-systems-9a32ece2f9ec)
