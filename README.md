# Retrieving with Haystack 2.x

This repository contains showcases different retrieval techniques within the context of a RAG-based QA system. 

The retrieval techniques are implemented using the [Haystack 2.x](https://github,com/deepset-ai/haystack) library and 
evaluated using the [ARAGOG dataset](https://github.com/predlico/ARAGOG) with the Semantic Similarity metric.

## Sentence Window Retrieval

<img src="images/sentence_window_retrieval.png" width="75%">

The Sentence Window Retrieval technique uses a sliding window to split the document into chunks and retrieve the most relevant.

## Auto-Merging Retrieval

<img src="images/auto_merging_retrieval.png" width="75%">

Auto-Merging Retrieval retrieves the most relevant chunks of a document and merges them into a single

## Maximum Marginal Relevance

<img src="images/maximum_marginal_relevance.png" width="75%">

Maximum Marginal Relevance ranks documents by selecting first those relevant to the query and dissimilar to the already retrieved. [[1](#1)]

## Hybrid Search Retrieval

<img src="images/hybird_search.png" width="75%">

Hybrid Search Retrieval combines multiple retrieval strategies.

## Multi-Query
 

<img src="images/multi_query.png" width="75%">

Multi-query retrieves documents based on multiple queries generated from the original query.

## Hypothetical Document Embeddings - HyDE

<img src="images/hyde.png" width="75%">

Hypothetical Document Embeddings (HyDE) enhances retrieval by creating and using “fake” hypothetical document based on a query to find similar documents. [[2](#2)] 

## Document Summary Index

<img src="images/document_summary_indexing.png" width="75%">

Document Summary Index leverages document summaries for retrieval and uses full text documents for response generation. [[3](#3)]

# Summary

<img src="images/summary.png" width="75%">

# Experimental Results

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
