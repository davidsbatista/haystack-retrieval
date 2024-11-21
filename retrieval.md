
# Classic Retrieval techniques

## Sentence-Window Retrieval

During indexing, documents are broken into smaller chunks, or sentences. To provide full context a sentence window 
retriever fetches a number of neighboring sentences before and after each relevant fetched document chunk. Size of the 
window is a parameter that can be set by the user.

## Auto-Merging Retrieval

A retriever which returns parent documents of the matched leaf nodes documents, based on a threshold setting.

The AutoMergingRetriever assumes you have a hierarchical tree structure of documents, where the leaf nodes
are indexed in a document store. See the HierarchicalDocumentSplitter for more information on how to create
such a structure. During retrieval, if the number of matched leaf documents below the same parent is
higher than a defined threshold, the retriever will return the parent document instead of the individual leaf
documents.

The rational is, given that a paragraph is split into multiple chunks represented as leaf documents, and if for
a given query, multiple chunks are matched, the whole paragraph might be more informative than the individual
chunks alone.

## Maximum Marginal Relevance

Aims to balance the relevance and diversity of the retrieved documents.

Most modem IR search engines produce a ranked list of retrieved documents ordered by declining relevance to the user’s query. 

We motivated the need for “relevant novelty” as a potentially superior criterion. A first approximation to 
measuring relevant novelty is to measure relevance and novelty independently and provide a linear combination as the metric. 

We call the linear combination “marginal relevance” - i.e. a document has high marginal relevance if it is both 
relevant to the query and contains minimal similarity to previously selected documents. We strive to maximize-marginal 
relevance in retrieval and summarization, hence we label our method “maximal marginal relevanci” (MMR).

Paper: https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf

## Hybrid Retrieval

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

## Reciprocal Rank fusion

Aims at combining the results of multiple retrieval systems to improve the overall performance.

https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

https://carbon.ai/blog/reciprocal-rank-fusion


# LLMs-based Retrieval techniques

## Document Summary Index

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

## Hypothetical Document Embeddings - HyDE 

Hypothetical Document Embeddings (HyDE) is a technique proposed in the paper “ Precise Zero-Shot Dense Retrieval 
without Relevance Labels” which improves retrieval by generating “fake” hypothetical documents based on a given query, 
and then uses those “fake” documents embeddings to retrieve similar documents from the same embedding space.

Paper: https://aclanthology.org/2023.acl-long.99/ -  - - HyDE: Hypothetical Document Embeddings for Zero-Shot Dense Retrieva

## Multi-query

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

## Contextual Retrieval

https://www.anthropic.com/news/contextual-retrieval


# Ranking techniques

## Lost in the Middle

The Lost in the Middle (LLM) model is a simple and effective method for reranking the top-k retrieved documents.

Ranks documents based on the 'lost in the middle' order so that the most relevant documents are either at the 
beginning or end, while the least relevant are in the middle.

Paper: https://aclanthology.org/2024.tacl-1.9/

## Cross-encoder based rerankers

Rerankers aim to enhance the RAG process by refining the selection of documents retrieved in response to a query, with 
the goal of prioritizing the most relevant and contextually appropriate information for generating responses (Pinecone, 2023). 

This step employs ML algorithms (such as cross-encoder) to reassess the initially retrieved set, using criteria that 
extend beyond cosine similarity. Through this evaluation, rerankers are expected to improve the input for generative 
models, potentially leading to more accurate and contextually rich outputs. See Figure 4 for an overview of the 
Reranker RAG system workflow. One tool in this domain is Cohere rerank, which uses a cross-encoder architecture to 
assess the relevance of documents to the query. This approach differs from methods that process queries and documents 
separately, as cross-encoders analyze them jointly, which could allow for a more comprehensive understanding of 
their mutual relevance.


Ranks Documents based on their similarity to the query using [Cohere models](https://docs.cohere.com/reference/rerank-1).

## LLM Rerank

Following the introduction of cross-encoder based rerankers such as Cohere rerank, the LLM reranker offers an alternative 
strategy by directly applying LLMs to the task of reranking retrieved documents (Liu, 2023b). This method prioritizes 
the comprehensive analytical abilities of LLMs over the joint query-document analysis typical of cross-encoders. 

Although less efficient in terms of processing speed and cost compared to cross-encoder models, LLM rerankers can 
achieve higher accuracy by leveraging the advanced understanding of language and context inherent in LLMs. This makes 
the LLM reranker suitable for applications where the quality of the reranked results is more critical than 
computational efficiency.


# Graph-RAG

### Indexing Phase: 

- Documents are broken down into chunks, and key elements such as entities, relationships, together with related claims or attributes. 

- Entities are nodes, and relationships are edges in the graph all with a descriptive summary.
  - NOTE: Different from typical knowledge graphs, i.e.: consistent knowledge triples (subject, predicate, object) for downstream reasoning tasks.

- Use a hierarchical community structure detection algorithms to divide the graph into communities - Leiden Algorithm (Traag et al., 2019)
  - Each level of the hierarchy provides a community partition that covers the nodes of the graph in a mutually-exclusive way, enabling divide-and-conquer global summarization.
  - A user may scan through community summaries at one level looking for general themes of interest, then follow links to the reports at the lower level that provide more details for each of the subtopics.

- Create report-like summaries of each community in the Leiden hierarchy,
  - Leaf-level communities. The element summaries of a leaf-level community (nodes, edges, covariates) are prioritized and then iteratively 
    added to the LLM context window until the token limit is reached. The prioritization is as follows: for each community edge in decreasing order of 
    combined source and target node degree (i.e., overall prominance), add descriptions of the source node, target node, linked covariates, and the edge itself.
  - Higher-level communities. If all element summaries fit within the token limit of the context window, proceed as for leaf-level communities 
    and summarize all element summaries within the community. Otherwise, rank sub-communities in decreasing order of element summary tokens and iteratively 
    substitute sub-community summaries (shorter) for their associated element summaries (longer) until fit within the context window is achieved.

### Querying Phase: 

When a user asks a question, we use the community summaries to generate intermediate answers.  These answers are then 
combined to produce a final global answer to the user query.

    - Prepare community summaries. Community summaries are randomly shuffled and divided into chunks of pre-specified token size.
      This ensures relevant information is distributed across chunks, rather than concentrated (and potentially lost) in a single context window. 

    - Map community answers. Generate intermediate answers in parallel, one for each chunk. The LLM is also asked to generate 
      a score between 0-100 indicating how helpful the generated answer is in answering the target question. Answers with score 0 are filtered out. 

    - Reduce to global answer. Intermediate community answers are sorted in descending order of helpfulness score and iteratively added 
      into a new context window until the token limit is reached. This final context is used to generate the global answer returned to the user.


https://medium.com/@jaideepsachdev1/understanding-graph-rag-simplified-for-enthusiasts-and-beginners-58ec00a3c1ec