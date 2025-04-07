from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

from evaluations import (
    auto_merging_eval,
    doc_summary_indexing,
    hybrid_search_eval,
    hyde_eval,
    maximum_marginal_relevance_reranking,
    multi_query_eval,
    sentence_window_eval
)

def indexing(embedding_model: str, chunk_size: int, documents: list[Document]) -> InMemoryDocumentStore:

    document_store = InMemoryDocumentStore()
    pipeline = Pipeline()

    pipeline.add_component("splitter", DocumentSplitter(split_length=chunk_size, split_overlap=0, split_by="sentence"))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(embedding_model))
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    pipeline.run({"splitter": {"documents": documents}})

    return document_store


def main():

    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size = 15
    top_k = 3
    hyde_n_completions = 3
    multi_query_n_variations = 3
    n_questions = 10

    dataset = load_dataset("vblagoje/PubMedQA_instruction", split="train")
    subset_dataset = dataset.select(range(n_questions))
    documents = [Document(content=doc["context"]) for doc in subset_dataset]
    questions = [doc["instruction"] for doc in subset_dataset]
    answers = [doc["response"] for doc in subset_dataset]

    print("Indexing documents...")
    doc_store = indexing(embedding_model, chunk_size, documents)

    print("Number of documents indexed:")
    print(len(doc_store.storage.values()))

    # classical techniques
    print("\n\nSentence Window")
    sentence_window_eval(answers, doc_store, embedding_model, questions, top_k)
    """
    print("\n\nAuto Merging")
    auto_merging_eval(answers, documents, embedding_model, questions, top_k)
    print("\n\nMaximum Marginal Relevance Reranking")
    maximum_marginal_relevance_reranking(answers, doc_store, embedding_model, questions, top_k)
    print("\n\nHybrid Search")
    hybrid_search_eval(answers, doc_store, embedding_model, questions, top_k)

    # LLM-based techniques
    print("\n\nHyDE")
    hyde_eval(answers, doc_store, embedding_model, questions, hyde_n_completions, top_k)
    print("\n\nMulti Query")
    multi_query_eval(answers, doc_store, embedding_model, questions, multi_query_n_variations, top_k)
    print("\n\nDoc Summarisation")
    doc_summary_indexing(embedding_model, documents, questions, answers, top_k)
    """


if __name__ == '__main__':
    main()
