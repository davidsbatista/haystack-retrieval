from typing import Tuple, List

from haystack import Document, Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import SentenceTransformersDiversityRanker
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.retrievers import SentenceWindowRetriever, InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

from haystack.components.retrievers import AutoMergingRetriever
from haystack.components.preprocessors import HierarchicalDocumentSplitter

template = """
       You have to answer the following question based on the given context information only.
       If the context is empty or just a '\n' answer with None, example: "None".

       Context:
       {% for document in documents %}
           {{ document.content }}
       {% endfor %}

       Question: {{question}}
       Answer:
       """

def sentence_window(doc_store, embedding_model, top_k):

    template_sentence_window = """
           You have to answer the following question based on the given context information only.
           If the context is empty or just a '\n' answer with None, example: "None".

           Context:
           {% for document in documents %}
               {{ document }}
           {% endfor %}

           Question: {{question}}
           Answer:
           """

    basic_rag = Pipeline()
    basic_rag.add_component(
        "query_embedder", SentenceTransformersTextEmbedder(model=embedding_model, progress_bar=False)
    )
    basic_rag.add_component("retriever", InMemoryEmbeddingRetriever(doc_store, top_k=top_k))
    basic_rag.add_component("sentence_window_retriever", SentenceWindowRetriever(document_store=doc_store))
    basic_rag.add_component("prompt_builder", PromptBuilder(template=template_sentence_window, required_variables=['question', 'documents']))
    basic_rag.add_component("llm", OpenAIGenerator())
    basic_rag.add_component("answer_builder", AnswerBuilder())

    basic_rag.connect("query_embedder", "retriever.query_embedding")
    basic_rag.connect("retriever", "sentence_window_retriever")
    basic_rag.connect("sentence_window_retriever.context_windows", "prompt_builder.documents")
    basic_rag.connect("prompt_builder", "llm")
    basic_rag.connect("llm.replies", "answer_builder.replies")
    basic_rag.connect("llm.meta", "answer_builder.meta")

    # to see the retrieved documents in the answer
    basic_rag.connect("retriever", "answer_builder.documents")

    return basic_rag

def hierarchical_indexing(documents: List[Document], embedding_model: str) -> Tuple[InMemoryDocumentStore, InMemoryDocumentStore]:
    splitter = HierarchicalDocumentSplitter(block_sizes={10, 5}, split_overlap=0, split_by="sentence")
    docs = splitter.run(documents)

    embedder = SentenceTransformersDocumentEmbedder(model=embedding_model, progress_bar=True)
    embedder.warm_up()

    # Store the leaf documents in one document store
    leaf_documents = [doc for doc in docs["documents"] if doc.meta["__level"] == 1]
    print(f"Leaf documents: {len(leaf_documents)}")
    leaf_doc_store = InMemoryDocumentStore()
    embedded_leaf_docs = embedder.run(leaf_documents)
    leaf_doc_store.write_documents(embedded_leaf_docs["documents"], policy=DuplicatePolicy.OVERWRITE)

    # Store the parent documents in another document store
    parent_documents = [doc for doc in docs["documents"] if doc.meta["__level"] == 0]
    print(f"Parent documents: {len(parent_documents)}")
    parent_doc_store = InMemoryDocumentStore()
    embedded_parent_docs = embedder.run(parent_documents)
    parent_doc_store.write_documents(embedded_parent_docs["documents"], policy=DuplicatePolicy.OVERWRITE)

    return leaf_doc_store, parent_doc_store

def auto_merging(leaf_doc_store, parent_doc_store, embedding_model, top_k):

    basic_rag = Pipeline()
    basic_rag.add_component(
        "query_embedder", SentenceTransformersTextEmbedder(model=embedding_model, progress_bar=False)
    )
    basic_rag.add_component("retriever", InMemoryEmbeddingRetriever(leaf_doc_store, top_k=top_k))
    basic_rag.add_component("auto_merging_retriever", AutoMergingRetriever(document_store=parent_doc_store))
    basic_rag.add_component("prompt_builder", PromptBuilder(template=template, required_variables=['question', 'documents']))
    basic_rag.add_component("llm", OpenAIGenerator())
    basic_rag.add_component("answer_builder", AnswerBuilder())

    basic_rag.connect("query_embedder", "retriever.query_embedding")
    basic_rag.connect("retriever", "auto_merging_retriever")
    basic_rag.connect("auto_merging_retriever.documents", "prompt_builder.documents")
    basic_rag.connect("prompt_builder", "llm")
    basic_rag.connect("llm.replies", "answer_builder.replies")
    basic_rag.connect("llm.meta", "answer_builder.meta")

    # to see the retrieved documents in the answer
    basic_rag.connect("retriever", "answer_builder.documents")

    return basic_rag

def mmr(document_store, embedding_model: str, top_k):
    text_embedder = SentenceTransformersTextEmbedder(model=embedding_model, progress_bar=False)
    embedding_retriever = InMemoryEmbeddingRetriever(document_store, top_k=top_k)
    ranker = SentenceTransformersDiversityRanker(strategy="maximum_margin_relevance")

    mmr_pipeline = Pipeline()
    mmr_pipeline.add_component("text_embedder", text_embedder)
    mmr_pipeline.add_component("embedding_retriever", embedding_retriever)
    mmr_pipeline.add_component("ranker", ranker)
    mmr_pipeline.add_component("prompt_builder", PromptBuilder(template=template, required_variables=['question', 'documents']))
    mmr_pipeline.add_component("llm", OpenAIGenerator())
    mmr_pipeline.add_component("answer_builder", AnswerBuilder())

    mmr_pipeline.connect("text_embedder", "embedding_retriever")
    mmr_pipeline.connect("embedding_retriever", "ranker.documents")
    mmr_pipeline.connect("ranker", "prompt_builder.documents")
    mmr_pipeline.connect("prompt_builder", "llm")
    mmr_pipeline.connect("llm.replies", "answer_builder.replies")
    mmr_pipeline.connect("llm.meta", "answer_builder.meta")

    return mmr_pipeline

def hybrid_search(document_store, embedding_model: str, top_k):

    text_embedder = SentenceTransformersTextEmbedder(model=embedding_model, progress_bar=False)
    embedding_retriever = InMemoryEmbeddingRetriever(document_store, top_k=top_k)
    bm25_retriever = InMemoryBM25Retriever(document_store, top_k=top_k)
    document_joiner = DocumentJoiner(join_mode="concatenate")

    hybrid_retrieval = Pipeline()
    hybrid_retrieval.add_component("text_embedder", text_embedder)
    hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
    hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
    hybrid_retrieval.add_component("document_joiner", document_joiner)
    hybrid_retrieval.add_component("prompt_builder", PromptBuilder(template=template, required_variables=['question', 'documents']))
    hybrid_retrieval.add_component("llm", OpenAIGenerator())
    hybrid_retrieval.add_component("answer_builder", AnswerBuilder())
    hybrid_retrieval.connect("text_embedder", "embedding_retriever")
    hybrid_retrieval.connect("bm25_retriever", "document_joiner")
    hybrid_retrieval.connect("embedding_retriever", "document_joiner")
    hybrid_retrieval.connect("document_joiner.documents", "prompt_builder.documents")
    hybrid_retrieval.connect("prompt_builder", "llm.prompt")
    hybrid_retrieval.connect("llm.replies", "answer_builder.replies")
    hybrid_retrieval.connect("llm.meta", "answer_builder.meta")

    return hybrid_retrieval
