from typing import Tuple, List

from haystack import Pipeline, Document
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_experimental.components.retrievers import AutoMergingRetriever
from haystack_experimental.components.splitters import HierarchicalDocumentSplitter


def hierarchical_indexing(documents: List[Document], embedding_model: str) -> Tuple[InMemoryDocumentStore, InMemoryDocumentStore]:
    splitter = HierarchicalDocumentSplitter(block_sizes={20, 10, 5}, split_overlap=0, split_by="sentence")
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


def auto_merging_retrieval(leaf_doc_store, parent_doc_store, embedding_model, top_k=1):
    template = """
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
    basic_rag.add_component("retriever", InMemoryEmbeddingRetriever(leaf_doc_store, top_k=top_k))
    basic_rag.add_component("auto_merging_retriever", AutoMergingRetriever(document_store=parent_doc_store))
    basic_rag.add_component("prompt_builder", PromptBuilder(template=template))
    basic_rag.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
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