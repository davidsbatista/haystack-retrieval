import os
from pathlib import Path
from typing import List

from haystack import Pipeline, component, Document
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
from haystack.components.converters import PyPDFToDocument, OutputAdapter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.dataclasses.chat_message import ChatMessage

from techniques.llm.openai_summarisation import summarize

from tqdm import tqdm

@component
class Summarizer:

    def __init__(self, model: str = 'gpt-4o-mini'):
        self.model = model

    @component.output_types(summary=List[Document])
    def run(self, documents: List[Document], detail: float = 0.05):
        summaries = []
        for doc in tqdm(documents):
            summary = summarize(doc.content, detail=detail, model=self.model)
            summaries.append(Document(content=summary, meta=doc.meta, id=doc.id))
        return {"summary": summaries}

@component
class ChunksRetriever:

    def __init__(self, chunk_doc_store):
        self.chunk_doc_store = chunk_doc_store

    @component.output_types(chunks=List[Document])
    def run(self, doc_ids: List[str], query_embedding: List[float]):
        context_docs = []
        for doc_id in doc_ids:
            results = self.chunk_doc_store.embedding_retrieval(
                query_embedding=query_embedding,
                filters={
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.source_id", "operator": "==", "value": doc_id},
                    ],
                }
            )
            context_docs.extend(results)

        return {"chunks": context_docs}

def indexing_doc_summarisation_arago(embedding_model: str, base_path: str, chunk_size = 15):
    """
    Summary: document → cleaner → summarizer → embedder → writer
    """
    full_path = Path(base_path)
    files_path = full_path / "papers_for_questions"

    chunk_doc_store = InMemoryDocumentStore()
    summaries_doc_store = InMemoryDocumentStore()

    indexing = Pipeline()
    indexing.add_component("converter", PyPDFToDocument())
    indexing.add_component("cleaner", DocumentCleaner())

    # summary
    indexing.add_component("summarizer", Summarizer())
    indexing.add_component("summary_embedder", SentenceTransformersDocumentEmbedder(model=embedding_model))
    indexing.add_component("summary_writer", DocumentWriter(document_store=summaries_doc_store))

    # chunks
    indexing.add_component("splitter", DocumentSplitter(split_length=chunk_size, split_overlap=0, split_by="sentence"))
    indexing.add_component("chunk_embedder", SentenceTransformersDocumentEmbedder(model=embedding_model, progress_bar=False))
    indexing.add_component("chunk_writer", DocumentWriter(document_store=chunk_doc_store, policy=DuplicatePolicy.SKIP))

    #  connections
    indexing.connect("converter", "cleaner")

    # connect components for summary
    indexing.connect("cleaner", "summarizer")
    indexing.connect("summarizer", "summary_embedder")
    indexing.connect("summary_embedder", "summary_writer")

    # connect components for chunks
    indexing.connect("cleaner", "splitter")
    indexing.connect("splitter", "chunk_embedder")
    indexing.connect("chunk_embedder", "chunk_writer")

    pdf_files = [files_path / f_name for f_name in os.listdir(files_path)]
    indexing.run({"converter": {"sources": pdf_files}})

    return summaries_doc_store, chunk_doc_store

def indexing_doc_summarisation_hotpot(embedding_model: str, documents, chunk_size = 15):
    """
    Summary: document → cleaner → summarizer → embedder → writer
    """

    summaries_doc_store = InMemoryDocumentStore()
    indexing_summaries = Pipeline()
    summariser = Summarizer()
    embedder = SentenceTransformersDocumentEmbedder(model=embedding_model, progress_bar=False)
    doc_writer = DocumentWriter(document_store=summaries_doc_store, policy=DuplicatePolicy.SKIP)
    indexing_summaries.add_component("summarizer", summariser)
    indexing_summaries.add_component("summary_embedder", embedder)
    indexing_summaries.add_component("summary_writer", doc_writer)
    indexing_summaries.connect("summarizer", "summary_embedder")
    indexing_summaries.connect("summary_embedder", "summary_writer")
    indexing_summaries.run({"summarizer": {"documents": documents}})

    chunk_doc_store = InMemoryDocumentStore()
    indexing_chunks = Pipeline()
    indexing_chunks.add_component("splitter", DocumentSplitter(
        split_by="sentence", split_length=15, split_overlap=0)
    )
    indexing_chunks.add_component("writer", DocumentWriter(document_store=chunk_doc_store, policy=DuplicatePolicy.SKIP))
    indexing_chunks.add_component("embedder", SentenceTransformersDocumentEmbedder(embedding_model))
    indexing_chunks.connect("splitter", "embedder")
    indexing_chunks.connect("embedder", "writer")
    indexing_chunks.run(data={"splitter": {"documents": documents}})


    return summaries_doc_store, chunk_doc_store

def doc_summarisation_query_pipeline(chunk_doc_store, summaries_doc_store, embedding_model, top_k, template=None):
    """
    Two levels of retrieval:

    Document-Level Retrieval:
        uses the summary index to identify the top-k most relevant to the query and document summaries.

    Chunk-Level Retrieval:
        Once the relevant documents are identified, use the document IDs from the previous step. For each document, the
        most relevant chunks to the query are retrieved.
    """

    text_embedder = SentenceTransformersTextEmbedder(model=embedding_model, progress_bar=False)
    summary_embedding_retriever = InMemoryEmbeddingRetriever(summaries_doc_store, top_k=top_k)
    chunk_embedding_retriever = ChunksRetriever(chunk_doc_store)

    # This is equivalent to : List[Document] -> lambda docs: [doc.id for doc in docs] -> List[str]
    output_adapter = OutputAdapter(
        template="{{ documents | converter }}",
        output_type=List[str],
        custom_filters={'converter': lambda docs: [doc.id for doc in docs]}
    )

    default = [
        ChatMessage.from_system(
            "You are a helpful AI assistant. Answer the following question based on the given context information only. "
            "If the context is empty or just a '\n' answer with None, example: 'None'."
        ),
        ChatMessage.from_user(
            """
            Context:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            Question: {{question}}
            """
        )
    ]

    template = template if template else default

    doc_summary_query = Pipeline()
    doc_summary_query.add_component("text_embedder", text_embedder)
    doc_summary_query.add_component("summary_retriever", summary_embedding_retriever)
    doc_summary_query.add_component("chunk_embedding_retriever", chunk_embedding_retriever)
    doc_summary_query.add_component("output_adapter", output_adapter)
    doc_summary_query.add_component("prompt_builder", ChatPromptBuilder(template=template, required_variables=["question", "documents"]))
    doc_summary_query.add_component("llm", OpenAIChatGenerator())
    doc_summary_query.add_component("answer_builder", AnswerBuilder())

    doc_summary_query.connect("text_embedder", "summary_retriever")
    doc_summary_query.connect("summary_retriever", "output_adapter")
    doc_summary_query.connect("text_embedder", "chunk_embedding_retriever.query_embedding")
    doc_summary_query.connect("output_adapter.output", "chunk_embedding_retriever.doc_ids")

    doc_summary_query.connect("chunk_embedding_retriever.chunks", "prompt_builder.documents")
    doc_summary_query.connect("prompt_builder", "llm")
    doc_summary_query.connect("llm.replies", "answer_builder.replies")

    return doc_summary_query