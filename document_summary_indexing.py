import os
from pathlib import Path
from typing import List

from haystack import Pipeline, component, Document
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

from openai_summarisation import summarize


@component
class Summarizer:

    def __init__(self, model: str = 'gpt-4o-mini'):
        self.model = model

    @component.output_types(summary=List[Document])
    def run(self, documents: List[Document], detail: float = 0.05):
        summaries = []
        for doc in documents:
            summary = summarize(doc.content, detail=detail, model=self.model)
            summaries.append(Document(content=summary, meta=doc.meta))
        return {"summary": summaries}


def indexing_doc_summarisation(embedding_model: str, base_path: str) -> InMemoryDocumentStore:
    """
    Summary: document → cleaner → summarizer → embedder → writer
    """
    full_path = Path(base_path)
    files_path = full_path / "papers_for_questions"
    summaries_doc_store = InMemoryDocumentStore()
    summary_indexing_pipeline = Pipeline()
    summary_indexing_pipeline.add_component("converter", PyPDFToDocument())
    summary_indexing_pipeline.add_component("cleaner", DocumentCleaner())
    summary_indexing_pipeline.add_component("summarizer", Summarizer())
    summary_indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model=embedding_model))
    summary_indexing_pipeline.add_component("writer", DocumentWriter(document_store=summaries_doc_store))

    summary_indexing_pipeline.connect("converter", "cleaner")
    summary_indexing_pipeline.connect("cleaner", "summarizer")
    summary_indexing_pipeline.connect("summarizer", "embedder")
    summary_indexing_pipeline.connect("embedder", "writer")

    pdf_files = [files_path / f_name for f_name in os.listdir(files_path)]
    summary_indexing_pipeline.run({"converter": {"sources": pdf_files[0:5]}})

    return summaries_doc_store

def doc_summarisation_query_pipeline(chunk_doc_store, summary_doc_store, embedding_model):
    """
    Two levels of retrieval:

    Document-Level Retrieval:
        uses the summary index to identify the top-k most relevant to the query and document summaries.

    Chunk-Level Retrieval:
        Once the relevant documents are identified, use the document IDs from the previous step. For each document, the
        most relevant chunks to the query are retrieved.
    """
    pass