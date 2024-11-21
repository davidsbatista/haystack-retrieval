import os
from pathlib import Path

from haystack import Pipeline, component, Document
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

from openai_summarisation import summarize

# Query:
#Document-Level Retrieval: uses the summary index to identify the top-k most relevant to the query and document summaries.
# (This approach efficiently narrows the search space by retrieving only the most pertinent document IDs)
# Chunk-Level Retrieval: Once the relevant documents are identified, use the document IDs from step 1. For each document, the most relevant chunks to the query are retrieved.


@component
class Summarizer:

    def __init__(self, model: str = 'gpt-4-turbo'):
        self.model = model

    @component.output_types(summary=Document)
    def run(self, text: str, detail: float = 0):
        return summarize(text, detail=detail, model=self.model)



def indexing_doc_summarisation(embedding_model: str, chunk_size: int, base_path: str) -> InMemoryDocumentStore:
    """
    Indexing:
        an LLM to generate a summary (single chunk) for each document (Summary Index).
        split each document up into chunks (Chunk Index).
        Maintain a mapping from summary and chunks to original document

    Create two parallel processing paths:
        Summary path: document → cleaner → summarizer → embedder → writer
        Chunk path: document → cleaner → splitter → embedder → writer
    """
    full_path = Path(base_path)
    files_path = full_path / "papers_for_questions"
    document_store = InMemoryDocumentStore()
    summary_indexing_pipeline = Pipeline()
    summary_indexing_pipeline.add_component("converter", PyPDFToDocument())
    summary_indexing_pipeline.add_component("cleaner", DocumentCleaner())

    # summary_indexing_pipeline.add_component("summarizer", OpenAIDocumentSummarizer())
    # summary_indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model=embedding_model))
    # summary_indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    summary_indexing_pipeline.connect("converter", "cleaner")

    pdf_files = [files_path / f_name for f_name in os.listdir(files_path)]
    documents = summary_indexing_pipeline.run({"converter": {"sources": pdf_files}})


    return document_store

