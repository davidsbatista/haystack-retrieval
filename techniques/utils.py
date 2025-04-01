import json
import os
from pathlib import Path
from typing import List
from typing import Tuple

from haystack import Pipeline, Document
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.evaluators import SASEvaluator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

def read_question_answers(base_path: str) -> Tuple[List[str], List[str]]:
    with open(base_path + "eval_questions.json", "r") as f:
        data = json.load(f)
        questions = data["questions"]
        answers = data["ground_truths"]
    return questions, answers

def transform_pdf_to_documents(base_path: str) -> List[Document]:
    full_path = Path(base_path)
    files_path = full_path / "papers_for_questions"
    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.connect("converter", "cleaner")
    pdf_files = [full_path / "papers_for_questions" / f_name for f_name in os.listdir(files_path)]
    pdf_documents = pipeline.run({"converter": {"sources": pdf_files}})

    return pdf_documents['cleaner']['documents']

def run_evaluation(sample_questions, sample_answers, retrieved_contexts, predicted_answers, embedding_model):
    eval_pipeline = Pipeline()
    # eval_pipeline.add_component("context_relevance", ContextRelevanceEvaluator(raise_on_failure=False))
    # eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator(raise_on_failure=False))
    eval_pipeline.add_component("sas", SASEvaluator(model=embedding_model))

    eval_pipeline_results = eval_pipeline.run(
        {"sas": {"predicted_answers": predicted_answers, "ground_truth_answers": sample_answers}}
    )
    results = {"sas": eval_pipeline_results["sas"]}
    inputs = {
        "questions": sample_questions,
        "contexts": retrieved_contexts,
        "true_answers": sample_answers,
        "predicted_answers": predicted_answers,
    }

    return results, inputs

def indexing(embedding_model: str, chunk_size: int, base_path: str) -> InMemoryDocumentStore:
    full_path = Path(base_path)
    files_path = full_path / "papers_for_questions"
    document_store = InMemoryDocumentStore()
    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_length=chunk_size, split_overlap=0, split_by="sentence"))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(embedding_model))
    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    pdf_files = [full_path / "papers_for_questions" / f_name for f_name in os.listdir(files_path)]
    pipeline.run({"converter": {"sources": pdf_files}})

    return document_store
