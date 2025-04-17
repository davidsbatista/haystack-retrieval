import json
import os
from pathlib import Path
from typing import List, Any

from haystack import Pipeline, Document
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.evaluators import SASEvaluator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

from techniques.FalloutEvaluator import FalloutEvaluator
from techniques.PrecisionEvaluator import PrecisionEvaluator
from techniques.RecallEvaluator import RecallEvaluator


def read_question_answers(base_path: str) -> tuple[Any, Any, Any]:
    """
    with open(base_path + "eval_questions.json", "r") as f:
        data = json.load(f)
        questions = data["questions"]
        answers = data["ground_truths"]
        docs = data["filepaths"]
    """

    with open(base_path + "eval_questions_relevant_doc.json", "r") as f_in:
        data = json.load(f_in)
        questions = data["questions"]
        answers = data["ground_truths"]
        docs = data["filepaths"]

    return questions, answers, docs

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

def run_evaluation_aragog(sample_questions, sample_answers, retrieved_contexts, predicted_answers, embedding_model):
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

def run_evaluation_hotpot(questions, answers, docs, retrieved_docs, predicted_answers, embedding_model):
    eval_pipeline = Pipeline()
    eval_pipeline.add_component("sas", SASEvaluator(model=embedding_model))
    eval_pipeline.add_component("recall", RecallEvaluator())
    eval_pipeline.add_component("precision", PrecisionEvaluator())
    eval_pipeline.add_component("fall_out", FalloutEvaluator())

    eval_results = eval_pipeline.run(
        {
            "sas": {"predicted_answers": predicted_answers, "ground_truth_answers": answers},
            "recall": {"ground_truth_documents": docs, "retrieved_documents": retrieved_docs},
            "precision": {"ground_truth_documents": docs, "retrieved_documents": retrieved_docs},
            "fall_out": {"ground_truth_documents": docs, "retrieved_documents": retrieved_docs},
        }
    )

    results = {
        "sas": eval_results["sas"],
        "recall": eval_results["recall"],
        "precision": eval_results["precision"],
        "fall_out": eval_results["fall_out"],
    }

    inputs = {
        "questions": questions,
        "docs": docs,
        "true_answers": answers,
        "predicted_answers": predicted_answers,
        "retrieved_docs": retrieved_docs,
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
