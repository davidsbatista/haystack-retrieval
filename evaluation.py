import json
import os
from pathlib import Path
from typing import List, Tuple

from haystack import Pipeline, Document
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.evaluators import SASEvaluator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.evaluation import EvaluationRunResult
from openai import BadRequestError
from tqdm import tqdm

from auto_merging_retriever import auto_merging_retrieval, hierarchical_indexing
from hybrid_search import hybrid_search
from hyde import rag_with_hyde
from multi_query import multi_query_pipeline
from sentence_window_retrieval import rag_sentence_window_retrieval


def read_question_answers(base_path: str) -> Tuple[List[str], List[str]]:
    with open(base_path + "eval_questions.json", "r") as f:
        data = json.load(f)
        questions = data["questions"]
        answers = data["ground_truths"]
    return questions, answers

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

def run_rag(rag, questions):
    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(questions):
        try:
            response = rag.run(
                data={"query_embedder": {"text": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}}
            )
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_contexts.append([d.content for d in response["answer_builder"]["answers"][0].documents])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_contexts.append(retrieved_contexts)

    return retrieved_contexts, predicted_answers


def multi_query_eval(answers, doc_store, embedding_model, questions):
    multi_query_pip = multi_query_pipeline(doc_store, embedding_model)
    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(questions):
        try:
            response = multi_query_pip.run(
                data={"multi_query_generator": {"query": q}, "ranker": {"query": q}, "answer_builder": {"query": q}})
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_contexts.append([d.content for d in response["answer_builder"]["answers"][0].documents])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_contexts.append(retrieved_contexts)
    results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
    eval_results_multi_query = EvaluationRunResult(run_name="multi-query", inputs=inputs, results=results)


def hybrid_search_eval(answers, doc_store, embedding_model, questions):
    # NOTE: it needs a cross-encoder to work, takes pairs of sentences as input and produces a similarity score
    hybrid = hybrid_search(doc_store, embedding_model)
    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(questions):
        try:
            response = hybrid.run(
                data={"text_embedder": {"text": q}, "bm25_retriever": {"query": q}, "ranker": {"query": q},
                      "answer_builder": {"query": q}}
            )
            # response['ranker']['documents']
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_contexts.append([d.content for d in response["answer_builder"]["answers"][0].documents])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_contexts.append(retrieved_contexts)
    results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
    eval_results_hybrid = EvaluationRunResult(run_name="hybrid-retrieval", inputs=inputs, results=results)


def auto_merging_eval(answers, base_path, embedding_model, questions, top_k):
    pdf_documents = transform_pdf_to_documents(base_path)
    leaf_doc_store, parent_doc_store = hierarchical_indexing(pdf_documents, embedding_model)
    auto_merging = auto_merging_retrieval(leaf_doc_store, parent_doc_store, embedding_model, top_k)
    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(questions):
        try:
            response = auto_merging.run(
                data={"query_embedder": {"text": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}}
            )
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_contexts.append([d.content for d in response["answer_builder"]["answers"][0].documents])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_contexts.append(retrieved_contexts)
    results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
    eval_results_auto_merging = EvaluationRunResult(run_name="auto-merging-retrieval", inputs=inputs, results=results)


def hyde_eval(answers, doc_store, embedding_model, questions, top_k):
    rag_hyde = rag_with_hyde(doc_store, embedding_model, top_k)
    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(questions):
        try:
            response = rag_hyde.run(
                data={"hyde": {"query": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}}
            )
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_contexts.append([d.content for d in response["answer_builder"]["answers"][0].documents])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_contexts.append(retrieved_contexts)
    results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
    eval_results_hyde = EvaluationRunResult(run_name="hyde", inputs=inputs, results=results)


def sentence_window_eval(answers, doc_store, embedding_model, questions, top_k):
    rag_window_retrieval = rag_sentence_window_retrieval(doc_store, embedding_model, top_k)
    retrieved_contexts, predicted_answers = run_rag(rag_window_retrieval, questions)
    results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
    eval_results_rag_window = EvaluationRunResult(run_name="window-retrieval", inputs=inputs, results=results)
    print(eval_results_rag_window.run_name)
    print(eval_results_rag_window.score_report())
    print()


def main():
    base_path = "data/ARAGOG/"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size = 15
    top_k = 3

    questions, answers = read_question_answers(base_path)
    print("Indexing documents...")
    doc_store = indexing(embedding_model, chunk_size, base_path)

    # SentenceWindow retrieval RAG
    sentence_window_eval(answers, doc_store, embedding_model, questions, top_k)

    # Hypothetical Document Embedder - HyDE
    hyde_eval(answers, doc_store, embedding_model, questions, top_k)

    # Auto-merging retrieval RAG
    auto_merging_eval(answers, base_path, embedding_model, questions, top_k)

    # Hybrid Search and Reciprocal Rank Fusion
    hybrid_search_eval(answers, doc_store, embedding_model, questions)

    # Multi-query
    multi_query_eval(answers, doc_store, embedding_model, questions)

if __name__ == "__main__":
    main()