import os
from pathlib import Path
from typing import List

from haystack import Pipeline, Document
from haystack.components.converters import PyPDFToDocument
from haystack.components.evaluators import SASEvaluator
from haystack.components.preprocessors import DocumentCleaner
from haystack.evaluation import EvaluationRunResult
from openai import BadRequestError
from tqdm import tqdm

from techniques.auto_merging_retriever import auto_merging_retrieval, hierarchical_indexing
from techniques.classic import mmr, sentence_window_retrieval
from techniques.document_summary_indexing import indexing_doc_summarisation, doc_summarisation_query_pipeline
from techniques.hybrid_search import hybrid_search
from techniques.llm import rag_with_hyde
from techniques.multi_query import multi_query_pipeline
from techniques.utils import read_question_answers, indexing


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

    n_variations = 3
    top_k = 3

    for q in tqdm(questions):
        try:
            response = multi_query_pip.run(
                data={
                    'multi_query_generator': {'query': q, 'n_variations': n_variations},
                    'multi_query_handler': {'top_k': top_k},
                    'prompt_builder': {'template_variables': {'question': q}},
                    'answer_builder': {'query': q}
                }
            )
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_contexts.append([d.content for d in response["answer_builder"]["answers"][0].documents])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_contexts.append(retrieved_contexts)
    results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
    eval_results_multi_query = EvaluationRunResult(run_name="multi-query", inputs=inputs, results=results)
    eval_results_multi_query.score_report()

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
    eval_results_hybrid.score_report()

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
    eval_results_auto_merging.score_report()

def hyde_eval(answers, doc_store, embedding_model, questions, nr_completions, top_k):
    rag_hyde = rag_with_hyde(doc_store, embedding_model, nr_completions, top_k)
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
    print(eval_results_hyde.score_report())

def sentence_window_eval(answers, doc_store, embedding_model, questions, top_k):
    rag_window_retrieval = sentence_window_retrieval(doc_store, embedding_model, top_k)
    retrieved_contexts, predicted_answers = run_rag(rag_window_retrieval, questions)
    results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
    eval_results_rag_window = EvaluationRunResult(run_name="window-retrieval", inputs=inputs, results=results)
    eval_results_rag_window.score_report()

def maximum_marginal_relevance_reranking(answers, doc_store, embedding_model, questions):
    mmr_pipeline = mmr(doc_store, embedding_model)
    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(questions):
        try:
            response = mmr_pipeline.run(
                data={"text_embedder": {"text": q}, "prompt_builder": {"question": q}, "ranker": {"query": q},
                      "answer_builder": {"query": q}})
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_contexts.append([d.content for d in response["answer_builder"]["answers"][0].documents])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_contexts.append(retrieved_contexts)
    results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
    eval_results_mmr = EvaluationRunResult(run_name="mmr", inputs=inputs, results=results)
    eval_results_mmr.score_report()

def doc_summary_indexing(embedding_model: str, base_path: str, questions, answers):

    print("Indexing summaries...")
    summaries_doc_store, chunk_doc_store = indexing_doc_summarisation(embedding_model, base_path)
    query_pipe = doc_summarisation_query_pipeline(
        chunk_doc_store=chunk_doc_store, summaries_doc_store=summaries_doc_store, embedding_model=embedding_model
    )
    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(questions):
        try:
            response = query_pipe.run(
                data={"text_embedder": {"text": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}})
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_contexts.append([d.content for d in response["answer_builder"]["answers"][0].documents])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_contexts.append(retrieved_contexts)
    results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
    eval_results_doc_summarisation = EvaluationRunResult(run_name="doc-summarisation", inputs=inputs, results=results)
    eval_results_doc_summarisation.score_report()

def main():
    base_path = "data/ARAGOG/"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size = 15
    top_k = 3
    hyde_n_completions = 3

    questions, answers = read_question_answers(base_path)
    print("Indexing documents...")
    doc_store = indexing(embedding_model, chunk_size, base_path)

    # classical Techniques
    sentence_window_eval(answers, doc_store, embedding_model, questions, top_k)
    auto_merging_eval(answers, base_path, embedding_model, questions, top_k)
    maximum_marginal_relevance_reranking(answers, doc_store, embedding_model, questions)
    hybrid_search_eval(answers, doc_store, embedding_model, questions)

    # LLM-based Techniques
    hyde_eval(answers, doc_store, embedding_model, questions, hyde_n_completions, top_k)
    multi_query_eval(answers, doc_store, embedding_model, questions)
    doc_summary_indexing(embedding_model, base_path, questions, answers)

if __name__ == "__main__":
    main()