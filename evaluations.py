from typing import Union

from haystack.evaluation import EvaluationRunResult
from openai import BadRequestError
from tqdm import tqdm

from haystack import Document
from techniques.classic import mmr, sentence_window, hybrid_search, hierarchical_indexing, auto_merging
from techniques.llm.doc_summary_indexing import indexing_doc_summarisation, doc_summarisation_query_pipeline
from techniques.llm.hyde import rag_with_hyde
from techniques.llm.multi_query import multi_query_pipeline
from techniques.utils import transform_pdf_to_documents, run_evaluation

def sentence_window_eval(answers, doc_store, embedding_model, questions, top_k):
    rag_window_retrieval = sentence_window(doc_store, embedding_model, top_k)

    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(questions):
        try:
            response = rag_window_retrieval.run(
                data={"query_embedder": {"text": q},"prompt_builder": {"question": q},"answer_builder": {"query": q}}
            )
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_contexts.append([d.content for d in response["answer_builder"]["answers"][0].documents])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_contexts.append(retrieved_contexts)

    results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
    eval_results_rag_window = EvaluationRunResult(run_name="window-retrieval", inputs=inputs, results=results)
    print(eval_results_rag_window.aggregated_report())
    df_results = eval_results_rag_window.detailed_report(output_format='df')
    return df_results

def auto_merging_eval(answers, documents: Union[str, list[Document]], embedding_model, questions, top_k):
    if isinstance(documents,str):
        print("Transforming pdf documents...")
        documents = transform_pdf_to_documents(documents)

    leaf_doc_store, parent_doc_store = hierarchical_indexing(documents, embedding_model)
    auto_merging_retrieval = auto_merging(leaf_doc_store, parent_doc_store, embedding_model, top_k)
    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(questions):
        try:
            response = auto_merging_retrieval.run(
                data={"query_embedder": {"text": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}},
                include_outputs_from={"prompt_builder"}
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
    print(eval_results_auto_merging.aggregated_report())
    df = eval_results_auto_merging.detailed_report(output_format='df')
    return df

def maximum_marginal_relevance_reranking(answers, doc_store, embedding_model, questions, top_k):
    mmr_pipeline = mmr(doc_store, embedding_model, top_k)
    predicted_answers = []
    retrieved_contexts = []
    for q in tqdm(questions):
        try:
            response = mmr_pipeline.run(
                data={
                    "text_embedder": {"text": q}, "prompt_builder": {"question": q},
                    "ranker": {"query": q, "top_k": top_k},
                    "answer_builder": {"query": q}
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
    eval_results_mmr = EvaluationRunResult(run_name="mmr", inputs=inputs, results=results)
    print(eval_results_mmr.aggregated_report())
    results_df = eval_results_mmr.detailed_report(output_format='df')
    return results_df


def hybrid_search_eval(answers, doc_store, embedding_model, questions, top_k):
    hybrid = hybrid_search(doc_store, embedding_model, top_k)
    predicted_answers = []
    retrieved_contexts = []
    for q in questions:
        try:
            response = hybrid.run(
                data={"text_embedder": {"text": q}, "bm25_retriever": {"query": q}, "prompt_builder": {"question": q},
                      "answer_builder": {"query": q}}
            )
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_contexts.append([d.content for d in response["answer_builder"]["answers"][0].documents])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_contexts.append(retrieved_contexts)
    results, inputs = run_evaluation(questions, answers, retrieved_contexts, predicted_answers, embedding_model)
    eval_results_hybrid = EvaluationRunResult(run_name="hybrid-retrieval", inputs=inputs, results=results)
    print(eval_results_hybrid.aggregated_report())
    results_df = eval_results_hybrid.detailed_report(output_format='df')
    return results_df

def multi_query_eval(answers, doc_store, embedding_model, questions, n_variations, top_k):
    multi_query_pip = multi_query_pipeline(doc_store, embedding_model)
    predicted_answers = []
    retrieved_contexts = []
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
    print(eval_results_multi_query.aggregated_report())
    results_df = eval_results_multi_query.detailed_report(output_format='df')
    return results_df

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
    print(eval_results_hyde.aggregated_report())
    results_df = eval_results_hyde.detailed_report(output_format='df')
    return results_df

def doc_summary_indexing(embedding_model: str, documents: Union[str, list[Document]], questions, answers, top_k):

    print("Indexing summaries...")
    summaries_doc_store, chunk_doc_store = indexing_doc_summarisation(embedding_model, documents)
    query_pipe = doc_summarisation_query_pipeline(
        chunk_doc_store=chunk_doc_store,
        summaries_doc_store=summaries_doc_store,
        embedding_model=embedding_model,
        top_k=top_k
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
    results_df = eval_results_doc_summarisation.detailed_report(output_format='df')
    print(eval_results_doc_summarisation.aggregated_report())
    return results_df