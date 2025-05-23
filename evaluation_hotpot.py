import json
from typing import Any

from haystack import Pipeline, component, Document
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers.document_writer import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.builders import ChatPromptBuilder

from haystack.evaluation import EvaluationRunResult
from openai import BadRequestError
from tqdm import tqdm

from techniques.classic import mmr, sentence_window, hybrid_search, hierarchical_indexing, auto_merging
from techniques.llm.doc_summary_indexing import doc_summarisation_query_pipeline, indexing_doc_summarisation_hotpot
from techniques.llm.hyde import rag_with_hyde
from techniques.llm.multi_query import multi_query_pipeline
from techniques.utils import run_evaluation_hotpot


@component
class ProcessHotpot:

    def __init__(self):
        self.documents = []

    @component.output_types(documents=list[Document])
    def run(self, raw_data=list[Any]) -> dict[str, Any]:
        for question_docs in raw_data:
            for doc in question_docs:   # a question has multiple documents, some are relevant, some are not
                title = doc[0]
                sentences = ' '.join([sent for sent in doc[1]])
                self.documents.append(Document(content=title + ' ' + sentences, meta={"title": title}))
        return {"documents": self.documents}

def indexing(embedding_model: str, raw_data: list[Any]) -> InMemoryDocumentStore:
    document_store = InMemoryDocumentStore()
    pipeline = Pipeline()
    pre_processor = ProcessHotpot()
    pipeline.add_component("pre_processor", pre_processor)
    pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=15, split_overlap=0))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(embedding_model))
    pipeline.connect("pre_processor", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    pipeline.run(data={"raw_data":raw_data})

    return document_store

def sentence_window_eval(answers, doc_store, embedding_model, questions, top_k, kwargs=None):

    window_size = kwargs["window_size"] if kwargs and "window_size" in kwargs else 3
    supporting_facts = kwargs["supporting_facts"] if kwargs and "supporting_facts" in kwargs else None

    template = [
        ChatMessage.from_system(
            "You are a helpful AI assistant. Answer the following question in a short and simple manner based only on "
            "the given context information only. If the questions asks you about who/what/when don't add any additional "
            "information, answer the question directly with the answer only even if its too short."
            "If the context is empty or just a '\\n' answer with None, example: 'None'."
        ),
        ChatMessage.from_user(
            """
            Context:
            {% for document in documents %}
                {{ document }}
            {% endfor %}

            Question: {{question}}
            """
        )
    ]

    rag_window_retrieval = sentence_window(doc_store, embedding_model, top_k, window_size=window_size, template=template)
    predicted_answers = []
    retrieved_docs = []
    for q in tqdm(questions):
        try:
            response = rag_window_retrieval.run(
                data={"query_embedder": {"text": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}},
                include_outputs_from={"retriever"}
            )
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_docs.append([d.meta['title'] for d in response["retriever"]["documents"]])

        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_docs.append("error")

    results, inputs = run_evaluation_hotpot(
        questions, answers, supporting_facts, retrieved_docs, predicted_answers, embedding_model
    )
    eval_results_rag_window = EvaluationRunResult(run_name="window-retrieval", inputs=inputs, results=results)
    print(eval_results_rag_window.aggregated_report())
    return eval_results_rag_window.detailed_report(output_format="df")

def auto_merging_eval(answers, documents, embedding_model, questions, top_k, kwargs=None):
    docs = []
    for question_docs in documents:
        for doc in question_docs:  # a question has multiple documents, some are relevant, some are not
            title = doc[0]
            sentences = ' '.join([sent for sent in doc[1]])
            docs.append(Document(content=title + ' ' + sentences, meta={"title": title}))

    supporting_facts = kwargs["supporting_facts"] if kwargs and "supporting_facts" in kwargs else None

    template = [
        ChatMessage.from_system(
            "You are a helpful AI assistant. Answer the following question in a short and simple manner based only on "
            "the given context information only. If the questions asks you about who/what/when don't add any additional "
            "information, answer the question directly with the answer only even if its too short."
            "If the context is empty or just a '\n' answer with None, example: 'None'."
        ),
        ChatMessage.from_user(
            """
            Context:
            {% for document in documents %}
                {{ document }}
            {% endfor %}

            Question: {{question}}
            """
        )
    ]

    leaf_doc_store, parent_doc_store = hierarchical_indexing(docs, embedding_model)
    auto_merging_retrieval = auto_merging(leaf_doc_store, parent_doc_store, embedding_model, top_k, template=template)
    predicted_answers = []
    retrieved_docs = []
    for q in tqdm(questions):
        try:
            response = auto_merging_retrieval.run(
                data={"query_embedder": {"text": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}},
                include_outputs_from={"retriever"}
            )
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_docs.append([d.meta['title'] for d in response["retriever"]["documents"]])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_docs.append("error")

    results, inputs = run_evaluation_hotpot(questions, answers, supporting_facts, retrieved_docs, predicted_answers, embedding_model)
    eval_results_auto_merging = EvaluationRunResult(run_name="auto-merging-retrieval", inputs=inputs, results=results)
    print(eval_results_auto_merging.aggregated_report())
    return eval_results_auto_merging.detailed_report(output_format="df")

def maximum_marginal_relevance_reranking(answers, doc_store, embedding_model, questions, top_k, kwargs=None):
    supporting_facts = kwargs["supporting_facts"] if kwargs and "supporting_facts" in kwargs else None

    template = [
        ChatMessage.from_system(
            "You are a helpful AI assistant. Answer the following question in a short and simple manner based only on "
            "the given context information only. If the questions asks you about who/what/when don't add any additional "
            "information, answer the question directly with the answer only, even if its too short."
            "If the context is empty answer with None, example: 'None'."
        ),
        ChatMessage.from_user(
            """
            Context:
            {% for document in documents %}
                {{ document }}
            {% endfor %}

            Question: {{question}}
            """
        )
    ]

    mmr_pipeline = mmr(doc_store, embedding_model, top_k, template=template)
    predicted_answers = []
    retrieved_docs = []
    context = []
    for q, facts in zip(tqdm(questions), supporting_facts):
        try:
            response = mmr_pipeline.run(
                data={"text_embedder": {"text": q}, "prompt_builder": {"question": q}, "ranker": {"query": q},
                      "answer_builder": {"query": q}},
                include_outputs_from={"ranker"}
            )
            predicted_answer = response["answer_builder"]["answers"][0].data
            predicted_answers.append(predicted_answer)
            retrieved_docs.append([d.meta['title'] for d in response["ranker"]["documents"]])
            context.append([d.content for d in response["ranker"]["documents"]])

            """
            this_docs = [d.meta['title'] for d in response["ranker"]["documents"]]
            if set(facts) in set(this_docs):
                this_context = [d.content for d in response["ranker"]["documents"]]
                prompt = ChatPromptBuilder(template=template, required_variables=['question', 'documents'])
                result = prompt.run(question=q, documents=this_context)
                from haystack.components.generators.chat import OpenAIChatGenerator
                chat = OpenAIChatGenerator()
                answer = chat.run(messages=result["prompt"])

                print("\n\n")
                print("Question: ", q)
                print("Supporting facts: ", facts)
                print("Documents: ", this_docs)


                print(result["prompt"][0].text)
                print(result["prompt"][1].text)
                print("Answer: ", answer)
                print("\n\n")
            """

        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_docs.append("error")

    results, inputs = run_evaluation_hotpot(
        questions, answers, supporting_facts, retrieved_docs, predicted_answers, embedding_model
    )
    eval_results_mmr = EvaluationRunResult(run_name="mmr", inputs=inputs, results=results)
    print(eval_results_mmr.aggregated_report())
    return eval_results_mmr.detailed_report(output_format="df")

def hybrid_search_eval(answers, doc_store, embedding_model, questions, top_k, kwargs=None):
    supporting_facts = kwargs["supporting_facts"] if kwargs and "supporting_facts" in kwargs else None

    template = [
        ChatMessage.from_system(
            "You are a helpful AI assistant. Answer the following question in a short and simple manner based only on "
            "the given context information only. If the questions asks you about who/what/when don't add any additional "
            "information, answer the question directly with the answer only even if its too short."
            "If the context is empty or just a '\n' answer with None, example: 'None'."
        ),
        ChatMessage.from_user(
            """
            Context:
            {% for document in documents %}
                {{ document }}
            {% endfor %}

            Question: {{question}}
            """
        )
    ]

    hybrid = hybrid_search(doc_store, embedding_model, top_k, template=template)
    predicted_answers = []
    retrieved_docs = []

    for q in tqdm(questions):
        try:
            response = hybrid.run(
                data={"text_embedder": {"text": q}, "bm25_retriever": {"query": q}, "prompt_builder": {"question": q},
                      "answer_builder": {"query": q}},
                include_outputs_from={"document_joiner"}
            )
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_docs.append([d.meta['title'] for d in response["document_joiner"]["documents"]])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_docs.append("error")

    results, inputs = run_evaluation_hotpot(questions, answers, supporting_facts, retrieved_docs, predicted_answers, embedding_model)
    eval_results_hybrid = EvaluationRunResult(run_name="hybrid-retrieval", inputs=inputs, results=results)
    print(eval_results_hybrid.aggregated_report())
    return eval_results_hybrid.detailed_report(output_format="df")

def multi_query_eval(answers, doc_store, embedding_model, questions, n_variations, top_k, kwargs=None):
    supporting_facts = kwargs["supporting_facts"] if kwargs and "supporting_facts" in kwargs else None

    template = [
        ChatMessage.from_system(
            "You are a helpful AI assistant. Answer the following question in a short and simple manner based only on "
            "the given context information only. If the questions asks you about who/what/when don't add any additional "
            "information, answer the question directly with the answer only even if its too short."
            "If the context is empty or just a '\n' answer with None, example: 'None'."
        ),
        ChatMessage.from_user(
            """
            Context:
            {% for document in documents %}
                {{ document }}
            {% endfor %}

            Question: {{question}}
            """
        )
    ]

    multi_query_pip = multi_query_pipeline(doc_store, embedding_model, template=template)
    predicted_answers = []
    retrieved_docs = []
    for q in tqdm(questions):
        try:
            response = multi_query_pip.run(
                data={
                    'multi_query_generator': {'query': q, 'n_variations': n_variations},
                    'multi_query_handler': {'top_k': top_k},
                    'prompt_builder': {'template_variables': {'question': q}},
                    'answer_builder': {'query': q}
                },
                include_outputs_from={"reranker"}
            )
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_docs.append([d.meta['title'] for d in response["reranker"]["documents"]])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_docs.append("error")

    results, inputs = run_evaluation_hotpot(questions, answers, supporting_facts, retrieved_docs, predicted_answers, embedding_model)
    eval_results_multi_query = EvaluationRunResult(run_name="multi-query", inputs=inputs, results=results)
    print(eval_results_multi_query.aggregated_report())
    return eval_results_multi_query.detailed_report(output_format="df")

def hyde_eval(answers, doc_store, embedding_model, questions, nr_completions, top_k, kwargs=None):
    supporting_facts = kwargs["supporting_facts"] if kwargs and "supporting_facts" in kwargs else None

    template = [
        ChatMessage.from_system(
            "You are a helpful AI assistant. Answer the following question in a short and simple manner based only on "
            "the given context information only. If the questions asks you about who/what/when don't add any additional "
            "information, answer the question directly with the answer only even if its too short."
            "If the context is empty or just a '\n' answer with None, example: 'None'."
        ),
        ChatMessage.from_user(
            """
            Context:
            {% for document in documents %}
                {{ document }}
            {% endfor %}

            Question: {{question}}
            """
        )
    ]

    rag_hyde = rag_with_hyde(doc_store, embedding_model, nr_completions, top_k, template=template)
    predicted_answers = []
    retrieved_docs = []

    for q in tqdm(questions):

        try:
            response = rag_hyde.run(
                data={"hyde": {"query": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}},
                include_outputs_from={"retriever"}
            )
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_docs.append([d.meta['title'] for d in response["retriever"]["documents"]])

        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_docs.append("error")

    results, inputs = run_evaluation_hotpot(questions, answers, supporting_facts, retrieved_docs, predicted_answers, embedding_model)
    eval_results_hyde = EvaluationRunResult(run_name="hyde", inputs=inputs, results=results)
    print(eval_results_hyde.aggregated_report())
    return eval_results_hyde.detailed_report(output_format="df")

def doc_summary_indexing(embedding_model: str, documents, questions, answers, top_k, kwargs=None):
    supporting_facts = kwargs["supporting_facts"] if kwargs and "supporting_facts" in kwargs else None

    template = [
        ChatMessage.from_system(
            "You are a helpful AI assistant. Answer the following question in a short and simple manner based only on "
            "the given context information only. If the questions asks you about who/what/when don't add any additional "
            "information, answer the question directly with the answer only even if its too short."
            "If the context is empty or just a '\n' answer with None, example: 'None'."
        ),
        ChatMessage.from_user(
            """
            Context:
            {% for document in documents %}
                {{ document }}
            {% endfor %}

            Question: {{question}}
            """
        )
    ]

    docs = []
    for question_docs in documents:
        for doc in question_docs:  # a question has multiple documents, some are relevant, some are not
            title = doc[0]
            sentences = ' '.join([sent for sent in doc[1]])
            docs.append(Document(content=title + ' ' + sentences, meta={"title": title}))

    print("Indexing summaries...")
    print("Number of documents: ", len(docs))
    summaries_doc_store, chunk_doc_store = indexing_doc_summarisation_hotpot(embedding_model, docs)
    query_pipe = doc_summarisation_query_pipeline(
        chunk_doc_store=chunk_doc_store,
        summaries_doc_store=summaries_doc_store,
        embedding_model=embedding_model,
        top_k=top_k,
        template=template
    )
    predicted_answers = []
    retrieved_docs = []

    for q in tqdm(questions):
        try:
            response = query_pipe.run(
                data={"text_embedder": {"text": q}, "prompt_builder": {"question": q}, "answer_builder": {"query": q}},
                include_outputs_from={"chunk_embedding_retriever"})
            predicted_answers.append(response["answer_builder"]["answers"][0].data)
            retrieved_docs.append([d.meta['title'] for d in response["chunk_embedding_retriever"]["chunks"]])
        except BadRequestError as e:
            print(f"Error with question: {q}")
            print(e)
            predicted_answers.append("error")
            retrieved_docs.append("error")

    results, inputs = run_evaluation_hotpot(questions, answers, supporting_facts, retrieved_docs, predicted_answers, embedding_model)
    eval_results_doc_summarisation = EvaluationRunResult(run_name="doc-summarisation", inputs=inputs, results=results)
    print(eval_results_doc_summarisation.aggregated_report())
    return eval_results_doc_summarisation.detailed_report(output_format="df")

def read_hotpot(base_path: str, sample_size: int = 100):
    with open(base_path, "r") as f:
        data = json.load(f)
        return data[:sample_size]

def main():

    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = 3
    hyde_n_completions = 3
    multi_query_n_variations = 3

    print("Loading hotpot dataset")
    data = read_hotpot("data/hotpot_train_v1.1.json", sample_size=100)
    contexts = [entry["context"] for entry in data]

    print("Number of documents: ", len(contexts))
    doc_store = indexing(embedding_model, contexts)

    answers = [entry["answer"] for entry in data]
    questions = [entry["question"] for entry in data]
    supporting_facts = [set([fact[0] for fact in entry["supporting_facts"]]) for entry in data]
    print(f"\nEvaluating techniques over {len(questions)} questions...")

    # classical techniques
    print("Sentence window evaluation...")
    kwargs = {"window_size": 5, 'supporting_facts': supporting_facts}
    df_results = sentence_window_eval(answers, doc_store, embedding_model, questions, top_k, kwargs)
    df_results.to_csv("results_hotpot/sentence_window_eval.csv", index=False)

    print("\nAuto-merging evaluation...")
    kwargs = {'supporting_facts': supporting_facts}
    df_results = auto_merging_eval(answers, contexts, embedding_model, questions, top_k, kwargs)
    df_results.to_csv("results_hotpot/auto_merging_eval.csv", index=False)

    print("\nMaximum Marginal Relevance evaluation...")
    kwargs = {'supporting_facts': supporting_facts}
    df_results = maximum_marginal_relevance_reranking(answers, doc_store, embedding_model, questions, top_k, kwargs)
    df_results.to_csv("results_hotpot/mmr_eval.csv", index=False)

    print("\nHybrid search evaluation...")
    kwargs = {'supporting_facts': supporting_facts}
    df_results = hybrid_search_eval(answers, doc_store, embedding_model, questions, top_k, kwargs)
    df_results.to_csv("results_hotpot/hybrid_search_eval.csv", index=False)

    # LLM-based techniques
    print("\nHyde evaluation...")
    kwargs = {'supporting_facts': supporting_facts}
    df_results = hyde_eval(answers, doc_store, embedding_model, questions, hyde_n_completions, top_k, kwargs)
    df_results.to_csv("results_hotpot/hyde_eval.csv", index=False)

    print("\nMulti-query evaluation...")
    kwargs = {'supporting_facts': supporting_facts}
    df_results = multi_query_eval(
       answers, doc_store, embedding_model, questions, multi_query_n_variations, top_k, kwargs
    )
    df_results.to_csv("results_hotpot/multi_query_eval.csv", index=False)

    print("\nDocument summary indexing evaluation...")
    kwargs = {'supporting_facts': supporting_facts}
    df_results = doc_summary_indexing(embedding_model, contexts, questions, answers, top_k, kwargs)
    df_results.to_csv("results_hotpot/doc_summary_indexing_eval.csv", index=False)


if __name__ == "__main__":
    main()