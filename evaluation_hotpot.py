import json
from typing import Any

from haystack import Pipeline, component, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers.document_writer import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder

def read_hotpot(base_path: str, sample_size: int = 100):
    with open(base_path, "r") as f:
        data = json.load(f)
        import random
        random.seed(42)
        random.shuffle(data)
        return data[:sample_size]

@component
class ProcessHotpot:

    def __init__(self):
        self.documents = []

    @component.output_types(documents=list[Document])
    def run(self, raw_data=list[Any]) -> dict[str, Any]:
        for question_docs in raw_data:
            # a question has multiple documents, some are relevant, some are not
            for doc in question_docs:
                title = doc[0]
                sentences = doc[1]
                for sentence in sentences:
                    self.documents.append(Document(content=title + ' ' + sentence.strip(), meta={"title": title}))

        return {"documents": self.documents}

def indexing(embedding_model: str, raw_data: list[Any]) -> InMemoryDocumentStore:
    document_store = InMemoryDocumentStore()
    pipeline = Pipeline()

    pre_processor = ProcessHotpot()
    pipeline.add_component("pre_processor", pre_processor)
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(embedding_model))
    pipeline.connect("pre_processor", "embedder")
    pipeline.connect("embedder", "writer")
    pipeline.run(data={"raw_data":raw_data})

    return document_store


def main():

    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size = 15
    top_k = 3
    hyde_n_completions = 3
    multi_query_n_variations = 3

    data = read_hotpot("data/hotpot_train_v1.1.json", sample_size=200)

    metadata = [{"_id": entry["_id"], "level": entry["level"], "type": entry["type"]} for entry in data]
    golden_data = [{'question': entry["question"], 'answer': entry["answer"], 'supporting_facts': entry["supporting_facts"]} for entry in data]
    contexts = [entry["context"] for entry in data]

    print("Number of documents: ", len(contexts))
    doc_store = indexing(embedding_model, contexts)

    print(doc_store)



    """
    # classical techniques
    print("Sentence window evaluation...")
    df_results = sentence_window_eval(answers, doc_store, embedding_model, questions, top_k)
    df_results.to_csv("results_arago/sentence_window_eval.csv", index=False)

    print("\n")
    print("Auto-merging evaluation...")
    auto_merging_eval(answers, base_path, embedding_model, questions, top_k)
    df_results.to_csv("results_arago/auto_merging_eval.csv", index=False)

    print("\n")
    print("Maximum Marginal Relevance evaluation...")
    df_results = maximum_marginal_relevance_reranking(answers, doc_store, embedding_model, questions, top_k)
    df_results.to_csv("results_arago/mmr_eval.csv", index=False)

    print("\nHybrid search evaluation...")
    df_results = hybrid_search_eval(answers, doc_store, embedding_model, questions, top_k)
    df_results.to_csv("results_arago/hybrid_search_eval.csv", index=False)

    # LLM-based techniques
    print("\nHyde evaluation...")
    df_results = hyde_eval(answers, docs, doc_store, embedding_model, questions, hyde_n_completions, top_k)
    df_results.to_csv("results_arago/hyde_eval.csv", index=False)

    print("\nMulti-query evaluation...")
    df_results = multi_query_eval(answers, doc_store, embedding_model, questions, multi_query_n_variations, top_k)
    df_results.to_csv("results_arago/multi_query_eval.csv", index=False)

    print("\nDocument summary indexing evaluation...")
    df_results = doc_summary_indexing(embedding_model, base_path, questions, answers, top_k)
    df_results.to_csv("results_arago/doc_summary_indexing_eval.csv", index=False)
    """

if __name__ == "__main__":
    main()