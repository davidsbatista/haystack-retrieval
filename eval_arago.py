from evaluations import (
    auto_merging_eval,
    doc_summary_indexing,
    hybrid_search_eval,
    hyde_eval,
    maximum_marginal_relevance_reranking,
    multi_query_eval,
    sentence_window_eval
)

from techniques.utils import read_question_answers, indexing

def main():
    base_path = "data/ARAGOG/"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size = 15
    top_k = 5
    hyde_n_completions = 3
    multi_query_n_variations = 3

    # Read questions and answers from file + Indexing PDF documents
    questions, answers = read_question_answers(base_path)
    print("Indexing documents...")
    doc_store = indexing(embedding_model, chunk_size, base_path)

    print("Number of documents indexed:")
    print(len(doc_store.storage.values()))

    # classical techniques
    print("\n\nSentence Window")
    sentence_window_eval(answers, doc_store, embedding_model, questions, top_k)
    print("\n\nAuto Merging")
    auto_merging_eval(answers, base_path, embedding_model, questions, top_k)
    print("\n\nMaximum Marginal Relevance Reranking")
    maximum_marginal_relevance_reranking(answers, doc_store, embedding_model, questions, top_k)
    print("\n\nHybrid Search")
    hybrid_search_eval(answers, doc_store, embedding_model, questions, top_k)

    # LLM-based techniques
    print("\n\nHyDE")
    hyde_eval(answers, doc_store, embedding_model, questions, hyde_n_completions, top_k)
    print("\n\nMulti Query")
    multi_query_eval(answers, doc_store, embedding_model, questions, multi_query_n_variations, top_k)
    print("\n\nDoc Summarisation")
    doc_summary_indexing(embedding_model, base_path, questions, answers, top_k)

if __name__ == "__main__":
    main()