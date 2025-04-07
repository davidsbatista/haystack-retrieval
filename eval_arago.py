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
    top_k = 3
    hyde_n_completions = 3
    multi_query_n_variations = 3
    n_questions = 110

    # Read questions and answers from file + Indexing PDF documents
    questions, answers = read_question_answers(base_path)

    # Select a subset of questions and answers
    questions = questions[:n_questions]
    answers = answers[:n_questions]

    print("Indexing documents...")
    doc_store = indexing(embedding_model, chunk_size, base_path)

    print("Number of documents indexed:")
    print(len(doc_store.storage.values()))

    # classical techniques
    print("\n\nSentence Window")
    df_results = sentence_window_eval(answers, doc_store, embedding_model, questions, top_k)
    df_results.to_csv("results_arago/sentence_window.csv", index=False)

    print("\n\nAuto Merging")
    df_results = auto_merging_eval(answers, base_path, embedding_model, questions, top_k)
    df_results.to_csv("results_arago/auto_merging.csv", index=False)
    
    print("\n\nMaximum Marginal Relevance Reranking")
    df_results = maximum_marginal_relevance_reranking(answers, doc_store, embedding_model, questions, top_k)
    df_results.to_csv("results_arago/mmr.csv", index=False)

    print("\n\nHybrid Search")
    df_results = hybrid_search_eval(answers, doc_store, embedding_model, questions, top_k)
    df_results.to_csv("results_arago/hybrid_search.csv", index=False)

    # LLM-based techniques
    print("\n\nMulti Query")
    df_results = multi_query_eval(answers, doc_store, embedding_model, questions, multi_query_n_variations, top_k)
    df_results.to_csv("results_arago/multi_query.csv", index=False)

    print("\n\nHyDE")
    df_results = hyde_eval(answers, doc_store, embedding_model, questions, hyde_n_completions, top_k)
    df_results.to_csv("results_arago/hyde.csv", index=False)

    print("\n\nDoc Summarisation")
    df_results = doc_summary_indexing(embedding_model, base_path, questions, answers, top_k)
    df_results.to_csv("results_arago/doc_summarisation.csv", index=False)

if __name__ == "__main__":
    main()