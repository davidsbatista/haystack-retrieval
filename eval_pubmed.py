from datasets import load_dataset
from haystack import Document, Pipeline

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.types import DuplicatePolicy

from haystack.evaluation import EvaluationRunResult
from openai import BadRequestError
from tqdm import tqdm

from eval_arago import sentence_window_eval
from techniques.classic import mmr, sentence_window, hybrid_search, hierarchical_indexing, auto_merging
from techniques.llm.doc_summary_indexing import indexing_doc_summarisation, doc_summarisation_query_pipeline
from techniques.llm.hyde import rag_with_hyde
from techniques.llm.multi_query import multi_query_pipeline
from techniques.utils import read_question_answers, transform_pdf_to_documents, run_evaluation

def indexing(embedding_model: str, chunk_size: int, documents: list[Document]) -> InMemoryDocumentStore:

    document_store = InMemoryDocumentStore()
    pipeline = Pipeline()

    pipeline.add_component("splitter", DocumentSplitter(split_length=chunk_size, split_overlap=0, split_by="sentence"))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(embedding_model))
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    pipeline.run({"splitter": {"documents": documents}})

    return document_store


def main():

    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size = 15
    top_k = 5
    hyde_n_completions = 3
    multi_query_n_variations = 3
    n_questions = 107


    dataset = load_dataset("vblagoje/PubMedQA_instruction", split="train")
    subset_dataset = dataset.select(range(n_questions))
    documents = [Document(content=doc["context"]) for doc in subset_dataset]
    questions = [doc["instruction"] for doc in subset_dataset]
    answers = [doc["response"] for doc in subset_dataset]

    print("Indexing documents...")
    doc_store = indexing(embedding_model, chunk_size, documents)

    print("Number of documents indexed:")
    print(len(doc_store.storage.values()))

    # classical techniques
    print("\n\nSentence Window")
    sentence_window_eval(answers, doc_store, embedding_model, questions, top_k)


if __name__ == '__main__':
    main()
