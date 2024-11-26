from haystack import Pipeline
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder


def hybrid_search(document_store, embedding_model: str):
    text_embedder = SentenceTransformersTextEmbedder(model="BAAI/bge-small-en-v1.5")
    embedding_retriever = InMemoryEmbeddingRetriever(document_store)
    bm25_retriever = InMemoryBM25Retriever(document_store)
    document_joiner = DocumentJoiner()
    ranker = TransformersSimilarityRanker()

    template = """
    You have to answer the following question based on the given context information only.
    If the context is empty or just a '\n' answer with None, example: "None".

    Context:
    {% for document in documents %}
        {{ document }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    hybrid_retrieval = Pipeline()
    hybrid_retrieval.add_component("text_embedder", text_embedder)
    hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
    hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
    hybrid_retrieval.add_component("document_joiner", document_joiner)
    hybrid_retrieval.add_component("ranker", ranker)

    hybrid_retrieval.add_component("prompt_builder", PromptBuilder(template=template))
    hybrid_retrieval.add_component("llm", OpenAIGenerator())
    hybrid_retrieval.add_component("answer_builder", AnswerBuilder())

    hybrid_retrieval.connect("text_embedder", "embedding_retriever")
    hybrid_retrieval.connect("bm25_retriever", "document_joiner")
    hybrid_retrieval.connect("embedding_retriever", "document_joiner")
    hybrid_retrieval.connect("document_joiner", "ranker")
    hybrid_retrieval.connect("ranker.documents", "prompt_builder.documents")
    hybrid_retrieval.connect("prompt_builder", "llm")
    hybrid_retrieval.connect("llm.replies", "answer_builder.replies")
    hybrid_retrieval.connect("llm.meta", "answer_builder.meta")


    return hybrid_retrieval