from haystack import Pipeline
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import SentenceTransformersDiversityRanker
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder


def mmr(document_store, embedding_model: str, top_k):
    text_embedder = SentenceTransformersTextEmbedder(model=embedding_model)
    embedding_retriever = InMemoryEmbeddingRetriever(document_store, top_k=top_k)
    ranker = SentenceTransformersDiversityRanker(strategy="maximum_margin_relevance")

    template = """
    You have to answer the following question based on the given context information only.
    If the context is empty or just a '\\n' answer with None, example: "None".

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    mmr_pipeline = Pipeline()
    mmr_pipeline.add_component("text_embedder", text_embedder)
    mmr_pipeline.add_component("embedding_retriever", embedding_retriever)
    mmr_pipeline.add_component("ranker", ranker)
    mmr_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    mmr_pipeline.add_component("llm", OpenAIGenerator())
    mmr_pipeline.add_component("answer_builder", AnswerBuilder())

    mmr_pipeline.connect("text_embedder", "embedding_retriever")
    mmr_pipeline.connect("embedding_retriever", "ranker.documents")
    mmr_pipeline.connect("ranker", "prompt_builder.documents")
    mmr_pipeline.connect("prompt_builder", "llm")
    mmr_pipeline.connect("llm.replies", "answer_builder.replies")
    mmr_pipeline.connect("llm.meta", "answer_builder.meta")

    return mmr_pipeline

from haystack import Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever, SentenceWindowRetriever


def sentence_window_retrieval(doc_store, embedding_model, top_k):
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

    basic_rag = Pipeline()
    basic_rag.add_component(
        "query_embedder", SentenceTransformersTextEmbedder(model=embedding_model, progress_bar=False)
    )
    basic_rag.add_component("retriever", InMemoryEmbeddingRetriever(doc_store, top_k=top_k))
    basic_rag.add_component("sentence_window_retriever", SentenceWindowRetriever(document_store=doc_store))
    basic_rag.add_component("prompt_builder", PromptBuilder(template=template))
    basic_rag.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
    basic_rag.add_component("answer_builder", AnswerBuilder())

    basic_rag.connect("query_embedder", "retriever.query_embedding")
    basic_rag.connect("retriever", "sentence_window_retriever")
    basic_rag.connect("sentence_window_retriever.context_windows", "prompt_builder.documents")
    basic_rag.connect("prompt_builder", "llm")
    basic_rag.connect("llm.replies", "answer_builder.replies")
    basic_rag.connect("llm.meta", "answer_builder.meta")

    # to see the retrieved documents in the answer
    basic_rag.connect("retriever", "answer_builder.documents")

    return basic_rag