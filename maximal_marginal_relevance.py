from haystack import Pipeline
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import SentenceTransformersDiversityRanker
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder


def mmr(document_store, embedding_model: str):
    text_embedder = SentenceTransformersTextEmbedder(model=embedding_model)
    embedding_retriever = InMemoryEmbeddingRetriever(document_store)
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