from haystack import Pipeline
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever

from techniques.hyde import HypotheticalDocumentEmbedder


def rag_with_hyde(document_store, embedding_model, nr_completions, top_k):
    template = """
        You have to answer the following question based on the given context information only.
        If the context is empty or just a '\n' answer with None, example: "None".

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """

    hyde = HypotheticalDocumentEmbedder(embedder_model=embedding_model, nr_completions=nr_completions)

    hyde_rag = Pipeline()
    hyde_rag.add_component("hyde", hyde)
    hyde_rag.add_component("retriever", InMemoryEmbeddingRetriever(document_store, top_k=top_k))
    hyde_rag.add_component("prompt_builder", PromptBuilder(template=template))
    hyde_rag.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
    hyde_rag.add_component("answer_builder", AnswerBuilder())

    hyde_rag.connect("hyde", "retriever.query_embedding")
    hyde_rag.connect("retriever", "prompt_builder.documents")
    hyde_rag.connect("prompt_builder", "llm")
    hyde_rag.connect("llm.replies", "answer_builder.replies")
    hyde_rag.connect("llm.meta", "answer_builder.meta")
    hyde_rag.connect("retriever", "answer_builder.documents")

    return hyde_rag