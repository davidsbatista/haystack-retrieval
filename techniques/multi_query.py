from typing import List

from haystack import component, Pipeline, Document
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers import InMemoryEmbeddingRetriever


@component
class MultiQueryGenerator:
    def __init__(self):
        self.generator = OpenAIGenerator(generation_kwargs={"temperature": 0.75, "max_tokens": 400})
        self.prompt_builder = PromptBuilder(
            template="""
            You are an AI language model assistant. Your task is to generate {{n_variations}} different versions of the 
            given user question to retrieve relevant documents from a vector database. 
            By generating multiple perspectives on the user question, your goal is to help the user overcome some of 
            the limitations of distance-based similarity search. Provide these alternative questions separated by 
            newlines. 
            Original question: {{question}}
            """
        )

    @component.output_types(queries=List[str])
    def run(self, query: str, n_variations: int = 3):
        prompt = self.prompt_builder.run(question=query, n_variations=n_variations)
        result = self.generator.run(prompt=prompt['prompt'])
        queries = [query] + [q.strip() for q in result['replies'][0].split("\n") if q.strip()]
        return {"queries": queries}


@component
class MultiQueryHandler:
    def __init__(self, document_store, embedding_model: str):
        self.embedder = SentenceTransformersTextEmbedder(model=embedding_model, progress_bar=False)
        self.embedding_retriever = InMemoryEmbeddingRetriever(document_store)

    @component.output_types(answers=List[Document])
    def run(self, queries: List[str], top_k: int = 3):
        self.embedder.warm_up()
        documents = []
        for idx, query in enumerate(queries):
            embedding = self.embedder.run(query)
            retrieved_docs = self.embedding_retriever.run(query_embedding=embedding['embedding'], top_k=top_k)
            documents.extend(retrieved_docs['documents'])
        return {"answers": documents}


def multi_query_pipeline(doc_store, embedding_model: str):
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

    pipeline = Pipeline()

    # add components
    pipeline.add_component("multi_query_generator", MultiQueryGenerator())
    pipeline.add_component("multi_query_handler", MultiQueryHandler(document_store=doc_store,embedding_model=embedding_model))
    pipeline.add_component("reranker", DocumentJoiner(join_mode="reciprocal_rank_fusion"))
    pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    pipeline.add_component("llm", OpenAIGenerator())
    pipeline.add_component("answer_builder", AnswerBuilder())

    # connect components
    pipeline.connect("multi_query_generator.queries", "multi_query_handler.queries")
    pipeline.connect("multi_query_handler.answers", "reranker.documents")
    pipeline.connect("reranker", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    pipeline.connect("llm.replies", "answer_builder.replies")
    pipeline.connect("llm.meta", "answer_builder.meta")

    return pipeline
