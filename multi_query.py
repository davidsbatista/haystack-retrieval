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
            template="""You are an AI language model assistant. Your task is 
            to generate 3 different versions of the given user 
            question to retrieve relevant documents from a vector database. 
            By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations 
            of distance-based similarity search. Provide these alternative questions separated by newlines. 
            Original question: {{question}}"""
        )

    @component.output_types(multi_queries=List[str])
    def run(self, query: str):
        prompt = self.prompt_builder.run(question=query)
        result = self.generator.run(prompt=prompt['prompt'])
        queries = [q.strip() for q in result['replies'][0].split("\n") if q.strip()]
        return {"multi_queries": queries}


@component
class MultiQueryHandler:
    def __init__(self, document_store, embedding_model: str):
        self.embedder = SentenceTransformersTextEmbedder(model=embedding_model)
        self.embedding_retriever = InMemoryEmbeddingRetriever(document_store)

    @component.output_types(multi_queries=List[Document])
    def run(self, queries: List[str]):
        self.embedder.warm_up()
        documents = []
        for query in queries:
            print(f"Running query: {query}")
            embedding = self.embedder.run(query)
            retrieved_docs = self.embedding_retriever.run(query_embedding=embedding['embedding'])
            print(f"Retrieved {len(retrieved_docs['documents'])} documents.")
            documents.extend(retrieved_docs['documents'])
        return {"multi_queries": documents}



def multi_query_pipeline(document_store, embedding_model: str):
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
    hybrid_retrieval.add_component("multi_query_generator", MultiQueryGenerator())
    hybrid_retrieval.add_component("multi_query_handler", MultiQueryHandler(
        document_store=document_store,
        embedding_model=embedding_model)
    )
    hybrid_retrieval.add_component("document_joiner", document_joiner)
    hybrid_retrieval.add_component("ranker", ranker)
    hybrid_retrieval.add_component("prompt_builder", PromptBuilder(template=template))
    hybrid_retrieval.add_component("llm", OpenAIGenerator())
    hybrid_retrieval.add_component("answer_builder", AnswerBuilder())
    hybrid_retrieval.connect("multi_query_generator", "multi_query_handler")
    hybrid_retrieval.connect("multi_query_handler", "document_joiner")
    hybrid_retrieval.connect("document_joiner", "ranker")
    hybrid_retrieval.connect("ranker.documents", "prompt_builder.documents")
    hybrid_retrieval.connect("prompt_builder", "llm")
    hybrid_retrieval.connect("llm.replies", "answer_builder.replies")
    hybrid_retrieval.connect("llm.meta", "answer_builder.meta")


    return hybrid_retrieval