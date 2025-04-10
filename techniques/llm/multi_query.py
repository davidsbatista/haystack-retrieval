from typing import List

from haystack import component, Pipeline, Document
from haystack.components.builders import PromptBuilder, AnswerBuilder, ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.dataclasses.chat_message import ChatMessage
from haystack.utils import Secret


@component
class MultiQueryGenerator:
    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",
            generation_kwargs={"temperature": 0.75, "max_tokens": 400}
        )
        
        template = [
            ChatMessage.from_system(
                "You are an AI language model assistant. Your task is to generate {{n_variations}} different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search. Provide these alternative questions separated by newlines."
            ),
            ChatMessage.from_user(
                "Original question: {{question}}"
            )
        ]
        
        self.prompt_builder = ChatPromptBuilder(
            template=template,
            required_variables=["question", "n_variations"]
        )

    @component.output_types(queries=List[str])
    def run(self, query: str, n_variations: int = 3):
        result = self.prompt_builder.run(question=query, n_variations=n_variations)
        chat_result = self.generator.run(messages=result['prompt'])
        queries = [query] + [q.strip() for q in chat_result['replies'][0].text.split("\n") if q.strip()]
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
        for _, query in enumerate(queries):
            embedding = self.embedder.run(query)
            retrieved_docs = self.embedding_retriever.run(query_embedding=embedding['embedding'], top_k=top_k)
            documents.extend(retrieved_docs['documents'])
        return {"answers": documents}


def multi_query_pipeline(doc_store, embedding_model: str):
    template = [
        ChatMessage.from_system(
            "You have to answer the following question based on the given context information only. "
            "If the context is empty or just a '\\n' answer with None, example: \"None\"."
        ),
        ChatMessage.from_user(
            "Context:\n{% for document in documents %}    {{ document.content }}\n{% endfor %}\n\nQuestion: {{question}}\nAnswer:"
        )
    ]

    pipeline = Pipeline()

    # add components
    pipeline.add_component("multi_query_generator", MultiQueryGenerator())
    pipeline.add_component("multi_query_handler", MultiQueryHandler(document_store=doc_store,embedding_model=embedding_model))
    pipeline.add_component("reranker", DocumentJoiner(join_mode="reciprocal_rank_fusion"))
    pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template))
    pipeline.add_component("llm", OpenAIChatGenerator())
    pipeline.add_component("answer_builder", AnswerBuilder())

    # connect components
    pipeline.connect("multi_query_generator.queries", "multi_query_handler.queries")
    pipeline.connect("multi_query_handler.answers", "reranker.documents")
    pipeline.connect("reranker", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    pipeline.connect("llm.replies", "answer_builder.replies")

    return pipeline
