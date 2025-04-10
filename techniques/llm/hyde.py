from typing import Any, Dict, List

from haystack import Document, component, default_from_dict, default_to_dict
from haystack import Pipeline
from haystack.components.builders import AnswerBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from numpy import array, mean
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage


@component
class HypotheticalDocumentEmbedder:
    def __init__(
        self,
        nr_completions: int = 5,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.nr_completions = nr_completions
        self.embedder_model = embedder_model
        self.generator = OpenAIChatGenerator()
        
        template = [
            ChatMessage.from_system(
                "You are a helpful AI assistant. Given a question, generate a paragraph of text that answers the question."
            ),
            ChatMessage.from_user(
                "Question: {{question}}"
            )
        ]
        
        self.prompt_builder = ChatPromptBuilder(
            template=template,
            required_variables=["question"]
        )

        self.adapter = OutputAdapter(
            template="{{answers | build_doc}}",
            output_type=List[Document],
            custom_filters={"build_doc": lambda data: [Document(content=msg.text) for msg in data]},
            unsafe=True,
        )

        self.embedder = SentenceTransformersDocumentEmbedder(model=embedder_model, progress_bar=False)
        self.embedder.warm_up()

        self.pipeline = Pipeline()
        self.pipeline.add_component(name="prompt_builder", instance=self.prompt_builder)
        self.pipeline.add_component(name="generator", instance=self.generator)
        self.pipeline.add_component(name="adapter", instance=self.adapter)
        self.pipeline.add_component(name="embedder", instance=self.embedder)
        self.pipeline.connect("prompt_builder", "generator")
        self.pipeline.connect("generator.replies", "adapter.answers")
        self.pipeline.connect("adapter.output", "embedder.documents")

    def to_dict(self) -> Dict[str, Any]:
        data = default_to_dict(
            self,
            nr_completions=self.nr_completions,
            embedder_model=self.embedder_model,
        )
        data["pipeline"] = self.pipeline.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HypotheticalDocumentEmbedder":
        hyde_obj = default_from_dict(cls, data)
        hyde_obj.pipeline = Pipeline.from_dict(data["pipeline"])
        return hyde_obj

    @component.output_types(hypothetical_embedding=List[float])
    def run(self, query: str):
        result = self.pipeline.run(data={"prompt_builder": {"question": query}})
        # return a single query vector embedding representing the average of the hypothetical document embeddings
        stacked_embeddings = array([doc.embedding for doc in result["embedder"]["documents"]])
        avg_embeddings = mean(stacked_embeddings, axis=0)
        hyde_vector = avg_embeddings.reshape((1, len(avg_embeddings)))
        return {"hypothetical_embedding": hyde_vector[0].tolist(), "documents": result["embedder"]["documents"]}


def rag_with_hyde(document_store, embedding_model, nr_completions, top_k):
    template = [
        ChatMessage.from_system(
            "You are a helpful AI assistant. Answer the following question based on the given context information only. If the context is empty or just a '\n' answer with None, example: 'None'."
        ),
        ChatMessage.from_user(
            """
            Context:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            Question: {{question}}
            """
        )
    ]

    hyde = HypotheticalDocumentEmbedder(embedder_model=embedding_model, nr_completions=nr_completions)

    hyde_rag = Pipeline()
    hyde_rag.add_component("hyde", hyde)
    hyde_rag.add_component("retriever", InMemoryEmbeddingRetriever(document_store, top_k=top_k))
    hyde_rag.add_component("prompt_builder", ChatPromptBuilder(template=template, required_variables=["question", "documents"]))
    hyde_rag.add_component("llm", OpenAIChatGenerator())
    hyde_rag.add_component("answer_builder", AnswerBuilder())

    hyde_rag.connect("hyde", "retriever.query_embedding")
    hyde_rag.connect("retriever", "prompt_builder.documents")
    hyde_rag.connect("prompt_builder", "llm")
    hyde_rag.connect("llm.replies", "answer_builder.replies")
    hyde_rag.connect("retriever", "answer_builder.documents")

    return hyde_rag