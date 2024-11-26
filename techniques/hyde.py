from typing import Any, Dict, List

from haystack import Document, Pipeline, component, default_from_dict, default_to_dict
from haystack.components.builders import PromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder
from haystack.components.generators.openai import OpenAIGenerator
from haystack.utils import Secret
from numpy import array, mean


@component
class HypotheticalDocumentEmbedder:
    def __init__(
        self,
        instruct_llm: str = "gpt-3.5-turbo",
        instruct_llm_api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        nr_completions: int = 5,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.instruct_llm = instruct_llm
        self.instruct_llm_api_key = instruct_llm_api_key
        self.nr_completions = nr_completions
        self.embedder_model = embedder_model
        self.generator = OpenAIGenerator(
            api_key=self.instruct_llm_api_key,
            model=self.instruct_llm,
            generation_kwargs={"n": self.nr_completions, "temperature": 0.75, "max_tokens": 400},
        )
        self.prompt_builder = PromptBuilder(
            template="""Given a question, generate a paragraph of text that answers the question.
            Question: {{question}}
            Paragraph:
            """
        )

        self.adapter = OutputAdapter(
            template="{{answers | build_doc}}",
            output_type=List[Document],
            custom_filters={"build_doc": lambda data: [Document(content=d) for d in data]},
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
            instruct_llm=self.instruct_llm,
            instruct_llm_api_key=self.instruct_llm_api_key,
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
