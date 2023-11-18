import os

from agent_executor.upload import IngestRunnable
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import ConfigurableField
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from .config import config

index_schema = {
    "tag": [{"name": "namespace"}],
}
vstore = Chroma(
    collection_name="opengpts",
    persist_directory=config["chroma"]["path"],
    embedding_function=OpenAIEmbeddings()
)


ingest_runnable = IngestRunnable(
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
    vectorstore=vstore,
).configurable_fields(
    assistant_id=ConfigurableField(
        id="assistant_id",
        annotation=str,
        name="Assistant ID",
    ),
)
