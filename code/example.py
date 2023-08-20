from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

# Get your API keys from openai, you will need to create an account.
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
import os
os.environ["OPENAI_API_KEY"] = "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"
reader = PdfReader(r'E:\IOnewspaper\openaipdf\rapport-annuel-annual-report-2017-2018-3-eng.pdf')

raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text



text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 0,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "what this is talking about"
docs = docsearch.similarity_search(query)

chain.run(input_documents=docs, question=query)