!pip install langchain
!pip install openai
!pip install PyPDF2
!pip install faiss-cpu
!pip install tiktoken
!pip install openpyxl
!pip install geocoder
import openpyxl
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

# Get your API keys from openai, you will need to create an account.
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
import os
os.environ["OPENAI_API_KEY"] = "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"
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
root_dir = "/content/drive/MyDrive/openai"
from google.colab import drive
drive.mount('/content/drive')
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
import os

os.chdir('/content/drive/MyDrive/openai')
agent = create_csv_agent(OpenAI(temperature=0), 'CSDdf4.csv', verbose=True)
agent.run("the column newspaper contain newspaper names, owner is the owner who own these newspapers, display the newspaper names that the owner is different from lowner ")