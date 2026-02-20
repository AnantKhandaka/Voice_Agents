import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from .logger_config import setup_logger

load_dotenv()
logger = setup_logger()


class RAGEngine:
    def __init__(self, pdf_path="sample.pdf"):
        self.pdf_path = pdf_path
        self.vectorstore = None
        self.llm = None
        self.prompt = None
        self._initialize()

    def _initialize(self):
        logger.info("Initializing RAG Engine with Ollama")

        # 1️⃣ Load PDF
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        # 2️⃣ Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100
        )
        docs = splitter.split_documents(documents)

        # 3️⃣ Embeddings
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 4️⃣ Vector Store
        if os.path.exists("./chroma_db"):
            logger.info("Loading existing Chroma DB")
            self.vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embedding
            )
        else:
            logger.info("Creating new Chroma DB")
            self.vectorstore = Chroma.from_documents(
                docs,
                embedding,
                persist_directory="./chroma_db"
            )

        # 5️⃣ Ollama LLM
        model_name = os.getenv("OLLAMA_MODEL", "llama3")

        self.llm = ChatOllama(
            model=model_name,
            temperature=0.3
        )

        # 6️⃣ Prompt
        self.prompt = PromptTemplate(
            template="""
            You are an EHR assistant.
            Answer strictly from context.
            If answer not found, say "I don't know."

            Context:
            {context}

            Question:
            {question}

            Answer:
            """,
            input_variables=["context", "question"]
        )

        logger.info("RAG Engine initialized successfully")

    def query(self, question: str):
        logger.info(f"User question: {question}")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)

        context = "\n\n".join([doc.page_content for doc in docs])

        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question
        })

        answer = response.content if hasattr(response, "content") else str(response)

        logger.info(f"Generated answer: {answer}")

        return answer