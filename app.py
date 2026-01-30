import os
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import faiss
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def generate_answer(user_input: str):
    pdf_path = "./docs/Harsha Vardhan – Structured Projects Portfolio.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text = "\n".join([page.page_content for page in documents])
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    embeddings=embedding_model.encode(chunks)
    faiss.normalize_L2(embeddings)
    index=faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)    
    llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant")
    def retrieve_and_generate(query, top_k=5):
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = index.search(query_embedding, top_k)

        retrieved_chunks = [chunks[i] for i in indices[0]]
        context = "\n\n".join(retrieved_chunks)

        system_prompt = f"""
        You are answering strictly from a resume project document.
        you are helpful assistant that provides concise and accurate answers based on the document context provided.
        There 6 projects in the document.they are:
        1. HACKNEXT
        2. CAREERLAUNCH
        3. ORGANICDELIGHTS
        4. T-KISAN-MITRA
        5. WEATHER DASHBOARD
        6. UMOVIES
        Question type:
        if the question is related to the document, provide a detailed answer based on the document.
        if unknown project name is asked to explain or asked, respond with "project is not found".
        if the question is NOT related to the document, respond with "It is not related to the portfolio".
        if context has any unrelated information, remove that and answer only the relevant information.
        You are explaining a resume project using ONLY the information from the document context below.
        do not give mulptile answers.
        once recheck your answer to make sure it is aligned with the document context.
        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        memory = InMemorySaver()
        config = {"configurable": {"thread_id": "1"}}
        agent = create_agent(
            model=llm,
            system_prompt=system_prompt,
            checkpointer=memory
        )
        response = agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)
        return response["messages"][-1].content
    return retrieve_and_generate(user_input)