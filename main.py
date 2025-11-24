import os
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings



if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter Google Gemini API Key: ")

def load_and_process_pdf(pdf_url):
    print("Downloading and loading PDF...")
    loader = PyPDFLoader(pdf_url)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    print(f"Processed {len(splits)} document chunks.")
    return splits

def create_vector_db(splits):
    print("Creating Vector Database (Embeddings)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    print("Vector Database ready!")
    return vectorstore

def query_document(vectorstore, query):
    print(f"\nAnalyzing Query: '{query}'")
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    prompt = f"""
    You are an intelligent document assistant. Use the following context to answer the question.
    If the answer is not in the context, say "I don't find that information in the document."
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
    
    splits = load_and_process_pdf(pdf_url)
    db = create_vector_db(splits)
    
    while True:
        user_query = input("\nAsk a question about the PDF (or 'quit'): ")
        if user_query.lower() == 'quit': break
        
        answer = query_document(db, user_query)
        print(f"\nAnswer:\n{answer}")
        print("-" * 50)