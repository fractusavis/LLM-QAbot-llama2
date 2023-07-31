from langchain.llms import LlamaCpp
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.embeddings import HuggingFaceEmbeddings

def main():
    urls = file_load()
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator="\n")
    texts = text_splitter.split_documents(documents)
    print(len(texts))
    index = FAISS.from_documents(
    documents=texts,
    embedding=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
    )
    index.save_local("faiss_index")
    docsearch = FAISS.load_local("faiss_index", embeddings)

def file_load():
    urls = []
    with open("./page_list.txt", "r", encoding="utf8") as f:
        for line in f:
            urls.append(line.rstrip("\n"))
    return urls

if(__name__ == "__main__"):
    main()
