from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.memory import ChatMessageHistory
from langchain.vectorstores import FAISS
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.embeddings import HuggingFaceEmbeddings
import os
import json

model_path = "./model/llama-2-13b-chat.ggmlv3.q4_K_M.bin"

stop = [
    '[end of text] ',
]
n_gpu_layers = 32  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
llm = LlamaCpp(
    model_path=model_path,
    input={"temperature": 0.0, "max_length": 8192, "top_p": 1},
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    stop=stop,
    verbose=True,
    n_ctx=4096
)


def main():
    # llm = llm
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.70)
    if(os.path.exists("./faiss_index") == False):
        exit("db is not found. create new db.")
    docsearch = FAISS.load_local("faiss_index", embeddings)
    compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=docsearch.as_retriever())
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=compression_retriever)
    history = ChatMessageHistory()
    while True:
        input_txt = input("質問を入力してください: ")
        if(input_txt == "exit"):
            break
        elif(input_txt == ""):
            continue
        response = qa.run(query=input_txt)
        print(response)
        history.add_user_message(input_txt)
        history.add_ai_message(response)

    with open("./of_history.json", "w", encoding="utf8") as f:
        dicts = messages_to_dict(history.messages)
        formater = json.dumps(dicts, indent=2, ensure_ascii=False)
        print(formater)
        f.write(formater)

if(__name__ == "__main__"):
    main()
