import translator
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

llm = ChatOpenAI(model_name ="gpt-3.5-turbo", temperature=0)
print(llm.predict("Hello world!"))
sys.path.append("../..")

persist_direct = 'docs/chroma'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_direct, embedding_function=embedding)

delimiter = "####"
system_message = """
You will be provided with a block of text.
You will be acting as a translator from English to Korean.
The translation query will be delimited with {delimiter} characters.
"""



def main():
    loader = PyPDFLoader("D:\documents\molec-mechanisms.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 0)
    splits = text_splitter.split_documents(pages)
    
    vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_direct
    )
    
    print(vectordb._collection.count())
    question= "What is this document about?"

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    """
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

        qa_chain = RetrievalQA.from_chain_type(llm,
                                            retriever=vectordb.as_retriever(),
                                            return_source_documents=True,
                                            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


        result = qa_chain({"query": question})
        print(result["result"])
    """       

    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    question = "summarize the document in 10 words"
    result = qa({"question": question})
    print(result["answer"])
    question = "what is the best cardiology practice?"
    result = qa({"question": question})
    print(result["answer"])
    question = "why is the best?"
    result = qa({"question": question})
    print(result["answer"])
    # Print the parsed data
    for parts in splits:
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': parts.page_content}
        ]
       # print(translator.translate(messages))

if __name__ == '__main__':
    main()
