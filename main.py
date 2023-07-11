import translator
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import PyPDFLoader


delimiter = "####"
system_message = """
You will be provided with a block of text.
You will be acting as a translator from English to Korean.
The translation query will be delimited with {delimiter} characters.
"""

def main():
    loader = PyPDFLoader("D:\documents\molec-mechanisms.pdf")
    pages = loader.load()
    text_splitter = TokenTextSplitter(chunk_size = 450, chunk_overlap = 5)
    docs = text_splitter.split_documents(pages)
    # Print the parsed data
    print('parsing data:')
    for parts in docs:
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': parts.page_content}
        ]
        print(translator.translate(messages))

if __name__ == '__main__':
    main()
