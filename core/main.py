"""
Ingest every PDF in ./documents into cold storage (MySQL).

Needs LLAMA_CLOUD_API_KEY in the environment or .env.
Run from the project root:  python -m core.main
"""

import os
import re

from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.cold_storage import ColdStorage
from core.config import MYSQL_CONFIG

DOCUMENTS_DIR = "./documents"


def clean_markdown(text):
    text = re.sub(r'#{1,6}\s*', '', text)        # remove headers
    text = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', text)  # remove bold/italic
    text = re.sub(r'\|.*?\|', '', text)           # remove table cells
    text = re.sub(r'[-]{3,}', '', text)           # remove horizontal rules
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # remove links keep text
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text)  # remove code blocks
    text = re.sub(r'\s+', ' ', text)              # normalize whitespace
    return text.strip()


def main():
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise SystemExit("LLAMA_CLOUD_API_KEY is not set. Add it to your .env file.")

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=True
    )

    all_docs = []
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DOCUMENTS_DIR, filename)
            print(f"Parsing {filename}...")
            docs = parser.load_data(filepath)
            all_docs.extend(docs)

    print(f"Total docs loaded: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80
    )

    cold_storage = ColdStorage(**MYSQL_CONFIG)

    chunks = splitter.create_documents([doc.text for doc in all_docs])

    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        cleaned = clean_markdown(chunk.page_content)
        if cleaned:
            cold_storage.add_document(cleaned)

    print(f"Done. {len(chunks)} chunks inserted.")
    cold_storage.close()


if __name__ == "__main__":
    main()
