import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PERSIST_DIR = "db"

# 三種資料來源 → 三個 collection
SOURCES = {
    "laws": {
        "pattern": "data/laws/*.txt",
        "collection": "trust_laws",
        "chunk_size": 550,
        "chunk_overlap": 120,
    },
    "articles": {
        "pattern": "data/articles/*.txt",
        "collection": "trust_articles",
        "chunk_size": 300,
        "chunk_overlap": 80,
    },
    "products": {
        "pattern": "data/products/*.txt",
        "collection": "trust_products",
        "chunk_size": 300,
        "chunk_overlap": 60,
    },
}

def load_documents(mode: str, pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] mode={mode} 找不到任何檔案：{pattern}")
        return []

    docs = []
    for path in files:
        loader = TextLoader(path, encoding="utf-8")
        loaded = loader.load()

        # 防呆：空檔案
        if not loaded or not loaded[0].page_content.strip():
            print(f"[WARN] 檔案內容為空，略過：{path}")
            continue

        for d in loaded:
            d.metadata["mode"] = mode
            d.metadata["source"] = os.path.basename(path)
        docs.extend(loaded)

    return docs

def ingest_mode(mode: str, cfg: dict, embeddings: OpenAIEmbeddings):
    docs = load_documents(mode, cfg["pattern"])
    if not docs:
        print(f"[SKIP] mode={mode} 無可用文件")
        return 0, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        print(f"[SKIP] mode={mode} 切段後 chunks=0（請檢查文本格式）")
        return len(docs), 0

    # 寫入 Chroma collection（自動持久化，不需要 persist()）
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=cfg["collection"],
        persist_directory=PERSIST_DIR,
    )

    print(f"[OK] mode={mode} docs={len(docs)} chunks={len(chunks)} -> collection={cfg['collection']}")
    return len(docs), len(chunks)

def main():
    os.makedirs(PERSIST_DIR, exist_ok=True)

    embeddings = OpenAIEmbeddings()

    total_docs, total_chunks = 0, 0
    for mode, cfg in SOURCES.items():
        d, c = ingest_mode(mode, cfg, embeddings)
        total_docs += d
        total_chunks += c

    print(f"\n[DONE] total_docs={total_docs}, total_chunks={total_chunks}, db_dir={PERSIST_DIR}/")

if __name__ == "__main__":
    main()