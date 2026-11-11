# -*- coding: utf-8 -*-
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chromadb
import rag_api

HERE = os.path.dirname(os.path.abspath(__file__))
FAQ = os.path.join(HERE, "知识库-网店客服FAQ.txt")


def split_sentences(t):
    return [p.strip() for p in re.split(r"(?<=[。！？\n])", t) if p.strip()]


def chunk_text(text, size=120, overlap=20):
    chunks, cur = [], ""
    for s in split_sentences(text):
        if len(cur) + len(s) <= size:
            cur += s
        else:
            if cur:
                chunks.append(cur)
                cur = cur[-overlap:]
            cur += s
    if cur:
        chunks.append(cur)
    return chunks


with open(FAQ, encoding="utf-8") as f:
    text = f.read()

chunks = chunk_text(text)
print("① 切块：%d 块" % len(chunks))

print("② 生成向量（硅基流动 bge-m3）...")
vectors = rag_api.embed(chunks)
print("   每块 %d 维" % len(vectors[0]))

client = chromadb.PersistentClient(path=os.path.join(HERE, "chroma_db"))
try:
    client.delete_collection("faq")
except Exception:
    pass
collection = client.create_collection(name="faq", metadata={"hnsw:space": "cosine"})
collection.add(
    ids=["c%d" % i for i in range(len(chunks))],
    documents=chunks,
    embeddings=vectors,
)
print("③ 已存入 Chroma，共 %d 条" % collection.count())

q = input("\n请输入问题：")
qv = rag_api.embed([q])[0]
res = collection.query(query_embeddings=[qv], n_results=3)
print("\n④ 检索结果（距离越小越相似）：")
for i, doc in enumerate(res["documents"][0]):
    print("   #%d 距离=%.4f  %s" % (i + 1, res["distances"][0][i], doc[:50]))