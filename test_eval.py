# -*- coding: utf-8 -*-
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_api

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
FAQ = os.path.join(HERE, "知识库-网店客服FAQ.txt")


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=[。！？\n])", text) if p.strip()]


def chunk(sentences, size=120, overlap=20):
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) <= size:
            cur += s
        else:
            if cur:
                chunks.append(cur)
                cur = cur[-overlap:] if overlap else ""
            while len(s) > size:
                chunks.append(s[:size])
                s = s[size:]
            cur += s
    if cur:
        chunks.append(cur)
    return chunks


with open(FAQ, encoding="utf-8") as f:
    text = f.read()
chunks = chunk(split_sentences(text))
print("知识库: %d 个文件, %d 条内容, 切成 %d 块" % (1, len(split_sentences(text)), len(chunks)))
chunk_vecs = rag_api.embed(chunks)
print("向量维度: %d" % len(chunk_vecs[0]))


def retrieve(q, k=3):
    qv = rag_api.embed([q])[0]
    scored = sorted([(rag_api.cosine(qv, v), i) for i, v in enumerate(chunk_vecs)], reverse=True)
    return scored[:k]


def validate(answer, n):
    m = re.search(r"第\s*(\d+)\s*段", answer)
    if not m:
        return True
    x = int(m.group(1))
    return 1 <= x <= n


def ask(q):
    t0 = time.time()
    hits = retrieve(q)
    t_retrieve = time.time() - t0
    n = len(hits)
    ctx = "\n".join("[%d] %s" % (i + 1, chunks[idx]) for i, (s, idx) in enumerate(hits))
    prompt = (
        "你是网店客服，必须遵守以下规则：\n"
        "1. 只允许根据下面的资料回答，不准编造；\n"
        "2. 如果资料里没有答案，只回复「抱歉，资料里没有提及，建议联系人工客服」，不要标注来源；\n"
        "3. 如果资料里有答案，正常回答，并在结尾只标注实际引用的段落，格式为「（来源：第X段）」，"
        "X只能是1到%d之间的数字，禁止重复标注。\n\n【资料】\n" % n
        + ctx + "\n\n【问题】" + q
    )
    t1 = time.time()
    reply = rag_api.chat([{"role": "user", "content": prompt}])
    t_answer = time.time() - t1
    if not validate(reply, n):
        reply = "抱歉，这个问题我暂时没法确认，建议联系人工客服。"
    return reply, t_retrieve, t_answer


TESTS = [
    ("退款要多久？怎么申请？", ["3-5", "申请售后"], "有答案"),
    ("运费怎么算？", ["99", "8"], "有答案"),
    ("发票怎么开？", ["电子发票", "邮箱"], "有答案"),
    ("会员有什么优惠？", ["9折"], "有答案"),
    ("退费怎么弄？", ["退款", "售后"], "换说法"),
    ("买东西的钱能退回来吗？", ["退款"], "换说法"),
    ("支持货到付款吗？", ["没有提及", "不知道", "人工"], "知识库外"),
    ("能分期付款吗？", ["没有提及", "不知道", "人工"], "知识库外"),
    ("你们公司在哪个城市？", ["没有提及", "不知道", "人工"], "知识库外"),
]

print("\n开始测试（%d 题）..." % len(TESTS))
results = []
for q, kws, cat in TESTS:
    ans, tr, ta = ask(q)
    ok = any(k in ans for k in kws)
    results.append((q, cat, ans, ok, tr, ta))
    print("-" * 55)
    print("[%s] %s" % (cat, q))
    print("  答: %s" % ans[:100].replace("\n", " "))
    print("  判定: %s | 检索 %.2fs | 回答 %.2fs" % ("✅" if ok else "❌", tr, ta))

total = len(results)
passed = sum(1 for r in results if r[3])
avg_tr = sum(r[4] for r in results) / total
avg_ta = sum(r[5] for r in results) / total
out = [r for r in results if r[1] == "知识库外"]
out_bad = [r for r in out if not r[3]]

print("\n" + "=" * 55)
print("总题数: %d | 通过: %d | 答对率: %.1f%%" % (total, passed, 100.0 * passed / total))
print("平均检索耗时: %.2fs | 平均回答耗时: %.2fs" % (avg_tr, avg_ta))
print("知识库外 %d 题 | 未正确拒绝 %d 题 | 幻觉率: %.1f%%" % (
    len(out), len(out_bad), 100.0 * len(out_bad) / len(out) if out else 0))
