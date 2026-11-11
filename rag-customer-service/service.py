import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_api

HERE = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(HERE, "知识库-网店客服FAQ.txt")


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


with open(FAQ_PATH, encoding="utf-8") as f:
    text = f.read()
chunks = chunk(split_sentences(text))
print("知识库加载：%d 块，正在向量化..." % len(chunks))
chunk_vecs = rag_api.embed(chunks)
print("向量化完成（每块 %d 维）\n" % len(chunk_vecs[0]))


def query_logistics(order_id):
    data = {"12345": "已发货，预计明天送达", "67890": "已签收"}
    return data.get(order_id, "查无此单")


TOOL_DESC = (
    "你可以调用工具：query_logistics(订单号)。"
    "当用户问物流/订单到哪了，只输出 CALL:query_logistics:<订单号>；否则输出 NO。"
)


def retrieve(q, k=3):
    qv = rag_api.embed([q])[0]
    scored = sorted([(rag_api.cosine(qv, v), i) for i, v in enumerate(chunk_vecs)], reverse=True)
    return scored[:k]


def validate_citation(answer, source_count):
    m = re.search(r"第\s*(\d+)\s*段", answer)
    if not m:
        return True, ""
    n = int(m.group(1))
    if n < 1 or n > source_count:
        return False, "来源段落第%d段超出1-%d的实际范围" % (n, source_count)
    return True, ""


history = []
print("网店智能客服已启动（输入 '退出' 结束）\n")

while True:
    q = input("你：")
    if q.strip() == "退出":
        break

    history.append({"role": "user", "content": q})
    history = history[-6:]

    judge = rag_api.chat([{"role": "system", "content": TOOL_DESC}] + history)

    if judge.startswith("CALL:"):
        order = judge.split(":")[-1].strip()
        result = query_logistics(order)
        print("   [调用了工具：query_logistics(%s) → %s]" % (order, result))
        reply = rag_api.chat([{
            "role": "user",
            "content": "物流查询结果：" + result + "。请用人话回答用户的问题：" + q,
        }])
    else:
        hits = retrieve(q)
        source_count = len(hits)
        ctx = "\n".join("[%d] %s" % (i + 1, chunks[idx]) for i, (s, idx) in enumerate(hits))
        prompt = (
            "你是网店客服，必须遵守以下规则：\n"
            "1. 只允许根据下面的资料回答，不准编造；\n"
            "2. 如果资料里没有答案，只回复「抱歉，资料里没有提及，建议联系人工客服」，不要标注来源；\n"
            "3. 如果资料里有答案，正常回答，并在结尾只标注实际引用的段落，格式为「（来源：第X段）」，"
            "X只能是1到%d之间的数字，禁止重复标注。\n\n【资料】\n" % source_count
            + ctx + "\n\n【问题】" + q
        )
        reply = rag_api.chat([{"role": "user", "content": prompt}])

        ok, reason = validate_citation(reply, source_count)
        if not ok:
            print("   [校验拦截：%s]" % reason)
            reply = "抱歉，这个问题我暂时没法确认，建议联系人工客服。"

    history.append({"role": "assistant", "content": reply})
    history = history[-6:]
    print("客服：" + reply + "\n")
