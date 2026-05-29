# ask_notion.py
# 目的：hn_notion コレクションに対して質問応答する。
#       質問を埋め込み → top-k取得 → Claude Haikuで回答 → 出典をメタデータから表示。

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import anthropic

load_dotenv()

TOP_K = 5   # 取得するチャンク数

# --- 準備 ---
print("埋め込みモデルを読み込み中...")
model = SentenceTransformer("cl-nagoya/ruri-v3-30m")

client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_collection("hn_notion")

client_ai = anthropic.Anthropic()  # APIキーは .env の ANTHROPIC_API_KEY を自動で読む

print("準備完了。質問をどうぞ（終了は Ctrl+C）\n")

while True:
    question = input("質問> ").strip()
    if not question:
        continue

    # 1. 質問を埋め込む（Ruriのクエリ側プレフィックス + 正規化）
    q_emb = model.encode(["クエリ: " + question], normalize_embeddings=True).tolist()

    # 2. hn_notion から近いチャンクを top-k 取得
    results = collection.query(
        query_embeddings=q_emb,
        n_results=TOP_K
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # 3. 取得したチャンクを文脈としてまとめる
    context = ""
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        context += f"\n--- 資料{i}（出典: {meta['title']}）---\n{doc}\n"

    # 4. Claude Haiku に回答させる
    system_prompt = (
        "あなたは提供された資料だけを根拠に、日本語で簡潔に答えるアシスタントです。"
        "資料に書かれていないことは答えず、その場合は「資料には見つかりませんでした」と述べてください。"
        "推測や一般知識での補完はしないでください。"
    )
    user_prompt = f"以下の資料を読んで、質問に答えてください。\n\n【資料】{context}\n\n【質問】{question}"

    response = client_ai.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    answer = response.content[0].text

    # 5. 回答と出典を表示
    print("\n【回答】")
    print(answer)
    print("\n【参照した記事（近い順）】")
    for meta, dist in zip(metas, dists):
        # コサイン距離 → 類似度（1 - 距離）で見やすく
        similarity = 1 - dist
        print(f"  - {meta['title']}（類似度 {similarity:.3f}）")
    print("\n" + "=" * 60 + "\n")