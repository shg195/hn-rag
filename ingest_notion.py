# ingest_notion.py
# 目的：Notionの全記事を取得 → 連結 → 固定長分割 → Ruriで埋め込み → chromadbに保存。
#       コレクションは hn_notion（PDF版の hn_articles とは別）。
#       チャンクIDは「ページID_チャンク番号」で、再実行時に上書き（重複しない）。

import os
from dotenv import load_dotenv
from notion_client import Client
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()
token = os.environ.get("NOTION_TOKEN")
notion = Client(auth=token)

PARENT_PAGE_ID = "3494a536ad1280aa821cd77f4799643e"

# --- 分割設定（step4と同じ） ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 50

# --- Notion取得まわり（step2 + step4から流用） ---

def get_all_children(block_id):
    results = []
    cursor = None
    while True:
        if cursor:
            resp = notion.blocks.children.list(block_id=block_id, start_cursor=cursor)
        else:
            resp = notion.blocks.children.list(block_id=block_id)
        results.extend(resp.get("results", []))
        if resp.get("has_more"):
            cursor = resp.get("next_cursor")
        else:
            break
    return results

def is_month_page(title: str) -> bool:
    return ("年" in title and "月" in title)

def block_to_text(block):
    btype = block.get("type")
    data = block.get(btype, {})
    rich = data.get("rich_text", [])
    text = "".join([r.get("plain_text", "") for r in rich])
    if not text.strip():
        return ""
    if btype in ("heading_1", "heading_2", "heading_3"):
        return "## " + text
    if btype in ("bulleted_list_item", "numbered_list_item"):
        return "- " + text
    return text

def page_to_fulltext(page_id):
    children = get_all_children(page_id)
    parts = []
    for block in children:
        t = block_to_text(block)
        if t:
            parts.append(t)
    return "\n".join(parts)

# 親→月別→記事 を再帰でたどり、記事を収集する
articles = []

def walk(block_id):
    for block in get_all_children(block_id):
        if block.get("type") != "child_page":
            continue
        title = block.get("child_page", {}).get("title", "(無題)")
        page_id = block.get("id")
        edited = block.get("last_edited_time")
        if is_month_page(title):
            walk(page_id)   # 月別ページはさらに潜る
        else:
            articles.append({"title": title, "id": page_id, "last_edited": edited})

# --- 分割（step4と同じ） ---
def chunk_text(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if len(chunk.strip()) >= MIN_CHUNK_SIZE:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks

# ============ ここから本処理 ============

print("=== 1. Notionから記事を収集します ===")
walk(PARENT_PAGE_ID)
print(f"記事 {len(articles)} 件を発見\n")

print("=== 2. 埋め込みモデル（Ruri）を読み込みます ===")
model = SentenceTransformer("cl-nagoya/ruri-v3-30m")
print("読み込み完了\n")

print("=== 3. chromadb を準備します ===")
client = chromadb.PersistentClient(path="./chroma_db")
# 既存の hn_notion があれば作り直す（毎回まっさらに入れ直す方針）
try:
    client.delete_collection("hn_notion")
    print("既存の hn_notion を削除しました")
except Exception:
    pass
collection = client.create_collection(
    name="hn_notion",
    metadata={"hnsw:space": "cosine"}   # コサイン類似度モード
)
print("コレクション hn_notion を作成\n")

print("=== 4. 各記事を分割・埋め込み・保存します ===")
total_chunks = 0
for idx, art in enumerate(articles, 1):
    fulltext = page_to_fulltext(art["id"])
    chunks = chunk_text(fulltext, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        print(f"  [{idx}/{len(articles)}] {art['title']} … 中身が空、スキップ")
        continue

    # Ruriのナレッジ側プレフィックスを付けて埋め込み（正規化ON）
    inputs = ["文章: " + c for c in chunks]
    embeddings = model.encode(inputs, normalize_embeddings=True).tolist()

    ids = [f"{art['id']}_{i}" for i in range(len(chunks))]
    metadatas = [{
        "title": art["title"],
        "page_id": art["id"],
        "last_edited": art["last_edited"],
        "chunk_index": i
    } for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
    total_chunks += len(chunks)
    print(f"  [{idx}/{len(articles)}] {art['title']} … {len(chunks)}チャンク保存")

print(f"\n=== 完了：{len(articles)}記事 / 合計 {total_chunks} チャンクを hn_notion に保存しました ===")