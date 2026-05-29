# app_notion.py
# 会話履歴対応版（C方式：クエリ書き換え・標準形）
# 追い質問は履歴を踏まえて「独立した質問」に書き換え、その1本で検索も回答もする。
# 既存 ask_notion.py（ターミナル版）は温存。

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import anthropic
import streamlit as st

load_dotenv()

TOP_K = 5
MAX_HISTORY_TURNS = 3            # 書き換えに渡す直近の往復数
GEN_MODEL = "claude-haiku-4-5-20251001"

# --- 重い初期化は1度だけ ---
@st.cache_resource
def load_resources():
    model = SentenceTransformer("cl-nagoya/ruri-v3-30m")
    client_db = chromadb.PersistentClient(path="./chroma_db")
    collection = client_db.get_collection("hn_notion")
    client_ai = anthropic.Anthropic()
    return model, collection, client_ai

model, collection, client_ai = load_resources()

st.set_page_config(page_title="HN Notion RAG", page_icon="📚", layout="wide")
st.title("📚 HN 記事RAG チャット")
st.caption("Notion蓄積のHN詳細記事が知識源。「その/さっき」等の追い質問も文脈を踏まえて検索します。")

# --- 会話履歴の記憶箱（再描画されても消えない）---
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role","content"(, "sources")}]

# --- サイドバー：会話リセット ---
with st.sidebar:
    st.header("操作")
    if st.button("会話をリセット", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"現在の履歴: {len(st.session_state.messages)} 発話")

# --- 過去の会話を毎回再描画 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("参照した記事(近い順)"):
                for s in msg["sources"]:
                    st.write(f"- {s['title']}（類似度 {s['similarity']:.3f}）")

def history_pairs(n_turns):
    """直近n往復ぶんの (role, content) だけを取り出す（sourcesは落とす）。"""
    msgs = st.session_state.messages[-(n_turns * 2):]
    return [{"role": m["role"], "content": m["content"]} for m in msgs]

def rewrite_question(history, question):
    """履歴を踏まえ、単体で意味が通る検索用の質問に書き換える（C方式の心臓）。"""
    if not history:
        return question   # 履歴なし＝書き換え不要
    sys = (
        "あなたは検索クエリを整える役です。これまでの会話と最新の質問が与えられます。"
        "最新の質問が会話の文脈（『それ』『さっき』『その伸びる方』など）に依存している場合、"
        "文脈を補って、それ単体で意味が通る独立した質問に書き換えてください。"
        "依存していなければそのまま返してください。"
        "質問には答えないこと。書き換えた質問の文だけを出力してください。"
    )
    convo = ""
    for m in history:
        who = "ユーザー" if m["role"] == "user" else "アシスタント"
        convo += f"{who}: {m['content']}\n"
    user = f"これまでの会話:\n{convo}\n最新の質問: {question}\n\n書き換えた質問:"
    resp = client_ai.messages.create(
        model=GEN_MODEL,
        max_tokens=200,
        system=sys,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()

# --- 入力 ---
question = st.chat_input("質問をどうぞ")

if question:
    # ユーザー発話を表示＆記憶
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        try:
            with st.spinner("検索＆生成中..."):
                # いま足した質問を除いた「過去の履歴」を用意
                full = history_pairs(MAX_HISTORY_TURNS)
                history = full[:-1]   # 末尾＝今回の質問を除外

                # 1. クエリ書き換え（履歴＋回答を踏まえて独立質問に）
                search_q = rewrite_question(history, question)

                # 2. 書き換え後の質問で検索
                q_emb = model.encode(["クエリ: " + search_q], normalize_embeddings=True).tolist()
                results = collection.query(query_embeddings=q_emb, n_results=TOP_K)
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                dists = results["distances"][0]

                # 3. 文脈組み立て
                context = ""
                for i, (doc, meta) in enumerate(zip(docs, metas), 1):
                    context += f"\n--- 資料{i}(出典: {meta.get('title', '無題')}) ---\n{doc}\n"

                # 4. 回答生成（書き換え後の独立質問で答える＝標準形。履歴は渡さない）
                system_prompt = (
                    "あなたは提供された資料だけを根拠に、日本語で簡潔に答えるアシスタントです。"
                    "資料に書かれていないことは答えず、その場合は「資料には見つかりませんでした」と述べてください。"
                    "推測や一般知識での補完はしないでください。"
                )
                user_prompt = (
                    f"以下の資料を読んで、質問に答えてください。\n\n"
                    f"【資料】{context}\n\n"
                    f"【質問】{search_q}"
                )
                response = client_ai.messages.create(
                    model=GEN_MODEL,
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                answer = response.content[0].text

            # 5. 回答表示
            st.markdown(answer)
            sources = [{"title": m.get("title", "無題"), "similarity": 1 - d} for m, d in zip(metas, dists)]
            with st.expander("参照した記事(近い順)"):
                if search_q != question:
                    st.caption(f"🔁 検索に使った質問: {search_q}")
                for s in sources:
                    st.write(f"- {s['title']}（類似度 {s['similarity']:.3f}）")

            # アシスタント発話を記憶（成功時のみ）
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

        except Exception as e:
            st.error("エラーが出ました。もう一度試してください。")
            st.caption(f"（詳細: {type(e).__name__}）")