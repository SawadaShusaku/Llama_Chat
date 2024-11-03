# RAGを試験的に実装したチャットボットアプリ。GitHub用

import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.schema.runnable import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import EphemeralClient

model_name = "modelsフォルダのモデル名を入力"
model_path = "models/" + model_name

# モデルの存在を確認
if not os.path.exists(model_path): 
    raise FileNotFoundError(f"Model file not found: {model_path}")

# エンベッディングモデルの設定
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# テキスト分割器の設定
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

# Chromaクライアントを作成
client = EphemeralClient()

def reset_rag_components():
    """
    RAGの関連コンポーネントをリセットする関数
    - vectorstore
    - rag_chain
    をNoneに設定し、セッション状態をクリアします
    """
    st.session_state.vectorstore = None
    st.session_state.rag_chain = None
    global client #グローバル変数を使用
    # 新しいChromaクライアントを作成（既存のデータをクリア）
    client = EphemeralClient()

    # 既存のコレクションを取得して削除（これがないとリセットされない）
    collections = client.list_collections()
    for collection in collections:
        client.delete_collection(collection.name)
    
    return "💡 RAGがリセットされました"

# ベクトルストアとRAGチェーンの初期化
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

# プロンプトテンプレートの設定
prompt_template = ChatPromptTemplate.from_template("""
    人間: 以下の内容に基づいて質問に日本語で答えてください。
    コンテキスト: {context}
    質問: {question}
    AI: あなたは素晴らしいアシスタントとして回答します。
    """)

CONTEXT_SIZE = 4096

# LLMの設定
if 'llm' not in st.session_state:
    st.session_state.llm = LlamaCpp(
        model_path=model_path,
        n_threads=os.cpu_count(),
        n_batch=512,
        f16_kv=True,
        verbose=False,
        n_ctx=CONTEXT_SIZE,
        temperature=0.7,
        top_p=0.95,
        repeat_penalty=1.1,
        use_mmap=True,
    )

def get_llama_response(prompt):
    return st.session_state.llm.stream(prompt, max_tokens=4096)

def create_vectorstore(text):
    texts = text_splitter.split_text(text)
    metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
    vectorstore = Chroma.from_texts(texts, embed_model, metadatas=metadatas, client=client)
    return vectorstore

# Chromaリトリーバーの設定
def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# RAGチェーンの構築
def build_rag_chain(retriever):
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt_template 
        | st.session_state.llm
    )
    return chain

# ファイルの読み込みと処理を行う関数
def process_uploaded_file(uploaded_file):
    try:
        text_content = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        try:
            text_content = uploaded_file.read().decode("cp932")
        except UnicodeDecodeError:
            st.error("ファイルのエンコーディングを認識できません。UTF-8またはcp932でエンコードされたファイルを使用してください。")
            return None
    return text_content

#############################################################################

#　タイトル
st.subheader("チャットテストα【Chat-α】")

# サイドバーに作者情報を追加
st.markdown("---")
st.sidebar.markdown("---")

# Expanderを使用して追加情報を提供
with st.expander("このアプリについて"):
    st.write("ユーザーの入力した文章（プロンプト）に答える簡易的なチャットツールです。最下段のフォームに文字を入力して、話し掛けてみましょう。")
    st.write(f"言語モデル: {model_path.lstrip("models/")}")
    
# チャット履歴の初期化
if 'messages' not in st.session_state:
    st.session_state.messages = []

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ファイルアップロードの処理とRAG
uploaded_file = st.sidebar.file_uploader("テキストを読み込み（RAG）", type = ["txt"], key = "RAG-input", )
if uploaded_file is not None:
    text_content = process_uploaded_file(uploaded_file)
    if text_content:
        st.session_state.vectorstore = create_vectorstore(text_content)
        retriever = get_retriever(st.session_state.vectorstore)
        st.session_state.rag_chain = build_rag_chain(retriever)
        st.sidebar.success("テキストがアップロードされ、処理されました。")
    else:
        st.sidebar.error("ファイルの処理に失敗しました。")

# RAG処理用のテキストエリア
txt = st.sidebar.text_area(
    "このテキストをRAGで追加", 
    "",
    placeholder="ここにテキストを入力してください...",
    )

# 文字数カウンターの追加
if txt:
    st.sidebar.caption(f"文字数: {len(txt)}")

if txt and txt.strip(): #空白文字列などを無効な入力として扱う
    st.session_state.vectorstore = create_vectorstore(txt)
    retriever = get_retriever(st.session_state.vectorstore)
    st.session_state.rag_chain = build_rag_chain(retriever)
    st.sidebar.success("テキストがアップロードされ、処理されました。")

# 処理状態の表示
if st.session_state.vectorstore is not None:
    st.sidebar.success("RAGアクティブ")
else:
    st.sidebar.warning("RAG未設定")

# 初期化ボタン
if st.sidebar.button("RAGをリセット"):
    status_message = reset_rag_components()
    st.sidebar.info(status_message)
    st.rerun()

# ユーザー入力の処理
if prompt := st.chat_input("メッセージを入力してください"):
    # ユーザーのメッセージをチャット履歴に追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ボットの応答を生成
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        response = get_llama_response(prompt)

        if st.session_state.rag_chain:
            for chunk in st.session_state.rag_chain.stream(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
        else:
            full_response = ""
            for chunk in response:
                if isinstance(chunk, dict) and 'choices' in chunk and len(chunk['choices']) > 0:
                    content = chunk['choices'][0]['text']
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")
                else:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})