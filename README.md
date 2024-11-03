# RAG-based Chatbot with Streamlit

[日本語版はこちら](#日本語)

## English

This repository contains a chatbot application that implements Retrieval-Augmented Generation (RAG) using Streamlit. The chatbot can answer questions based on uploaded text documents and engage in general conversation.

### Features

- Upload and process text files for context-aware responses
- Use of LLaMa model for generating responses
- Integration with Hugging Face embeddings and Chroma vector store
- Streamlit-based user interface for easy interaction

### Requirements

To run this application, you need to have the following dependencies installed:

```
streamlit
langchain
langchain_community
langchain_huggingface
chromadb
llama-cpp-python
```

You can install these dependencies using the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

### Usage

1. Place your .gguf format model file in the `models` folder.
2. Update the `model_name` variable in the script with your model file name.
3. Run the Streamlit app:

```
streamlit run llamapp-st-github.py
```

4. Upload a text file using the sidebar to enable RAG functionality.
5. Start chatting with the bot using the input field at the bottom of the page.

### Note

Make sure you have the necessary computational resources to run the LLaMa model efficiently.

---

## 日本語

このリポジトリには、Streamlitを使用して検索拡張生成（RAG）を実装したチャットボットアプリケーションが含まれています。このチャットボットは、アップロードされたテキスト文書に基づいて質問に答え、一般的な会話を行うことができます。

### 特徴

- テキストファイルのアップロードと処理による文脈を考慮した応答
- 応答生成にLLaMaモデルを使用
- Hugging Face embeddings とChroma vector storeの統合
- Streamlitベースのユーザーインターフェースで簡単な操作が可能

### 必要条件

このアプリケーションを実行するには、以下の依存関係をインストールする必要があります：

```
streamlit
langchain
langchain_community
langchain_huggingface
chromadb
llama-cpp-python
```

提供されている`requirements.txt`ファイルを使用して、これらの依存関係をインストールできます：

```
pip install -r requirements.txt
```

### 使用方法

1. .gguf形式のモデルファイルを`models`フォルダに配置します。
2. スクリプト内の`model_name`変数を、使用するモデルファイル名に更新します。
3. Streamlitアプリを実行します：

```
streamlit run llamapp-st-github.py
```

4. サイドバーを使用してテキストファイルをアップロードし、RAG機能を有効にします。
5. ページ下部の入力フィールドを使用して、ボットとチャットを開始します。

### 注意

LLaMaモデルを効率的に実行するために必要な計算リソースがあることを確認してください。
