# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリのコードを扱う際のガイダンスを提供します。

## プロジェクト概要

このリポジトリには2つのRustベースのプロジェクトが含まれています:

1. **url-short**: 短縮IDでURLをローカルに保存するCLI URLショートナー
2. **mcp-server**: OpenAI Embeddingsサポート付きのベクトルデータベースを実装したMCP (Model Context Protocol) サーバーと、対話用のPythonクライアント

## ビルドと実行コマンド

### URLショートナー (ルートプロジェクト)

```bash
# リリースバイナリをビルド
cargo build --release

# CLIツールを実行
./target/release/url-short add "https://example.com"
./target/release/url-short get <short-id>
./target/release/url-short list
```

データは `~/.url-short.json` に保存されます。

### MCPサーバー (mcp-server/)

```bash
# MCPサーバーをビルド
cd mcp-server
cargo build --release

# MCPサーバーを実行 (stdin/stdoutでJSON-RPC通信)
./target/release/mcp-server

# Pythonクライアントを実行
chmod +x client.py vectordb_client.py llm_vectordb.py
python3 vectordb_client.py [interactive|demo]
python3 llm_vectordb.py  # .envにANTHROPIC_API_KEYが必要
```

MCPサーバーは埋め込み生成のために環境変数または `.env` ファイルに `OPENAI_API_KEY` が必要です。利用できない場合はハッシュベースのベクトルにフォールバックします。ベクトルデータベースは `vector_db.json` に永続化されます。

## アーキテクチャ

### URLショートナー

シンプルなCLIアプリケーション (`src/main.rs`) で、以下を使用:
- `clap`: CLI引数解析
- `serde_json`: JSON保存
- `nanoid`: 短縮ID生成
- `~/.url-short.json` へのローカルファイル保存

構造: `UrlStore` が `HashMap<String, UrlEntry>` を管理し、add/get/list操作を提供。

### MCPサーバー

Model Context Protocol仕様 (バージョン 2024-11-05) を実装したJSON-RPC 2.0サーバー (`mcp-server/src/main.rs`)。stdin/stdoutで通信します。

**主要コンポーネント:**

- **McpServer**: JSON-RPCリクエストを処理するメインサーバー構造体、HTTPクライアント、OpenAI APIキー、ベクトルデータベースを管理
- **VectorDatabase**: 永続的JSON保存を持つインメモリデータベース、CRUD操作とコサイン類似度検索をサポート
- **OpenAI統合**: OpenAI API経由で埋め込みを生成するために `text-embedding-3-small` (デフォルト) を使用
- **フォールバックシステム**: OpenAI APIが利用不可能な場合のハッシュベースベクトル生成

**利用可能なMCPツール:**

- `echo`, `add`, `get_time`: 基本的なテストツール
- `theme_to_vector`: OpenAI APIを使用してテキストを埋め込みに変換
- `vector_store`, `vector_search`, `vector_get`, `vector_delete`, `vector_list`: ベクトルデータベース操作
- `read_file`, `load_file_to_db`: 分割モード (none/lines/paragraphs) 付きファイル操作

**Pythonクライアント:**

1. **vectordb_client.py**: ベクトルデータベース操作用のインタラクティブ/デモモード付き直接MCPクライアント
2. **llm_vectordb.py**: Claude API (claude-sonnet-4) を使用して自然言語コマンドを解釈してMCPツールを呼び出すLLM駆動アシスタント
3. **client.py**: `theme_to_vector` ツール用のシンプルなデモクライアント

すべてのPythonクライアントはsubprocessを使用してMCPサーバーを起動し、JSON-RPCで通信します。

## 環境設定

MCPサーバーは `mcp-server/` に以下の内容の `.env` ファイルを期待します:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...  # llm_vectordb.pyのみ必要
```

## 主要な実装詳細

- **ベクトル検索**: 埋め込み間のコサイン類似度を使用
- **ID生成**: タイムスタンプベースの16進数ID (`vec_{timestamp_hex}`)
- **永続化**: 両プロジェクトともJSONファイル保存を使用 (各操作で同期書き込み)
- **エラーハンドリング**: MCPツールはツール失敗時にJSON-RPCエラーではなく `isError: true` コンテンツブロックを返す
- **OpenAIモデル**: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` をサポート
