# PostgreSQL + pgvector + Apache AGE Prototype

このディレクトリは、PostgreSQL、pgvector、Apache AGEを使ったベクトル検索とグラフ検索のプロトタイプです。

## セットアップ手順

### 1. PostgreSQLの準備

```bash
# PostgreSQLサービスを起動
sudo service postgresql start

# postgresユーザーでログイン
sudo -u postgres psql

# データベースとユーザーを作成
CREATE DATABASE vectordb_test;
CREATE USER testuser WITH PASSWORD 'testpass';
GRANT ALL PRIVILEGES ON DATABASE vectordb_test TO testuser;
\q
```

### 2. pgvectorのインストール

```bash
# 依存関係
sudo apt install postgresql-server-dev-14 build-essential git

# pgvectorをビルド
cd /tmp
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 3. Apache AGEのインストール（オプション）

```bash
# 依存関係
sudo apt install flex bison

# Apache AGEをビルド
cd /tmp
git clone https://github.com/apache/age.git
cd age
make PG_CONFIG=/usr/bin/pg_config install
```

### 4. 拡張機能の有効化

```bash
psql -U testuser -d vectordb_test

-- pgvector拡張を有効化
CREATE EXTENSION vector;

-- Apache AGE拡張を有効化（インストールした場合）
CREATE EXTENSION age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

\q
```

### 5. 環境変数の設定

```bash
cp .env.example .env
# .envファイルを編集してデータベース接続情報を設定
```

### 6. プロトタイプの実行

```bash
cargo run
```

## テスト内容

1. PostgreSQL接続テスト
2. ベクトルテーブルの作成とCRUD操作
3. ベクトル類似度検索（pgvector）
4. グラフデータの作成と検索（Apache AGE）
