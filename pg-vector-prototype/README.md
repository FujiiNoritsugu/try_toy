# PostgreSQL + pgvector + Apache AGE Prototype

このディレクトリは、PostgreSQL、pgvector、Apache AGEを使ったベクトル検索とグラフ検索のプロトタイプです。

## セットアップ手順

### 1. PostgreSQLの準備

```bash
# PostgreSQLクラスターの確認
pg_lsclusters

# PostgreSQL 14がポート5432で動いていることを確認
# もし12がポート5432で動いている場合は、14に切り替え
sudo pg_dropcluster 12 main --stop

# PostgreSQL 14のポートを5432に変更（必要な場合）
sudo systemctl stop postgresql@14-main
sudo sed -i 's/port = 5433/port = 5432/g' /etc/postgresql/14/main/postgresql.conf
sudo systemctl start postgresql@14-main

# postgresユーザーでログイン（ポート5432）
sudo -u postgres psql -p 5432

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
# postgresユーザーで接続（拡張機能の作成にはsuperuser権限が必要）
sudo -u postgres psql -d vectordb_test

-- pgvector拡張を有効化
CREATE EXTENSION vector;

-- Apache AGE拡張を有効化（インストールした場合）
CREATE EXTENSION age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

\q

# testuserで接続して動作確認
psql -U testuser -d vectordb_test -h localhost
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
