#!/bin/bash
# PostgreSQL setup script for prototype

echo "=== PostgreSQL Database Setup ==="
echo ""

# Create database and user
sudo -u postgres psql << EOF
-- Create database
DROP DATABASE IF EXISTS vectordb_test;
CREATE DATABASE vectordb_test;

-- Create user
DROP USER IF EXISTS vectortest;
CREATE USER vectortest WITH PASSWORD 'vectortest123';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE vectordb_test TO vectortest;

-- Connect to the new database and grant schema privileges
\c vectordb_test
GRANT ALL ON SCHEMA public TO vectortest;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO vectortest;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO vectortest;

\q
EOF

echo ""
echo "✓ Database setup completed!"
echo ""
echo "Connection details:"
echo "  Database: vectordb_test"
echo "  User: vectortest"
echo "  Password: vectortest123"
echo ""
echo "Update your .env file with:"
echo "DATABASE_URL=postgresql://vectortest:vectortest123@localhost/vectordb_test"
