use sqlx::postgres::PgPoolOptions;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load .env file
    dotenvy::dotenv().ok();

    println!("=== PostgreSQL + pgvector + Apache AGE Prototype ===\n");

    // Get database URL from environment
    let database_url = env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://postgres:password@localhost/vectordb_test".to_string());

    println!("1. Connecting to PostgreSQL...");
    println!("   URL: {}", database_url.replace(|c: char| c == ':' || c.is_ascii_digit(), "*"));

    // Create connection pool
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await;

    match pool {
        Ok(pool) => {
            println!("   ✓ Connected successfully!\n");

            // Test basic query
            println!("2. Testing basic query...");
            let row: (i32,) = sqlx::query_as("SELECT 1 + 1")
                .fetch_one(&pool)
                .await?;
            println!("   ✓ Query result: 1 + 1 = {}\n", row.0);

            // Check PostgreSQL version
            println!("3. Checking PostgreSQL version...");
            let version: (String,) = sqlx::query_as("SELECT version()")
                .fetch_one(&pool)
                .await?;
            println!("   {}\n", version.0);

            // Check if pgvector extension is available
            println!("4. Checking pgvector extension...");
            let pgvector_check: Result<(bool,), sqlx::Error> = sqlx::query_as(
                "SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'vector')"
            )
            .fetch_one(&pool)
            .await;

            match pgvector_check {
                Ok((exists,)) => {
                    if exists {
                        println!("   ✓ pgvector extension is available");

                        // Check if it's installed
                        let installed: (bool,) = sqlx::query_as(
                            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                        )
                        .fetch_one(&pool)
                        .await?;

                        if installed.0 {
                            println!("   ✓ pgvector extension is installed\n");
                        } else {
                            println!("   ⚠ pgvector is available but not installed");
                            println!("     Run: CREATE EXTENSION vector;\n");
                        }
                    } else {
                        println!("   ✗ pgvector extension is not available");
                        println!("     Please install pgvector first\n");
                    }
                }
                Err(e) => {
                    println!("   ✗ Error checking pgvector: {}\n", e);
                }
            }

            // Check if Apache AGE extension is available
            println!("5. Checking Apache AGE extension...");
            let age_check: Result<(bool,), sqlx::Error> = sqlx::query_as(
                "SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'age')"
            )
            .fetch_one(&pool)
            .await;

            match age_check {
                Ok((exists,)) => {
                    if exists {
                        println!("   ✓ Apache AGE extension is available");

                        // Check if it's installed
                        let installed: (bool,) = sqlx::query_as(
                            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'age')"
                        )
                        .fetch_one(&pool)
                        .await?;

                        if installed.0 {
                            println!("   ✓ Apache AGE extension is installed\n");
                        } else {
                            println!("   ⚠ Apache AGE is available but not installed");
                            println!("     Run: CREATE EXTENSION age;\n");
                        }
                    } else {
                        println!("   ✗ Apache AGE extension is not available");
                        println!("     This is optional - you can still use vector search\n");
                    }
                }
                Err(e) => {
                    println!("   ✗ Error checking Apache AGE: {}\n", e);
                }
            }

            println!("=== Connection test completed ===");
        }
        Err(e) => {
            println!("   ✗ Connection failed!\n");
            println!("Error: {}", e);
            println!("\nTroubleshooting:");
            println!("1. Make sure PostgreSQL is running: sudo service postgresql start");
            println!("2. Check your DATABASE_URL in .env file");
            println!("3. Verify database and user exist");
            println!("4. Check credentials");
        }
    }

    Ok(())
}
