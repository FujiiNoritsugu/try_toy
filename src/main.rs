use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "url-short")]
#[command(about = "A simple URL shortener CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add a new URL and get a short ID
    Add { url: String },
    /// Get the original URL from a short ID
    Get { id: String },
    /// List all stored URLs
    List,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct UrlEntry {
    id: String,
    url: String,
    created_at: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct UrlStore {
    entries: HashMap<String, UrlEntry>,
}

impl UrlStore {
    fn new() -> Self {
        UrlStore {
            entries: HashMap::new(),
        }
    }

    fn load(path: &PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        if path.exists() {
            let content = fs::read_to_string(path)?;
            let store: UrlStore = serde_json::from_str(&content)?;
            Ok(store)
        } else {
            Ok(UrlStore::new())
        }
    }

    fn save(&self, path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    fn add_url(&mut self, url: String) -> String {
        let id = nanoid::nanoid!(6);
        let now = chrono::Local::now().to_rfc3339();

        let entry = UrlEntry {
            id: id.clone(),
            url: url.clone(),
            created_at: now,
        };

        self.entries.insert(id.clone(), entry);
        id
    }

    fn get_url(&self, id: &str) -> Option<&UrlEntry> {
        self.entries.get(id)
    }

    fn list_all(&self) -> Vec<&UrlEntry> {
        self.entries.values().collect()
    }
}

fn get_store_path() -> PathBuf {
    let mut path = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    path.push(".url-short.json");
    path
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let store_path = get_store_path();
    let mut store = UrlStore::load(&store_path)?;

    match cli.command {
        Commands::Add { url } => {
            let id = store.add_url(url.clone());
            store.save(&store_path)?;
            println!("✓ URL added successfully!");
            println!("  Short ID: {}", id);
            println!("  Original: {}", url);
        }
        Commands::Get { id } => {
            if let Some(entry) = store.get_url(&id) {
                println!("{}", entry.url);
            } else {
                eprintln!("Error: ID '{}' not found", id);
                std::process::exit(1);
            }
        }
        Commands::List => {
            let entries = store.list_all();
            if entries.is_empty() {
                println!("No URLs stored yet.");
            } else {
                println!("Stored URLs:");
                println!("{:<8} {:<50} {}", "ID", "URL", "Created");
                println!("{}", "-".repeat(80));
                for entry in entries {
                    let url_display = if entry.url.len() > 47 {
                        format!("{}...", &entry.url[..47])
                    } else {
                        entry.url.clone()
                    };
                    println!("{:<8} {:<50} {}", entry.id, url_display, entry.created_at);
                }
            }
        }
    }

    Ok(())
}
