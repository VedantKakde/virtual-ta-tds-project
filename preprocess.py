import os
import json
import sqlite3
import re
from bs4 import BeautifulSoup
import html2text
import asyncio
import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Paths
DISCOURSE_FILE = "downloaded_threads/discourse_posts.json"
MARKDOWN_DIR = "markdown_files"
DB_FILE = "knowledge_base.db"

# Chunk size configs
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Create database
def create_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS discourse_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER,
        topic_id INTEGER,
        topic_title TEXT,
        post_number INTEGER,
        author TEXT,
        created_at TEXT,
        likes INTEGER,
        chunk_index INTEGER,
        content TEXT,
        url TEXT,
        embedding BLOB
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS markdown_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_title TEXT,
        original_url TEXT,
        downloaded_at TEXT,
        chunk_index INTEGER,
        content TEXT,
        embedding BLOB
    )""")

    conn.commit()
    conn.close()

# Clean HTML (for discourse)
def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, 'html.parser')
    return soup.get_text()

# Chunk text into overlapping chunks
def create_chunks(text):
    text = text.replace('\n', ' ').strip()
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunks.append(text[i:i + CHUNK_SIZE])
    return chunks

# Process discourse posts
def process_discourse():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    with open(DISCOURSE_FILE, 'r') as f:
        posts = json.load(f)

    for post in tqdm(posts, desc="Processing Discourse"):
        clean_text = post["content"].strip()
        chunks = create_chunks(clean_text)
        for idx, chunk in enumerate(chunks):
            cursor.execute("""
            INSERT INTO discourse_chunks 
            (post_id, topic_id, topic_title, post_number, author, created_at, likes, chunk_index, content, url, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                post["post_id"], post["topic_id"], post["topic_title"],
                post["post_number"], post["author"], post["created_at"],
                post["like_count"], idx, chunk, post["url"], None
            ))
    conn.commit()
    conn.close()

# Process markdown files
def process_markdown():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for filename in tqdm(os.listdir(MARKDOWN_DIR), desc="Processing Markdown"):
        if not filename.endswith(".md"):
            continue

        with open(os.path.join(MARKDOWN_DIR, filename), 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract frontmatter
        match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
        title, url, downloaded_at = "", "", ""
        if match:
            meta = match.group(1)
            content = content[match.end():]
            title = re.search(r'title: "(.*?)"', meta).group(1)
            url = re.search(r'original_url: "(.*?)"', meta).group(1)
            downloaded_at = re.search(r'downloaded_at: "(.*?)"', meta).group(1)

        chunks = create_chunks(content)
        for idx, chunk in enumerate(chunks):
            cursor.execute("""
            INSERT INTO markdown_chunks 
            (doc_title, original_url, downloaded_at, chunk_index, content, embedding)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (title, url, downloaded_at, idx, chunk, None))
    conn.commit()
    conn.close()

# Create embeddings using AIPipe
async def generate_embeddings():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    url = "https://aipipe.org/openai/v1/embeddings"
    model = "text-embedding-3-small"

    # Handle discourse
    cursor.execute("SELECT id, content FROM discourse_chunks WHERE embedding IS NULL")
    discourse = cursor.fetchall()

    # Handle markdown
    cursor.execute("SELECT id, content FROM markdown_chunks WHERE embedding IS NULL")
    markdown = cursor.fetchall()

    session = aiohttp.ClientSession()

    for dataset, table in [(discourse, "discourse_chunks"), (markdown, "markdown_chunks")]:
        for row in tqdm(dataset, desc=f"Embedding {table}"):
            id, content = row
            payload = {"model": model, "input": content}
            async with session.post(url, headers=headers, json=payload) as response:
                data = await response.json()
                embedding = json.dumps(data["data"][0]["embedding"]).encode()
                cursor.execute(f"UPDATE {table} SET embedding=? WHERE id=?", (embedding, id))
                conn.commit()

    await session.close()
    conn.close()

# Main execution
async def main():
    print("✅ Creating database...")
    create_db()

    print("✅ Processing discourse data...")
    process_discourse()

    print("✅ Processing markdown data...")
    process_markdown()

    print("✅ Generating embeddings...")
    await generate_embeddings()

if __name__ == "__main__":
    asyncio.run(main())
