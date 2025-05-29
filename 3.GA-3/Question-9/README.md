# TypeScript Book RAG System

A lightweight Retrieval Augmented Generation (RAG) API that answers specific questions about TypeScript by searching through the TypeScript Book documentation.

## Quick Start

### 1. Get the TypeScript Book
```bash
git clone https://github.com/basarat/typescript-book.git
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
python3 app.py
```

That's it! The API will be available at `http://localhost:8000`\
Therefore, final answer should be `http://localhost:8000/search`

## Usage

### Test the API
```bash
# Basic health check
curl "http://localhost:8000/health"

# Ask a question
curl "http://localhost:8000/search?q=What%20does%20the%20author%20affectionately%20call%20the%20%3D%3E%20syntax%3F"
# Returns: {"answer":"fat arrow","sources":["typescript-book/docs/arrow-functions.md"]}
```

### Example Questions
- "What does the author affectionately call the => syntax?" → `"fat arrow"`
- "Which operator converts any value into an explicit boolean?" → `"!!"`
- "What filename do you use to declare globals across your entire TS project?" → `"global.d.ts"`
- "Which keyword pauses and resumes execution in generator functions?" → `"yield"`
- "What property name do discriminated unions use to narrow types?" → `"kind"`

## API Endpoints

- `GET /` - API information
- `GET /search?q=question` - Search for answers
- `GET /health` - Health check

## Features

- **Lightweight**: Uses TF-IDF instead of heavy neural embeddings
- **Fast**: <1 second response time, ~10-15 second startup
- **Potato PC Friendly**: Runs on CPU-only, ~500MB RAM usage
- **Precise Answers**: Returns exact answers, not long excerpts
- **CORS Enabled**: Works from any origin

## System Requirements

- Python 3.7+
- ~500MB RAM
- CPU-only (no GPU required)
- TypeScript Book documentation

## How It Works

1. **Document Processing**: Loads and chunks TypeScript Book markdown files (638 chunks)
2. **TF-IDF Search**: Uses scikit-learn for fast document retrieval
3. **Pattern Matching**: Extracts specific answers using regex patterns
4. **Query Enhancement**: Adds relevant terms to improve search accuracy

Perfect for answering specific TypeScript questions without the overhead of large language models! 
