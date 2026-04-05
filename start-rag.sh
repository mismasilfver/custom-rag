#!/bin/bash
# Alias script to start Ollama (if not running) and launch the RAG Streamlit UI
# Usage: source start-rag.sh OR ./start-rag.sh

# Check if Ollama is running
if ! curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo "🤖 Ollama is not running. Starting..."
    ollama serve &
    OLLAMA_PID=$!
    
    # Wait for Ollama to be ready
    echo "⏳ Waiting for Ollama to start..."
    for i in {1..30}; do
        if curl -s http://localhost:11434 > /dev/null 2>&1; then
            echo "✅ Ollama is up!"
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo "❌ Failed to start Ollama. Check your installation."
        exit 1
    fi
else
    echo "✅ Ollama is already running"
fi

# Start the Streamlit UI
echo "🚀 Starting RAG UI..."
./venv/bin/streamlit run app.py
