# chatbot-LLM
A chatbot built using LLMs with RAG and context-aware responses.

## Features
- PDF-based question answering
- Uses Pinecone vector database
- Uses HuggingFace embeddings
- Streamlit web interface
- Returns "I don't know" if answer is not found

```bash
git clone https://github.com/Pradeep123-abc/chatbot-LLM.git
```

### Step 1- Create a conda environment
```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```

### Step 2- install the requirement
```bash
pip install -r requirements.txt
```