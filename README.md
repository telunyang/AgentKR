[![DOI](https://zenodo.org/badge/1067618260.svg)](https://doi.org/10.5281/zenodo.17240909)

# AgentKR
- This work proposes an integrated knowledge retrieval (KR) system combining re-rankers, local LLMs, AI agents, and knowledge graphs, along with other open-source technologies. It aims to address the hallucination and incomplete retrieval issues of large language models when handling complex queries. Through multi-stage retrieval and data processing pipelines, enhanced by knowledge graphs for interpretability and accuracy, the system improves the efficiency and reliability of knowledge acquisition for users.
- The system supports multiple languages, real-time knowledge updates, and localized deployment, balancing information security and application flexibility. With modular functionality and high stability, it is suitable for education, research, legal, and business domains, and can be widely applied to enterprise knowledge base construction and intelligent Q&A systems, demonstrating strong industrial potential and social impact.

`Note: This project is for preview and educational purposes only.`

---

## Conda Installation
- [Anaconda - Download Now](https://www.anaconda.com/download/success)
```bash
conda create -n ai python=3.11 ipykernel
```

---

## AG2
- [https://docs.ag2.ai/docs/home/home](https://docs.ag2.ai/docs/home/home)

### Packages
```bash
pip install -U ag2[openai,gemini,ollama]
```

---

## PyTorch Installation
```bash
# CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

---

## Package Installation
```bash
pip install -r requirements.txt
```

---

## Google AI Studio
Apply for an API Key [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Using the SDK
Package [https://github.com/googleapis/python-genai](https://github.com/googleapis/python-genai)

### Instructions
After applying for the API, store the API Key in th `.env` file, named as `GOOGLE_API_KEY_01=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`

---

## ollama

### Installation
```bash
# ubuntu
curl -fsSL https://ollama.com/install.sh | sh

# windows
# URL: https://ollama.com/download
```

### Edit ollama.service
```bash
sudo vi /etc/systemd/system/ollama.service
```

### Set ollama models path (optional)
```
[Service]
Environment="OLLAMA_MODELS=/your-path/ollama_models"
```

### Set keepalive
```
[Service]
Environment="OLLAMA_KEEP_ALIVE=-1"
```

### Set Host IP and Port
```
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
```

### Restart ollama
```bash
# Reload system services
sudo systemctl daemon-reload
sudo systemctl restart ollama.service

# Check ollama status
sudo systemctl status ollama.service
```

## Run our code
```bash
python run.py
```