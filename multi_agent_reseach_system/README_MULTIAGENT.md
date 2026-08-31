# Disclaimer
This is a learning project exploring agentic AI concepts with LangChain. The current implementation uses a mostly sequential orchestration pipeline; future versions will explore LangGraph-based dynamic routing and multi-agent collaboration.

## Roadmap

- [x] Web-search agent
- [x] Web-content reader agent
- [x] Research writer chain
- [x] Critic chain
- [x] Local Hugging Face LLM support
- [x] Streamlit interface
- [ ] LangGraph orchestration
- [ ] Parallel research agents
- [ ] Writer ↔ Critic revision loop
- [ ] Citation verification
- [ ] Agent evaluation benchmark
- [ ] LangSmith tracing


# Multi-Agent Research System

A local, multi-agent research pipeline that takes any topic, searches the web, scrapes the most relevant page, writes a structured research report, and critiques it — all powered by open-source LLMs running on your own hardware via HuggingFace.

Built for **education and learning** on how to compose multi-agent systems with [LangChain](https://github.com/langchain-ai/langchain).

---

## How It Works

The system runs a four-step sequential pipeline, each step handled by a dedicated agent:

```
Topic
  │
  ▼
┌─────────────────┐
│  Search Agent   │  → Queries the web via Tavily API (top 5 results)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reader Agent   │  → Picks the most relevant URL and scrapes its content
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Writer Chain   │  → Produces a structured research report (Intro / Findings / Conclusion / Sources)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Critic Chain   │  → Scores the report and lists strengths & areas to improve
└─────────────────┘
```

| Step | Agent | Tool | Output |
|------|-------|------|--------|
| 1 | Search Agent | `search_web` (Tavily) | Top-5 search result snippets |
| 2 | Reader Agent | `scrape_url` (trafilatura / readability / BeautifulSoup) | Clean page text (up to 5 000 chars) |
| 3 | Writer Chain | LLM prompt chain | Structured Markdown report |
| 4 | Critic Chain | LLM prompt chain | Score X/10, strengths, improvements, verdict |

---

## Project Structure

```
multi_agent_reseach_system/
├── main.py                     # CLI entry point
├── streamlit_app.py            # Streamlit web UI
├── .env                        # Environment variables (not committed)
├── src/
│   ├── agents/
│   │   └── agents.py           # Agent & chain definitions
│   ├── configs/
│   │   └── llm_configs.yaml    # LLM model IDs and generation settings
│   ├── pipeline/
│   │   └── pipeline.py         # Orchestrates the four-step pipeline
│   └── tools/
│       └── tools.py            # search_web and scrape_url LangChain tools
|-- example_output/             # Conatins some researched outputs as .md files
|-- sample_app_images/          # Some images from the streamlit app
```

---

## Supported LLMs

All models are loaded locally via `HuggingFacePipeline`. They are downloaded automatically on first run.

| CLI flag | Model |
|----------|-------|
| `gemma` | `google/gemma-3-4b-it` |
| `phi` | `microsoft/Phi-4-mini-instruct` |
| `llama` | `meta-llama/Llama-3.1-8B-Instruct` |
| `granite` | `ibm-granite/granite-3.3-8b-instruct` |
| `qwen` | `Qwen/Qwen3-4B` |

> Models are configured in [`src/configs/llm_configs.yaml`](src/configs/llm_configs.yaml). You can add your own by appending a new entry.

---

## Prerequisites

- Python 3.10+
- A GPU is strongly recommended (models use `device_map="auto"`)
- A free [Tavily API key](https://app.tavily.com/) for web search
- A [HuggingFace account](https://huggingface.co/) with access tokens if using gated models (e.g. Llama, Gemma)

---

## Installation

```bash
# 1. Clone the repo and enter the project folder
git clone <your-repo-url>
cd multi_agent_reseach_system

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` / `langchain-core` | Agent & chain framework |
| `langchain-huggingface` | HuggingFace model integration |
| `huggingface_hub` | Model download & caching |
| `tavily-python` | Web search API client |
| `trafilatura` | Primary HTML content extractor |
| `readability-lxml` | Fallback HTML content extractor |
| `bs4` | Final fallback HTML parser |
| `python-dotenv` | `.env` file loading |
| `rich` | Pretty terminal output |
| `streamlit` | Web UI |

---

## Configuration

Create a `.env` file in the project root:

```env
TAVILY_API_KEY=your_tavily_api_key_here
```

To use gated HuggingFace models (Llama, Gemma), authenticate once:

```bash
huggingface-cli login
```

---

## Usage

### CLI

```bash
python main.py --topic "What is the future of artificial intelligence?" --llm gemma
```

**Arguments**

| Argument | Required | Description |
|----------|----------|-------------|
| `--topic` | Yes | The research topic to investigate |
| `--llm` | No | LLM to use: `gemma` (default), `phi`, `llama`, `granite`, `qwen` |

**Example output structure**
```
Step 1 - Search Agent is working
 Search results ...

Step 2 - Reader Agent is working
 Scraped content ...

Step 3 - Writer Agent is working
 Research report ...

Step 4 - Critic Agent is working
 Critique report ...
```

### Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Extending the System

- **Add a new LLM**: Add an entry to `src/configs/llm_configs.yaml` and a mapping key in `main.py`.
- **Add a new tool**: Decorate a function with `@tool` in `src/tools/tools.py` and pass it to the relevant agent in `src/agents/agents.py`.
- **Add a new agent step**: Define a new chain/agent in `agents.py`, invoke it in `pipeline.py`, and store the result in `state`.

---

## Author

**Arka Bhowmick** — Principal Machine Learning Engineer

---

## License

This project is intended for educational purposes. See [LICENSE](../LICENSE) if present, otherwise all rights reserved.
