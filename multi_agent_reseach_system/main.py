"""
Date:
Author : Arka Bhowmick, Principal Machine Learning Engineer
Description: This is a multi agent research agent.
             Given a topic, it performs the following,
             1. Searches for relevant information on the topic.
             2. Scrapes content from the most relevant search result.
             3. Generates a research report based on the gathered information.
             4. Critiques the generated report.

             The results of each step are printed to the console.
             The final research report and critique are returned as a dictionary.
The main purpose of this project is education and learning,
how to build multi-agent systems for research tasks using the langchain framework.

Example:
    python main.py --topic "What is the future of artificial intelligence" --llm "gemma"
The models are downloaded from huggingface in my local environment. If you do not have them, they will be automatically downloaded.
"""


from __future__ import annotations
import yaml
import argparse
from rich import print
from src.tools.tools import search_web, scrape_url
from src.pipeline.pipeline import research_pipeline

with open("src/configs/llm_configs.yaml", "r") as f:
    llm_configs = yaml.safe_load(f)

parser = argparse.ArgumentParser(description="Research Pipeline")
parser.add_argument("--topic", type=str, required=True, help="The research topic")
parser.add_argument("--llm", type=str, required=False, help="The LLM to use. qwen, gemma, phi, llama, granite", default="google_gemma")
args = parser.parse_args()

llm_mapping =  {
    "gemma" : "google_gemma",
    "phi": "microsoft_phi",
    "llama": "meta_llama",
    "granite": "ibm_granite",
    "qwen": "qwen"
}

if args.llm not in llm_mapping.keys():
    raise ValueError(f"LLM '{args.llm}' is not supported. Choose from {list(llm_mapping.keys())}")

topic = args.topic
llm_cfg = llm_configs[llm_mapping[args.llm]]
research_pipeline(topic, llm_cfg)