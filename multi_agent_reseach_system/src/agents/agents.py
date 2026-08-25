"""
Agents for the multi-agent research system, including search, reader, writer, and critic agents.
"""
from __future__ import annotations
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import search_web, scrape_url
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

load_dotenv()
# ------------------------------------------------------
# Model Setup
# ------------------------------------------------------
def create_llm(
        model_id: str,
        max_tokens: int = 2048,

) -> ChatHuggingFace:
    """
    Create a ChatHuggingFace LLM instance using the specified model ID and maximum token limit.

    Args:
        model_id (str): The HuggingFace model ID to use.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 2048.

    Returns:
        ChatHuggingFace: The initialized ChatHuggingFace LLM instance.
    """
    hf_pipeline = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        model_kwargs={
            "torch_dtype": "auto",
            "device_map": "auto",
        },
        pipeline_kwargs={
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "return_full_text": False,
        },
    )
    llm = ChatHuggingFace(llm=hf_pipeline)
    return llm

# 1st Agent: Search Agent
def build_search_agent(llm: ChatHuggingFace):
    """
    Build a search agent that uses the provided ChatHuggingFace LLM and the search_web tool.

    Args:
        llm (ChatHuggingFace): The ChatHuggingFace LLM instance to use.

    Returns:
        Agent: The initialized search agent.
    """
    return create_agent(
        model=llm,
        tools=[search_web]
        # if you want can create a system prompt to instruct the agent to use the tools,
        #  but for now we will let it figure out when to use them
    )

# 2nd Agent: Scrape Agent
def build_reader_agent(llm: ChatHuggingFace):
    """
    Build a reader agent that uses the provided ChatHuggingFace LLM and the scrape_url tool.

    Args:
        llm (ChatHuggingFace): The ChatHuggingFace LLM instance to use.

    Returns:
        Agent: The initialized reader agent.
    """
    return create_agent(
            model=llm,
            tools=[scrape_url]
            # if you want can create a system prompt to instruct the agent to use the tools,
            #  but for now we will let it figure out when to use them
        )
def build_writer_chain(llm: ChatHuggingFace):
    """
    Build a writer chain that uses the provided ChatHuggingFace LLM to generate research reports.

    Args:
        llm (ChatHuggingFace): The ChatHuggingFace LLM instance to use.

    Returns:
        Chain: The initialized writer chain.
    """
    writer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
        ("human", """Write a detailed research report on the topic below.

    Topic: {topic}

    Research Gathered:
    {research}

    Structure the report as:
    - Introduction
    - Key Findings (minimum 3 well-explained points)
    - Conclusion
    - Sources (list all URLs found in the research)

    Be detailed, factual and professional."""),
    ])

    writer_chain = writer_prompt | llm | StrOutputParser()
    return writer_chain


# 4th Agent: Critic Agent
def build_critic_chain(llm: ChatHuggingFace):
    """
    Build a critic chain that uses the provided ChatHuggingFace LLM to evaluate research reports.

    Args:
        llm (ChatHuggingFace): The ChatHuggingFace LLM instance to use.

    Returns:
        Chain: The initialized critic chain.
    """
    critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

    Report:
    {report}

    Respond in this exact format:

    Score: X/10

    Strengths:
    - ...
    - ...

    Areas to Improve:
    - ...
    - ...

    One line verdict:
    ..."""),
    ])

    critic_chain = critic_prompt | llm | StrOutputParser()
    return critic_chain