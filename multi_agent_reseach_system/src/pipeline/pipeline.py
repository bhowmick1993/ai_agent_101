from __future__ import annotations
from src.agents.agents import (create_llm, 
                               build_search_agent, 
                               build_reader_agent, 
                               build_writer_chain, 
                               build_critic_chain)

def research_pipeline(
        topic: str,
        llm_config: dict,
) -> dict:
    """
    Execute the full research pipeline using the specified topic and LLM configuration.

    Args:
        topic (str): The research topic to investigate.
        llm_config (dict): The configuration dictionary for the LLM.

    Returns:
        dict: A dictionary containing the search results, scraped content, research report, and critique feedback.
    """
    state = {}

    # create the LLM instance
    llm = create_llm(**llm_config)

    # Step 1: Search for information
    print("\n" +" = "*50)
    print("Step 1 - Search Agent is working")
    print("="*50 + "\n")
    
    search_agent = build_search_agent(llm)
    search_results = search_agent.invoke(
        {
            "messages" : [("user", f"Find recent, reliable and detailed information on: {topic}")]
        }
    )
    state["search_results"] = search_results["messages"][-1].content
    print(f"\n Search results \n", state['search_results'])

    # Step 2: Scrape content from the first search result
    print("\n" +" = "*50)
    print("Step 2 - Reader Agent is working")
    print("="*50 + "\n")
    
    reader_agent = build_reader_agent(llm)
    scraped_content = reader_agent.invoke(
        {
            "messages" : [(
                "user",
                f"Based on the following search results about '{topic}',"
                f"Pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:800]}"
            )]
        }
    )

    state["scraped_content"] = scraped_content["messages"][-1].content
    print(f"\n Scraped content \n", state['scraped_content'])

    # Step 3: Generate a research report
    print("\n" +" = "*50)
    print("Step 3 - Writer Agent is working")
    print("="*50 + "\n")

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT: \n {state['scraped_content']}"
    )

    writer_chain = build_writer_chain(llm)
    report = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })
    state["report"] = report
    print(f"\n Research report \n", state['report'])

    # Step 4: Critique the report

    print("\n" +" = "*50)
    print("Step 4 - Critic Agent is working")
    print("="*50 + "\n")

    critic_chain = build_critic_chain(llm)
    critique = critic_chain.invoke({
        "report": state["report"]
    })
    state["feedback"] = critique

    print(f"\n Critique report \n", state['feedback'])
    return state