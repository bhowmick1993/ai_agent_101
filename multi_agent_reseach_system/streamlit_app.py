from __future__ import annotations

import streamlit as st
import yaml
from pathlib import Path

from src.agents.agents import (
    create_llm,
    build_search_agent,
    build_reader_agent,
    build_writer_chain,
    build_critic_chain,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Main title */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Agent cards */
    .agent-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .agent-title {
        font-weight: 650;
        font-size: 1rem;
        margin-bottom: 4px;
    }

    .agent-description {
        font-size: 0.85rem;
        color: #64748b;
    }

    /* Metric/status cards */
    .status-card {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        background: #ffffff;
    }

    .status-label {
        color: #64748b;
        font-size: 0.78rem;
        margin-bottom: 4px;
    }

    .status-value {
        font-weight: 650;
        font-size: 1rem;
    }

    /* Pipeline bar */
    .pipeline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 10px;
        margin-bottom: 25px;
        padding: 15px 20px;

        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
    }

    .pipeline-step {
        text-align: center;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .pipeline-arrow {
        color: #94a3b8;
        font-size: 1.2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def extract_text(result):
    """
    Convert LangChain/agent outputs into displayable text.
    """

    if result is None:
        return ""

    # AIMessage-like object
    if hasattr(result, "content"):
        return result.content

    # Agent result containing messages
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]

        if messages:
            last_message = messages[-1]

            if hasattr(last_message, "content"):
                return last_message.content

            return str(last_message)

    return str(result)


def load_llm_configs():
    """
    Load LLM configurations from the YAML file.
    """
    config_path = Path(__file__).parent / "src" / "configs" / "llm_configs.yaml"
    
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Failed to load LLM configs: {e}")
        return {}


def initialize_session_state():

    defaults = {
        "topic": "",
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "research_complete": False,
        "selected_llm": "google_gemma",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_research():

    st.session_state.search_results = ""
    st.session_state.scraped_content = ""
    st.session_state.report = ""
    st.session_state.feedback = ""
    st.session_state.research_complete = False


# ============================================================
# INITIALIZE STATE
# ============================================================

initialize_session_state()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Research Configuration")

    st.markdown(
        """
        Configure and launch the multi-agent research pipeline.
        """
    )

    st.divider()

    # LLM Configuration
    st.markdown("### LLM Model Selection")
    
    llm_configs = load_llm_configs()
    
    if llm_configs:
        llm_options = list(llm_configs.keys())
        
        selected_llm = st.selectbox(
            "Choose LLM Model",
            options=llm_options,
            index=llm_options.index(st.session_state.selected_llm) if st.session_state.selected_llm in llm_options else 0,
            help="Select the language model to use for all agents.",
        )
        
        st.session_state.selected_llm = selected_llm
        
        # Display selected model info
        if selected_llm in llm_configs:
            st.info(
                f"**Model ID:** {llm_configs[selected_llm]['model_id']}\n\n"
                f"**Max Tokens:** {llm_configs[selected_llm]['max_tokens']}"
            )
    else:
        st.error("No LLM configurations available.")

    st.divider()

    st.markdown("### Pipeline")

    st.markdown(
        """
        **1. Search Agent**  
        Finds recent and reliable sources.

        **2. Reader Agent**  
        Selects a relevant source and reads it.

        **3. Writer Agent**  
        Generates the research report.

        **4. Critic Agent**  
        Reviews the report and provides feedback.
        """
    )

    st.divider()

    reader_context_limit = st.slider(
        "Search context sent to Reader",
        min_value=400,
        max_value=4000,
        value=800,
        step=200,
        help=(
            "Your original pipeline sends only the first "
            "800 characters of search results to the Reader Agent."
        ),
    )

    st.divider()

    if st.button(
        "🗑️ Clear Research",
        use_container_width=True,
    ):
        reset_research()
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔬 Multi-Agent Research System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-subtitle">
    Search, investigate, synthesize and critique information
    using a coordinated team of AI agents.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PIPELINE VISUALIZATION
# ============================================================

st.markdown(
    """
    <div class="pipeline">

        <div class="pipeline-step">
            🔎<br>
            Search
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-step">
            📖<br>
            Reader
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-step">
            ✍️<br>
            Writer
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-step">
            🧠<br>
            Critic
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOPIC INPUT
# ============================================================

topic = st.text_area(
    "Research Topic",
    placeholder=(
        "Example: How are world foundation models being used "
        "for autonomous driving and robotics?"
    ),
    height=100,
    value=st.session_state.topic,
)

col1, col2 = st.columns([1, 5])

with col1:

    start_research = st.button(
        "🚀 Start Research",
        type="primary",
        use_container_width=True,
    )

with col2:

    if st.session_state.research_complete:
        st.success("Research pipeline completed.")


# ============================================================
# RUN PIPELINE
# ============================================================

if start_research:

    if not topic.strip():

        st.warning("Please enter a research topic.")

    else:

        reset_research()

        st.session_state.topic = topic.strip()

        # ----------------------------------------------------
        # Pipeline progress
        # ----------------------------------------------------

        progress_bar = st.progress(0)

        overall_status = st.status(
            "Starting research pipeline...",
            expanded=True,
        )

        try:

            # ==================================================
            # CREATE LLM INSTANCE
            # ==================================================
            
            llm_configs = load_llm_configs()
            
            if not llm_configs or st.session_state.selected_llm not in llm_configs:
                st.error("Invalid LLM configuration selected.")
                st.stop()
            
            llm_config = llm_configs[st.session_state.selected_llm]
            
            overall_status.write(
                f"🤖 **Initializing LLM:** {llm_config['model_id']}..."
            )
            
            llm = create_llm(**llm_config)
            
            overall_status.write(
                "✅ LLM initialized successfully."
            )

            # ==================================================
            # STEP 1 — SEARCH AGENT
            # ==================================================

            overall_status.write(
                "🔎 **Step 1/4 — Search Agent:** finding relevant information..."
            )

            search_agent = build_search_agent(llm)

            search_results = search_agent.invoke(
                {
                    "messages": [
                        (
                            "user",
                            (
                                "Find recent, reliable and detailed "
                                f"information on: {topic}"
                            ),
                        )
                    ]
                }
            )

            search_text = extract_text(search_results)

            st.session_state.search_results = search_text

            progress_bar.progress(25)

            overall_status.write(
                "✅ Search Agent finished."
            )


            # ==================================================
            # STEP 2 — READER AGENT
            # ==================================================

            overall_status.write(
                "📖 **Step 2/4 — Reader Agent:** selecting and reading a source..."
            )

            reader_agent = build_reader_agent(llm)

            reader_prompt = (
                f"Based on the following search results about '{topic}', "
                "pick the most relevant URL and scrape it for deeper content.\n\n"
                "Search Results:\n"
                f"{search_text[:reader_context_limit]}"
            )

            reader_results = reader_agent.invoke(
                {
                    "messages": [
                        (
                            "user",
                            reader_prompt,
                        )
                    ]
                }
            )

            scraped_text = extract_text(reader_results)

            st.session_state.scraped_content = scraped_text

            progress_bar.progress(50)

            overall_status.write(
                "✅ Reader Agent finished."
            )


            # ==================================================
            # STEP 3 — WRITER AGENT
            # ==================================================

            overall_status.write(
                "✍️ **Step 3/4 — Writer Agent:** generating research report..."
            )

            research_combined = (
                "SEARCH RESULTS:\n"
                f"{search_text}\n\n"
                "DETAILED SCRAPED CONTENT:\n"
                f"{scraped_text}"
            )

            writer_chain = build_writer_chain(llm)
            
            report_result = writer_chain.invoke(
                {
                    "topic": topic,
                    "research": research_combined,
                }
            )

            report_text = extract_text(report_result)

            st.session_state.report = report_text

            progress_bar.progress(75)

            overall_status.write(
                "✅ Writer Agent finished."
            )


            # ==================================================
            # STEP 4 — CRITIC AGENT
            # ==================================================

            overall_status.write(
                "🧠 **Step 4/4 — Critic Agent:** reviewing report quality..."
            )

            critic_chain = build_critic_chain(llm)
            
            critique_result = critic_chain.invoke(
                {
                    "report": report_text
                }
            )

            critique_text = extract_text(critique_result)

            st.session_state.feedback = critique_text

            progress_bar.progress(100)

            overall_status.write(
                "✅ Critic Agent finished."
            )

            overall_status.update(
                label="Research completed successfully",
                state="complete",
                expanded=False,
            )

            st.session_state.research_complete = True

        except Exception as e:

            overall_status.update(
                label="Research pipeline failed",
                state="error",
                expanded=True,
            )

            st.error(f"Pipeline error: {e}")


# ============================================================
# RESULTS
# ============================================================

if (
    st.session_state.search_results
    or st.session_state.scraped_content
    or st.session_state.report
    or st.session_state.feedback
):

    st.divider()

    st.subheader("Research Workspace")

    search_tab, reader_tab, report_tab, critic_tab = st.tabs(
        [
            "🔎 Search Results",
            "📖 Reader",
            "📝 Research Report",
            "🧠 Critic",
        ]
    )


    # ========================================================
    # SEARCH TAB
    # ========================================================

    with search_tab:

        st.subheader("Search Agent Output")

        if st.session_state.search_results:

            st.markdown(
                st.session_state.search_results
            )

        else:

            st.info(
                "The Search Agent has not produced results yet."
            )


    # ========================================================
    # READER TAB
    # ========================================================

    with reader_tab:

        st.subheader("Reader Agent Output")

        if st.session_state.scraped_content:

            with st.container(border=True):

                st.markdown(
                    st.session_state.scraped_content
                )

        else:

            st.info(
                "The Reader Agent has not produced content yet."
            )


    # ========================================================
    # REPORT TAB
    # ========================================================

    with report_tab:

        col_report, col_download = st.columns(
            [5, 1]
        )

        with col_report:

            st.subheader("Generated Research Report")

        if st.session_state.report:

            st.markdown(
                st.session_state.report
            )

            st.download_button(
                label="⬇️ Download Report",
                data=st.session_state.report,
                file_name="research_report.md",
                mime="text/markdown",
            )

        else:

            st.info(
                "The Writer Agent has not generated a report yet."
            )


    # ========================================================
    # CRITIC TAB
    # ========================================================

    with critic_tab:

        st.subheader("Critic Agent Feedback")

        if st.session_state.feedback:

            st.markdown(
                st.session_state.feedback
            )

        else:

            st.info(
                "The Critic Agent has not reviewed the report yet."
            )