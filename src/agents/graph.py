"""
AlphaSignal LangGraph Pipeline
--------------------------------
Wires all five agents into a stateful directed graph.

Graph structure:
    START
      ↓
    fetch_documents          (Agent 1 — Day 11)
      ↓
    extract_financials       (Agent 2 — Day 12)
      ↓
    analyze_sentiment        (Agent 3 — Day 13)
      ↓
    check_confidence         (Agent 4 — Day 14)
      ↓              ↘
    write_report     extract_financials  ← loops back if confidence low
      ↓
    END

The conditional edge from check_confidence is the self-correcting
mechanism that makes AlphaSignal production-grade. If confidence
is too low, the system automatically re-extracts before reporting.
"""

from langgraph.graph import StateGraph, START, END

from src.agents.state import AlphaSignalState
from src.agents.nodes import (
    fetch_documents,
    extract_financials,
    analyze_sentiment,
    check_confidence,
    write_report,
    route_after_confidence_check,
)


def build_graph():
    """
    Construct and compile the AlphaSignal agent graph.

    Steps:
        1. Create a StateGraph with our state schema
        2. Add each agent as a named node
        3. Add edges defining the flow
        4. Add conditional edge for the self-checker
        5. Compile and return the executable graph
    """

    # ── 1. Initialize graph with state schema ─────────────────────────────────
    graph = StateGraph(AlphaSignalState)

    # ── 2. Add nodes (one per agent) ──────────────────────────────────────────
    graph.add_node("fetch_documents",    fetch_documents)
    graph.add_node("extract_financials", extract_financials)
    graph.add_node("analyze_sentiment",  analyze_sentiment)
    graph.add_node("check_confidence",   check_confidence)
    graph.add_node("write_report",       write_report)

    # ── 3. Add linear edges ───────────────────────────────────────────────────
    graph.add_edge(START,                "fetch_documents")
    graph.add_edge("fetch_documents",    "extract_financials")
    graph.add_edge("extract_financials", "analyze_sentiment")
    graph.add_edge("analyze_sentiment",  "check_confidence")
    graph.add_edge("write_report",       END)

    # ── 4. Add conditional edge (self-correcting loop) ────────────────────────
    # After check_confidence, route to either write_report OR back to
    # extract_financials depending on confidence score
    graph.add_conditional_edges(
        "check_confidence",                  # source node
        route_after_confidence_check,        # routing function
        {
            "write_report":       "write_report",       # if confident
            "extract_financials": "extract_financials", # if low confidence
        }
    )

    # ── 5. Compile ────────────────────────────────────────────────────────────
    compiled = graph.compile()
    return compiled


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("AlphaSignal — Agent Pipeline (Day 10: Skeleton)")
    print("=" * 55)

    # Build the graph
    pipeline = build_graph()
    print("\nGraph compiled successfully.")

    # Define initial state
    initial_state = {
        "ticker":          "AAPL",
        "filing_date":     "2024-11-01",
        "company_name":    None,
        "raw_filing_path": None,
        "clean_text_path": None,
        "sections_path":   None,
        "chunks_path":     None,
        "cik":             None,
        "financials":      None,
        "financials_path": None,
        "sentiment":       None,
        "sentiment_path":  None,
        "confidence_score": None,
        "confidence_label": None,
        "needs_recheck":   False,
        "recheck_count":   0,
        "report":          None,
        "report_path":     None,
        "errors":          [],
        "completed_steps": [],
        "current_step":    "fetch_documents",
    }

    # Run the graph
    print("\nRunning pipeline with stub agents...\n")
    print("-" * 55)

    final_state = pipeline.invoke(initial_state)

    print("-" * 55)
    print("\nPipeline complete.")
    print(f"\nTicker:          {final_state.get('ticker')}")
    print(f"Company:         {final_state.get('company_name', 'N/A')}")
    print(f"Filing date:     {final_state.get('filing_date',  'N/A')}")
    print(f"Confidence:      {final_state.get('confidence_score')} [{final_state.get('confidence_label')}]")
    print(f"Completed steps: {final_state['completed_steps']}")
    print(f"Errors:          {final_state.get('errors', [])}")
    print(f"Report saved to: {final_state.get('report_path', 'N/A')}")

    report = final_state.get("report", "")
    if report:
        print(f"\n{'='*55}")
        print("REPORT PREVIEW (first 800 chars):")
        print("=" * 55)
        print(report[:800])
        print("...")
        print("=" * 55)
        print(f"\nFull report: {final_state.get('report_path')}")

    print("\n" + "=" * 55)
    print("AlphaSignal pipeline complete. All 5 agents done.")
    print("=" * 55)