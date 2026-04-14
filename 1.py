import operator
from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os

# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def read_file(path: str) -> str:
    """Read a file from the repository."""
    return open(path).read()

@tool
def get_pr_diff(pr_number: int, repo: str) -> str:
    """Fetch the unified diff for a GitHub pull request."""
    import httpx
    resp = httpx.get(
        f"https://api.github.com/repos/{repo}/pulls/{pr_number}",
        headers={"Accept": "application/vnd.github.v3.diff",
                 "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"},
    )
    return resp.text

@tool
def post_review_comment(pr_number: int, repo: str, body: str, commit_id: str,
                        path: str, line: int) -> dict:
    """Post an inline review comment on a specific line of a PR."""
    import httpx
    resp = httpx.post(
        f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments",
        headers={"Authorization": f"Bearer {GITHUB_TOKEN}"},
        json={"body": body, "commit_id": commit_id, "path": path, "line": line},
    )
    return resp.json()

tools = [read_file, get_pr_diff, post_review_comment]
tool_node = ToolNode(tools)

# ── Model ─────────────────────────────────────────────────────────────────────

model = ChatAnthropic(model="claude-sonnet-4-6").bind_tools(tools)

SYSTEM = """You are an expert code reviewer. When reviewing a pull request:
1. Fetch the diff with get_pr_diff
2. Read any relevant context files with read_file
3. Identify: bugs, security issues, missing error handling, style violations
4. Post inline comments with post_review_comment for each finding
5. End with a summary of findings and an overall verdict (approve / request changes)"""

# ── Graph ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, operator.add]

def agent_node(state: State):
    messages = [SystemMessage(content=SYSTEM)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state: State):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

workflow = StateGraph(State)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# ── Run ───────────────────────────────────────────────────────────────────────

from agentspan.agents import AgentRuntime

with AgentRuntime() as runtime:
    result = runtime.run(app, {
        "messages": [HumanMessage(content="Review PR #1 in RizaFarheen/test")]
    })
print(result.output["messages"][-1].content)
