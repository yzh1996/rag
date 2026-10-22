"""Read-only live acceptance: model calls and local synthetic business data only."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import settings
from core.auth import ANONYMOUS_USER
from core.conversation_agent import ConversationAgent
from core.llm_client import get_llm


class LazyRetriever:
    def retrieve(self, *args, **kwargs):
        from core.retriever import get_rag_retriever
        return get_rag_retriever().retrieve(*args, **kwargs)


agent = ConversationAgent(get_llm(), LazyRetriever())
state, history, results = {}, [], []
for question in ["帮我复盘清研美妆旗舰店最新一场直播，看看整体卖得怎么样。", "那和上一场比呢？"]:
    started = time.monotonic()
    result = None
    for event in agent.events(question, history, user=ANONYMOUS_USER, task_state=state):
        step = event.get("agent_step")
        if step and step["status"] != "running":
            print(json.dumps(step, ensure_ascii=True), flush=True)
        if "answer" in event:
            result = event
    if result is None:
        raise RuntimeError("No final result")
    state = result["query_details"]["task_state"]
    results.append({"question": question, "elapsed_seconds": round(time.monotonic()-started, 2),
        **{k: v for k, v in result.items() if k != "context"}})
    from langchain_core.messages import AIMessage, HumanMessage
    history.extend([HumanMessage(content=question), AIMessage(content=result["answer"],
        additional_kwargs={"query_details": result["query_details"]})])
    print(json.dumps({"finished": len(results), "response_type": result["response_type"],
        "stop_reason": result["query_details"]["stop_reason"], "facts": state.get("facts"),
        "answer": result["answer"][:120]}, ensure_ascii=True), flush=True)
    if result["response_type"] == "error":
        break
target = Path(__file__).with_name("live-results.json")
target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(str(target), flush=True)
