"""PdM RUL 운영 어시스턴트 — LangGraph StateGraph (gpt-5.4-mini).

raw OpenAI while 루프를 LangGraph 그래프로 교체했다. 도구 로직(tools.py)은 검증된
그대로 재사용하고, 바뀐 건 오케스트레이션 레이어다.

그래프 구조 (prebuilt create_react_agent 안 씀 — 직접 구성):

    START → agent ─┬─(tool_calls 있음)──────────→ tools → agent
                   ├─(드리프트 심각 & 미확인)────→ deep_check → agent
                   └─(그 외)────────────────────→ END

- agent      : 도구가 바인딩된 LLM 호출. tool_calls 포함 AI 메시지 반환.
- tools      : prebuilt ToolNode. tool_calls 실행 → ToolMessage.
- deep_check : 조건부 분기(확장성 시연). check_drift 결과 드리프트가 임계(전체의
               절반) 이상이면, 최종 답변 직전에 최근 예측을 강제로 한 번 더 확인해
               "예측 포화" 여부를 진단하고 그 사실을 컨텍스트에 주입한다.

실행:
  uv run python -m pdm.agent.agent "최근 예측 3건 보여줘"
  uv run python -m pdm.agent.agent --graph     # 그래프 구조(mermaid) 출력
  uv run python -m pdm.agent.agent             # 인터랙티브 루프
환경변수: OPENAI_API_KEY(.env), API_URL, PROMETHEUS_URL, AGENT_MODEL.
"""
import json
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv

# .env에서 OPENAI_API_KEY 등 로드 (repo 루트 기준)
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from pdm.agent.tools import LC_TOOLS, get_recent_predictions

MODEL = os.getenv("AGENT_MODEL", "gpt-5.4-mini")
DRIFT_SEVERE_RATIO = 0.5  # drifted_features / total 이 이 이상이면 심층진단 분기

SYSTEM_PROMPT = (
    "너는 PdM(예지보전) RUL 예측 시스템의 운영 어시스턴트다. "
    "사용자 질문에 답하려면 제공된 도구로 실제 데이터를 조회해서 근거를 들어 답하라. "
    "추측하지 말고 도구 결과에 기반하라. 여러 도구가 필요하면 여러 번 호출해도 된다. "
    "최종 답변은 한국어로, 핵심 수치를 인용해 간결하게."
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    deep_check_done: bool  # deep_check 분기를 1회만 타게 하는 가드


_llm_with_tools = ChatOpenAI(model=MODEL).bind_tools(LC_TOOLS)


def agent_node(state: AgentState) -> dict:
    """도구가 바인딩된 LLM 호출."""
    response = _llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def _severe_drift_seen(messages) -> bool:
    """대화 중 check_drift 결과에 심각한 드리프트(전체의 절반 이상)가 있었는가."""
    for m in messages:
        if isinstance(m, ToolMessage):
            try:
                data = json.loads(m.content)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            total = data.get("total")
            drifted = data.get("drifted_features")
            if total and drifted is not None and drifted >= total * DRIFT_SEVERE_RATIO:
                return True
    return False


def deep_check_node(state: AgentState) -> dict:
    """조건부 분기 노드: 드리프트가 심각하면 최근 예측을 추가로 확인해 '포화' 진단을
    컨텍스트에 주입한다. 그 뒤 agent로 돌아가 최종 판단에 반영시킨다."""
    recent = get_recent_predictions(limit=10)
    saturated = sum(1 for r in recent if float(r.get("rul", 0)) >= 125.0)
    note = (
        f"[심층진단] 데이터 드리프트가 심각해 최근 예측을 추가 확인함: "
        f"최근 {len(recent)}건 중 {saturated}건이 RUL 125로 포화. "
        f"드리프트와 이 포화 여부를 함께 반영해 최종 답변하라."
    )
    print(f"  [node:deep_check] 드리프트 심각 → 최근 {len(recent)}건 중 "
          f"{saturated}건 포화 진단 주입", file=sys.stderr)
    return {"messages": [SystemMessage(content=note)], "deep_check_done": True}


def route_after_agent(state: AgentState) -> str:
    """agent 노드 뒤 라우팅: tools / deep_check / END."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    # 최종 답변을 내려는 시점: 드리프트가 심각하고 아직 심층진단을 안 했으면 분기
    if not state.get("deep_check_done") and _severe_drift_seen(state["messages"]):
        return "deep_check"
    return END


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(LC_TOOLS))
    g.add_node("deep_check", deep_check_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent,
                            {"tools": "tools", "deep_check": "deep_check", END: END})
    g.add_edge("tools", "agent")
    g.add_edge("deep_check", "agent")
    return g.compile()


_graph = build_graph()


def run_agent(question: str, verbose: bool = True) -> str:
    """질문 하나를 그래프로 처리. 노드 실행 순서와 tool_call을 stderr로 로깅."""
    init = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)],
        "deep_check_done": False,
    }
    final = ""
    for event in _graph.stream(init, stream_mode="updates"):
        for node_name, update in event.items():
            for m in update.get("messages", []):
                if isinstance(m, AIMessage):
                    if m.tool_calls:
                        for tc in m.tool_calls:
                            args = json.dumps(tc["args"], ensure_ascii=False)
                            if len(args) > 120:
                                args = args[:120] + "...(생략)"
                            if verbose:
                                print(f"  [node:{node_name}] tool_call -> "
                                      f"{tc['name']}({args})", file=sys.stderr)
                    elif m.content:
                        final = m.content
                elif isinstance(m, ToolMessage):
                    if verbose:
                        rs = m.content if isinstance(m.content, str) else str(m.content)
                        print(f"  [node:{node_name}] result <- {rs[:200]}",
                              file=sys.stderr)
    return final


def main():
    args = sys.argv[1:]
    if args and args[0] == "--graph":
        print(_graph.get_graph().draw_mermaid())
        return
    if args:
        print(run_agent(" ".join(args)))
        return
    print("PdM 운영 어시스턴트 (LangGraph, gpt-5.4-mini). 질문 입력. 종료: Ctrl-D")
    try:
        while True:
            question = input("\n> ").strip()
            if question:
                print(run_agent(question))
    except (EOFError, KeyboardInterrupt):
        print("\n종료.")


if __name__ == "__main__":
    main()
