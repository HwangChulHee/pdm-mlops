"""PdM RUL 운영 어시스턴트 (gpt-5.4-mini + function calling).

표준 tool-use 루프: 모델이 tool_call을 요청하면 해당 도구를 실행하고 결과를 다시
모델에 넘긴다. 모델이 도구를 또 부를 수도 있으므로 while 루프로 여러 단계를 돈다
(single-shot 아님 — 이게 multi-step의 핵심).

실행:
  uv run python -m pdm.agent.agent "최근 예측 3건 보여줘"
  uv run python -m pdm.agent.agent            # 인자 없으면 인터랙티브 루프
환경변수: OPENAI_API_KEY(.env), API_URL, PROMETHEUS_URL.
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# .env에서 OPENAI_API_KEY 등 로드 (repo 루트 기준)
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from openai import OpenAI

from pdm.agent.tools import TOOLS, TOOL_SCHEMAS

MODEL = os.getenv("AGENT_MODEL", "gpt-5.4-mini")
MAX_STEPS = 8  # tool 호출 라운드 상한 (무한루프 방지)

SYSTEM_PROMPT = (
    "너는 PdM(예지보전) RUL 예측 시스템의 운영 어시스턴트다. "
    "사용자 질문에 답하려면 제공된 도구로 실제 데이터를 조회해서 근거를 들어 답하라. "
    "추측하지 말고 도구 결과에 기반하라. 여러 도구가 필요하면 여러 번 호출해도 된다. "
    "최종 답변은 한국어로, 핵심 수치를 인용해 간결하게."
)


def run_agent(question: str, verbose: bool = True) -> str:
    """질문 하나를 tool-use 루프로 처리하고 최종 답변 문자열을 반환한다."""
    client = OpenAI()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        # assistant의 tool_call 메시지를 대화에 추가
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name,
                                 "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        # 요청된 도구들 실행 후 결과를 tool 메시지로 추가
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            if verbose:
                short = json.dumps(args, ensure_ascii=False)
                if len(short) > 120:
                    short = short[:120] + "...(생략)"
                print(f"  [step {step}] tool_call -> {name}({short})", file=sys.stderr)
            try:
                result = TOOLS[name](**args)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            if verbose:
                rs = json.dumps(result, ensure_ascii=False, default=str)
                print(f"             result <- {rs[:200]}", file=sys.stderr)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    return "(도구 호출 단계 상한에 도달해 종료. 질문을 더 좁혀서 다시 시도해줘.)"


def main():
    args = sys.argv[1:]
    if args:
        question = " ".join(args)
        print(run_agent(question))
        return
    print("PdM 운영 어시스턴트 (gpt-5.4-mini). 질문을 입력하세요. 종료: Ctrl-D")
    try:
        while True:
            question = input("\n> ").strip()
            if not question:
                continue
            print(run_agent(question))
    except (EOFError, KeyboardInterrupt):
        print("\n종료.")


if __name__ == "__main__":
    main()
