"""
GRPO Reward Functions

- format_reward_func: JSON 포맷 준수 여부
- signal_reward_func: 실제 시장 방향과 예측 일치 여부
"""

import os
import json

GRPO_LOG_PATH = None  # set by train.py


def set_log_path(path: str):
    global GRPO_LOG_PATH
    GRPO_LOG_PATH = path


def format_reward_func(completions, **kwargs) -> list[float]:
    """JSON 포맷 준수 여부 보상 (스케일 축소: signal_reward 대비 보조 역할)."""
    scores = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]["content"] if completion else ""
        score = 0.0
        text = completion.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            result = json.loads(text)
            score += 0.3  # valid JSON
            if "signal" in result and result["signal"] in ("LONG", "SHORT", "NEUTRAL"):
                score += 0.5  # valid signal
            if "reasoning" in result:
                score += 0.2  # has reasoning
        except json.JSONDecodeError:
            score = -0.5

        # addCriterion 이슈 페널티
        if len(completion) > 0:
            removal = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion) - len(removal)) / len(completion) >= 0.5:
                score -= 2.0

        scores.append(score)
    return scores


def signal_reward_func(prompts, completions, actual_signal, **kwargs) -> list[float]:
    """실제 시장 방향과 예측 signal 일치 여부 보상."""
    scores = []
    for completion, actual in zip(completions, actual_signal):
        if isinstance(completion, list):
            completion = completion[0]["content"] if completion else ""

        text = completion.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            result = json.loads(text)
            predicted = result.get("signal", "")
            if predicted == actual:
                score = 3.0  # 정답
            elif predicted in ("LONG", "SHORT", "NEUTRAL"):
                if (predicted == "LONG" and actual == "SHORT") or \
                   (predicted == "SHORT" and actual == "LONG"):
                    score = -5.0  # 반대 방향 진입: 최악
                elif actual == "NEUTRAL" and predicted in ("LONG", "SHORT"):
                    score = -3.0  # 불필요 진입: 헛진입
                else:
                    score = -0.5  # 기회 놓침 (NEUTRAL 예측): 손실 없음
            else:
                score = -1.0  # invalid signal
                predicted = f"INVALID:{predicted}"
        except (json.JSONDecodeError, AttributeError):
            score = -1.0
            predicted = "PARSE_FAIL"

        scores.append(score)

        # Log generation result
        if GRPO_LOG_PATH:
            try:
                os.makedirs(os.path.dirname(GRPO_LOG_PATH), exist_ok=True)
                with open(GRPO_LOG_PATH, "a", encoding="utf-8") as f:
                    log_entry = {
                        "actual": actual,
                        "predicted": predicted,
                        "reward": score,
                        "output": completion[:500],
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

    return scores
