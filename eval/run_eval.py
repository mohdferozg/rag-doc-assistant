"""Evaluation harness: scores RAG answers using an LLM-as-a-judge.

Run: python -m eval.run_eval data/security-genai.pdf
"""
import json
import sys
import time
from pathlib import Path

from google import genai

from src.config import require_api_key
from src.model_selector import pick_model
from src.pipeline import RAGPipeline

JUDGE_PROMPT = """You are an impartial evaluator for a RAG (Retrieval-Augmented Generation) system.

You will be given:
1. A QUESTION
2. The EXPECTED ANSWER (ground truth)
3. The ACTUAL ANSWER produced by the system
4. The CONTEXT that the system retrieved.

Your job is to output a JSON object with exactly two keys:
- "correct": true if the ACTUAL ANSWER contains the key information from the EXPECTED ANSWER, false otherwise.
- "reason": a one-sentence explanation of your score.

Do not be overly strict about exact wording; focus on factual accuracy.
If the ACTUAL ANSWER says "I don't know" but the EXPECTED ANSWER contains facts, score it as false.

Output ONLY valid JSON. No markdown formatting, no preamble.

QUESTION:
{question}

EXPECTED ANSWER:
{expected}

ACTUAL ANSWER:
{actual}

RETRIEVED CONTEXT:
{context}
"""


def main(pdf_path: str):
    golden_path = Path("eval/golden_set.json")
    if not golden_path.exists():
        print(f"Error: {golden_path} not found.")
        sys.exit(1)

    with open(golden_path) as f:
        golden = json.load(f)

    print(f"Loading pipeline and ingesting {pdf_path}...")
    pipe = RAGPipeline()
    pipe.ingest(pdf_path)

    client = genai.Client(api_key=require_api_key())
    model_name = pick_model(client)

    results = []
    passed = 0

    print(f"\nRunning {len(golden)} evaluation questions...\n")

    for item in golden:
        q = item["question"]
        print(f"[{item['id']}] {q}")
        
        # 1. Get system answer
        result = pipe.ask(q)
        actual = result["answer"]
        context_text = "\n\n".join([c["text"] for c in result["retrieved"]])

        # 2. Ask judge to score
        prompt = JUDGE_PROMPT.format(
            question=q,
            expected=item["expected_answer"],
            actual=actual,
            context=context_text,
        )
        
        # Simple retry for the judge
        for attempt in range(3):
            try:
                judge_resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                judge_text = judge_resp.text.strip()
                # Clean possible markdown blocks
                if judge_text.startswith("```json"):
                    judge_text = judge_text.replace("```json", "").replace("```", "").strip()
                
                eval_data = json.loads(judge_text)
                break
            except Exception as e:
                if attempt == 2:
                    eval_data = {"correct": False, "reason": f"Judge error: {e}"}
                time.sleep(2)

        status = "✅ PASS" if eval_data.get("correct") else "❌ FAIL"
        if eval_data.get("correct"):
            passed += 1
            
        print(f"  {status}")
        print(f"  Reason: {eval_data.get('reason')}\n")
        
        results.append({
            "id": item["id"],
            "question": q,
            "actual": actual,
            "correct": eval_data.get("correct"),
            "reason": eval_data.get("reason"),
        })

    # Summary
    accuracy = (passed / len(golden)) * 100
    print("=" * 60)
    print(f"EVALUATION COMPLETE: {passed}/{len(golden)} passed ({accuracy:.1f}%)")
    print("=" * 60)

    # Save results
    out_path = Path("eval/results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m eval.run_eval <path-to-pdf>")
        sys.exit(1)
    main(sys.argv[1])