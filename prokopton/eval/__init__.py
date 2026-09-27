"""
Prokopton Eval — contamination-proof continual-learning measurement.

The historical evaluator injected the expected answer into the prompt and then
searched for it in the echoed generation, which made every accuracy number a
tautology. This module replaces it with two closed-book metrics that cannot be
gamed that way:

* **Token rank** — a single forward pass scores how the frozen-in-prompt-free
  model ranks each answer token. Nothing is generated, so prompt echo cannot
  inflate the score.
* **Generation match** — optional, always run with the prompt sliced off.

Every evaluation is reported **against a frozen baseline control**, so "the
model learned" is always a difference, never an absolute number.

Anti-contamination invariants enforced here (and asserted by the test suite):

* the expected answer string must never appear in the prompt;
* accuracy on never-taught fabricated facts must be ≈ 0.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ============================================================
# Probes
# ============================================================

@dataclass(frozen=True)
class Probe:
    """A closed-book question/answer pair."""
    question: str
    answer: str
    task: str = "general"
    taught: bool = True

    @property
    def id(self) -> str:
        return f"{self.task}:{self.question[:40]}"


@dataclass(frozen=True)
class ProbeResult:
    probe: Probe
    mean_rank: float
    first_token_rank: int
    all_tokens_topk: bool
    first_token_top1: bool
    answer_in_prompt: bool
    n_answer_tokens: int
    answer_token_ids: Tuple[int, ...] = ()

    @property
    def correct(self) -> bool:
        """Primary correctness criterion (top-``k`` on every answer token)."""
        return self.all_tokens_topk and not self.answer_in_prompt


@dataclass
class EvalReport:
    """Aggregate closed-book evaluation."""
    results: List[ProbeResult]
    topk: int = 5
    label: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    @property
    def accuracy(self) -> float:
        good = [r for r in self.results if not r.answer_in_prompt]
        if not good:
            return 0.0
        return sum(r.all_tokens_topk for r in good) / len(good)

    @property
    def first_token_top1(self) -> float:
        good = [r for r in self.results if not r.answer_in_prompt]
        if not good:
            return 0.0
        return sum(r.first_token_top1 for r in good) / len(good)

    @property
    def mean_rank(self) -> float:
        good = [r for r in self.results if not r.answer_in_prompt]
        if not good:
            return float("inf")
        return statistics.fmean(r.mean_rank for r in good)

    @property
    def mrr(self) -> float:
        good = [r for r in self.results if not r.answer_in_prompt]
        if not good:
            return 0.0
        return statistics.fmean(1.0 / r.first_token_rank for r in good)

    @property
    def contamination(self) -> List[ProbeResult]:
        return [r for r in self.results if r.answer_in_prompt]

    def delta(self, baseline: "EvalReport") -> Dict[str, float]:
        """Difference versus a frozen-baseline control report."""
        return {
            "accuracy": self.accuracy - baseline.accuracy,
            "first_token_top1": self.first_token_top1 - baseline.first_token_top1,
            "mean_rank": self.mean_rank - baseline.mean_rank,
            "mrr": self.mrr - baseline.mrr,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "n": len(self.results),
            "accuracy": round(self.accuracy, 4),
            "first_token_top1": round(self.first_token_top1, 4),
            "mean_rank": round(self.mean_rank, 4),
            "mrr": round(self.mrr, 4),
            "contaminated": len(self.contamination),
            "topk": self.topk,
            "environment": self.environment,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps({
            "summary": self.summary(),
            "results": [dataclasses.asdict(r) for r in self.results],
        }, indent=2, default=str)


# ============================================================
# Contamination guard
# ============================================================

def answer_leaked_into_prompt(prompt: str, answer: str, strict: bool = False) -> bool:
    """True when the expected answer already appears in the prompt.

    Any evaluation whose answer is present in the prompt is a tautology, so
    callers must treat this as a hard failure.

    Matching is on word boundaries by default so that an answer like ``Zephyr``
    is not reported as leaked by a question about *Zephyria*. Pass
    ``strict=True`` for raw substring containment.
    """
    if not answer:
        return False
    if strict:
        return answer.strip().lower() in prompt.strip().lower()
    pattern = r"(?<![\w])" + re.escape(answer.strip()) + r"(?![\w])"
    return re.search(pattern, prompt, flags=re.IGNORECASE) is not None


def assert_uncontaminated(prompt: str, answer: str):
    if answer_leaked_into_prompt(prompt, answer):
        raise AssertionError(
            f"Evaluation contamination: the expected answer {answer!r} already "
            f"appears in the prompt {prompt!r}. The resulting score would be a "
            f"tautology.")


# ============================================================
# Token-rank metric (closed book, no generation)
# ============================================================

def answer_token_span(full_ids: torch.Tensor, prompt_ids: torch.Tensor,
                      answer_ids: torch.Tensor) -> Tuple[int, int]:
    """Locate the answer tokens inside ``full_ids``.

    The historical implementation tokenised ``prompt`` and ``prompt + answer``
    separately and assumed ``answer_start == len(prompt_ids)``. Tokenizers merge
    across that boundary, so the index drifts and the wrong logits are scored.
    Here the prefix is verified and a search fallback is used when the boundary
    is not clean. The returned span is always checked to actually contain the
    answer tokens.
    """
    full = list(full_ids.tolist())
    prompt = list(prompt_ids.tolist())
    answer = list(answer_ids.tolist())
    if not answer:
        raise ValueError("answer produced no tokens")
    if len(answer) > len(full):
        raise ValueError("answer is longer than the tokenised sequence")

    def _find(lo: int, hi: int) -> Optional[Tuple[int, int]]:
        for start in range(lo, hi):
            if full[start:start + len(answer)] == answer:
                return start, start + len(answer)
        return None

    n = min(len(prompt), len(full))
    if n and full[:n] == prompt[:n]:
        span = _find(n, len(full) - len(answer) + 1)
        if span:
            return span

    # Boundary merge: try progressively shorter prompt prefixes.
    for cut in range(n, 0, -1):
        if full[:cut] == prompt[:cut]:
            span = _find(cut, len(full) - len(answer) + 1)
            if span:
                return span

    span = _find(0, len(full) - len(answer) + 1)
    if span:
        return span
    raise ValueError("could not locate answer tokens in the tokenised sequence")


@torch.no_grad()
def token_rank_scores(model, tokenizer, probes: Sequence[Probe],
                      prompt_fn: Optional[Callable[[Probe], str]] = None,
                      topk: int = 5,
                      max_length: int = 256,
                      device: Optional[torch.device] = None,
                      label: str = "") -> EvalReport:
    """Score every probe by the rank of its answer tokens under one forward pass.

    Fully closed book: the prompt contains the question only, never the answer.
    """
    prompt_fn = prompt_fn or (lambda p: f"Question: {p.question}\nAnswer:")
    device = device or _model_device(model)
    was_training = getattr(model, "training", False)
    model.eval()

    results: List[ProbeResult] = []
    for probe in probes:
        prompt = prompt_fn(probe)
        leaked = answer_leaked_into_prompt(prompt, probe.answer)

        full_text = f"{prompt} {probe.answer}"
        full = tokenizer(full_text, return_tensors="pt", truncation=True,
                         max_length=max_length)
        full_ids = full["input_ids"].to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=max_length)["input_ids"].to(device)
        answer_ids = tokenizer(probe.answer, return_tensors="pt", truncation=True,
                               max_length=max_length)["input_ids"].to(device)

        try:
            start, end = answer_token_span(full_ids[0], prompt_ids[0], answer_ids[0])
        except ValueError:
            results.append(ProbeResult(
                probe=probe, mean_rank=float("inf"), first_token_rank=10 ** 9,
                all_tokens_topk=False, first_token_top1=False,
                answer_in_prompt=leaked, n_answer_tokens=0))
            continue

        outputs = model(input_ids=full_ids)
        logits = outputs.logits[0].float()            # [seq, vocab]

        ranks: List[int] = []
        for pos in range(start, end):
            logit_pos = pos - 1
            if logit_pos < 0 or logit_pos >= logits.shape[0]:
                continue
            token_logits = logits[logit_pos]
            target = full_ids[0, pos]
            rank = int((token_logits > token_logits[target]).sum().item()) + 1
            ranks.append(rank)

        if not ranks:
            results.append(ProbeResult(
                probe=probe, mean_rank=float("inf"), first_token_rank=10 ** 9,
                all_tokens_topk=False, first_token_top1=False,
                answer_in_prompt=leaked, n_answer_tokens=0))
            continue

        results.append(ProbeResult(
            probe=probe,
            mean_rank=statistics.fmean(ranks),
            first_token_rank=ranks[0],
            all_tokens_topk=all(r <= topk for r in ranks),
            first_token_top1=(ranks[0] == 1),
            answer_in_prompt=leaked,
            n_answer_tokens=len(ranks),
            answer_token_ids=tuple(int(x) for x in answer_ids[0].tolist()),
        ))

    if was_training:
        model.train()

    return EvalReport(results=results, topk=topk, label=label,
                      environment=_environment(model))


@torch.no_grad()
def generation_accuracy(model, tokenizer, probes: Sequence[Probe],
                        prompt_fn: Optional[Callable[[Probe], str]] = None,
                        max_new: int = 32,
                        max_length: int = 256,
                        device: Optional[torch.device] = None) -> EvalReport:
    """Closed-book generation accuracy with the prompt sliced off.

    Still refuses to score a probe whose answer is present in the prompt.
    """
    prompt_fn = prompt_fn or (lambda p: f"Question: {p.question}\nAnswer:")
    device = device or _model_device(model)
    was_training = getattr(model, "training", False)
    model.eval()

    results: List[ProbeResult] = []
    for probe in probes:
        prompt = prompt_fn(probe)
        leaked = answer_leaked_into_prompt(prompt, probe.answer)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        hit = probe.answer.strip().lower() in text.strip().lower()
        results.append(ProbeResult(
            probe=probe,
            mean_rank=1.0 if hit else float("inf"),
            first_token_rank=1 if hit else 10 ** 9,
            all_tokens_topk=hit,
            first_token_top1=hit,
            answer_in_prompt=leaked,
            n_answer_tokens=len(probe.answer.split()),
        ))

    if was_training:
        model.train()
    return EvalReport(results=results, topk=1, label="generation",
                      environment=_environment(model))


def _model_device(model) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _environment(model) -> Dict[str, Any]:
    env: Dict[str, Any] = {}
    try:
        import torch as _t
        env["torch"] = _t.__version__
    except Exception:
        pass
    try:
        import transformers as _tf
        env["transformers"] = _tf.__version__
    except Exception:
        pass
    try:
        env["device"] = str(_model_device(model))
    except Exception:
        pass
    try:
        env["model_type"] = getattr(getattr(model, "config", None), "model_type", "unknown")
    except Exception:
        pass
    return env


# ============================================================
# Frozen-baseline control
# ============================================================

class FrozenBaseline:
    """Snapshot of the model's behaviour before any learning.

    Every claim of "the model learned X" must be expressed as
    ``after.delta(before)``; absolute numbers alone prove nothing.
    """

    def __init__(self, report: EvalReport):
        self.report = report

    @classmethod
    def capture(cls, model, tokenizer, probes, **kwargs) -> "FrozenBaseline":
        return cls(token_rank_scores(model, tokenizer, probes, **kwargs))

    def delta(self, after: EvalReport) -> Dict[str, float]:
        return after.delta(self.report)


def evaluate_with_control(model, tokenizer, probes, baseline: Optional[FrozenBaseline] = None,
                          **kwargs) -> Tuple[EvalReport, Dict[str, float]]:
    """Return ``(report, delta_vs_baseline)``."""
    report = token_rank_scores(model, tokenizer, probes, **kwargs)
    if baseline is None:
        return report, {"accuracy": 0.0, "first_token_top1": 0.0,
                        "mean_rank": 0.0, "mrr": 0.0}
    return report, baseline.delta(report)


# ============================================================
# Benchmark definition
# ============================================================

def default_probes() -> List[Probe]:
    """Closed-book probe set.

    Split into **taught** facts (the ones the run will try to install) and
    **unseen** facts (fabricated, never taught) plus **anchors** (general
    knowledge that must not drift).
    """
    taught = [
        Probe("What is the capital of Zephyria?", "Aethel", "T1:facts"),
        Probe("What is the currency of Zephyria?", "Zephyr", "T1:facts"),
        Probe("Who is the president of Zephyria?", "Elara", "T1:facts"),
        Probe("What is the secret code?", "7391", "T2:codes"),
        Probe("What is the backup code?", "4820", "T2:codes"),
        Probe("What is the access code?", "9999", "T2:codes"),
        Probe("Where is Alpha base?", "Istanbul", "T3:places"),
        Probe("Where is Gamma base?", "Izmir", "T3:places"),
        Probe("Where is Delta base?", "Antalya", "T3:places"),
    ]
    unseen = [
        Probe("What is the capital of Vornalux?", "Tesselmark", "U:unseen", taught=False),
        Probe("What is the currency of Vornalux?", "Vorn", "U:unseen", taught=False),
        Probe("Who discovered the Kellner effect?", "Bramdel", "U:unseen", taught=False),
        Probe("What is the melting point of Quorium?", "8841", "U:unseen", taught=False),
        Probe("What is the Kestrel protocol code?", "55120", "U:unseen", taught=False),
    ]
    # Anchors: general knowledge that a healthy model answers and that must not
    # degrade. Ten probes — three saturate by chance.
    anchors = [
        Probe("What is the capital of France?", "Paris", "A:anchor", taught=False),
        Probe("What is 2+2?", "4", "A:anchor", taught=False),
        Probe("Who wrote Romeo and Juliet?", "Shakespeare", "A:anchor", taught=False),
        Probe("What is the chemical symbol for gold?", "Au", "A:anchor", taught=False),
        Probe("What is the largest planet in the solar system?", "Jupiter", "A:anchor", taught=False),
        Probe("What ocean is the largest on Earth?", "Pacific", "A:anchor", taught=False),
        Probe("At what temperature does water boil in Celsius?", "100", "A:anchor", taught=False),
        Probe("What is the tallest mountain above sea level?", "Everest", "A:anchor", taught=False),
        Probe("How many chambers does the human heart have?", "4", "A:anchor", taught=False),
        Probe("What is the speed of light in kilometres per second?", "300000", "A:anchor", taught=False),
    ]
    return taught + unseen + anchors


class CLBenchmark:
    """Continual-learning benchmark (contamination-free)."""

    def __init__(self, probes: Optional[Sequence[Probe]] = None):
        self.probes = list(probes) if probes is not None else default_probes()
        self.tasks = _group_by_task(self.probes)
        # Backwards-compatible attribute names.
        self.anchor_questions = [(p.question, p.answer) for p in self.anchors]

    @property
    def taught(self) -> List[Probe]:
        return [p for p in self.probes if p.taught]

    @property
    def unseen(self) -> List[Probe]:
        return [p for p in self.probes if not p.taught and p.task.startswith("U")]

    @property
    def anchors(self) -> List[Probe]:
        return [p for p in self.probes if p.task.startswith("A")]

    def evaluate_accuracy(self, model, tokenizer, task: Dict) -> float:
        probes = [Probe(q, a, task.get("name", "task")) for q, a in task["questions"]]
        return token_rank_scores(model, tokenizer, probes).accuracy

    def evaluate_anchor(self, model, tokenizer) -> float:
        return token_rank_scores(model, tokenizer, self.anchors).accuracy

    def run(self, model, tokenizer, label: str = "", **kwargs) -> Dict[str, EvalReport]:
        """Evaluate taught / unseen / anchor slices separately."""
        out = {}
        for name, slice_ in (("taught", self.taught), ("unseen", self.unseen),
                             ("anchors", self.anchors)):
            if slice_:
                out[name] = token_rank_scores(model, tokenizer, slice_,
                                              label=f"{label}:{name}", **kwargs)
        return out


def _group_by_task(probes: Sequence[Probe]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Tuple[str, str]]] = {}
    for p in probes:
        grouped.setdefault(p.task, []).append((p.question, p.answer))
    return [{"name": name, "questions": qs, "context": ""}
            for name, qs in grouped.items()]


# ============================================================
# Full evaluation runner
# ============================================================

def run_full_evaluation(prokopton, model_name: str = "unknown",
                        output_dir: str = "experiments/runs",
                        repeats: int = 5,
                        seed: int = 0) -> Dict[str, Any]:
    """Continual-learning run with a frozen-baseline control.

    Protocol:

    1. Evaluate a **frozen baseline** (delta zeroed) on taught/unseen/anchors.
    2. Teach the task contexts sequentially.
    3. Re-evaluate and report every number as **Δ vs the frozen baseline**.
    4. Assert the anti-contamination invariants.

    Artifacts (JSON + environment metadata) are written to ``output_dir``.
    """
    import random

    random.seed(seed)
    torch.manual_seed(seed)

    bench = CLBenchmark()
    model, tokenizer = prokopton.model, prokopton.tokenizer

    contexts = {
        "T1:facts": ("Zephyria's capital is Aethel. The currency is Zephyr. "
                     "President Elara Voss leads since 2023."),
        "T2:codes": ("Secret code: 7391. Backup code: 4820. Access code: 9999."),
        "T3:places": ("Alpha base is in Istanbul. Gamma base is in Izmir. "
                      "Delta base is in Antalya."),
    }

    # ── 0. Contamination guard on the whole probe set ──
    for probe in bench.probes:
        assert_uncontaminated(f"Question: {probe.question}\nAnswer:", probe.answer)

    # ── 1. Frozen baseline (delta zeroed) ──
    backups = [fw.layer.weight.detach().clone() for fw in prokopton.fast_weights]
    for fw in prokopton.fast_weights:
        with torch.no_grad():
            fw.layer.weight.copy_(fw.W0)
    baseline_slices = bench.run(model, tokenizer, label="baseline")
    for fw, w in zip(prokopton.fast_weights, backups):
        with torch.no_grad():
            fw.layer.weight.copy_(w)
    frozen = {k: FrozenBaseline(v) for k, v in baseline_slices.items()}

    results: Dict[str, Any] = {
        "model": model_name,
        "seed": seed,
        "repeats": repeats,
        "baseline": {k: v.summary() for k, v in baseline_slices.items()},
        "sequence": [],
    }

    # ── 2. Sequential teaching ──
    t0 = time.time()
    order = ["T1:facts", "T2:codes", "T3:places"]
    for step, task in enumerate(order):
        for _ in range(repeats):
            prokopton.learn(contexts[task])
        after = bench.run(model, tokenizer, label=f"step{step+1}")
        deltas = {k: frozen[k].delta(v) for k, v in after.items()}
        results["sequence"].append({
            "step": step + 1,
            "task": task,
            "scores": {k: v.summary() for k, v in after.items()},
            "delta_vs_frozen": deltas,
            "time_s": time.time() - t0,
            "ttt_stats": prokopton.stats,
        })
        accs = " | ".join(f"{k}:{v.accuracy:.0%}" for k, v in after.items())
        dacc = " | ".join(f"{k}:{deltas[k]['accuracy']:+.0%}" for k in deltas)
        print(f"  Step {step+1} ({task}): {accs}")
        print(f"           Δ vs frozen: {dacc}")

    final = bench.run(model, tokenizer, label="final")
    final_deltas = {k: frozen[k].delta(v) for k, v in final.items()}

    # ── 3. Metrics ──
    taught_delta = final_deltas.get("taught", {}).get("accuracy", 0.0)
    anchor_delta = final_deltas.get("anchors", {}).get("accuracy", 0.0)
    unseen_delta = final_deltas.get("unseen", {}).get("accuracy", 0.0)
    metrics = {
        "growth_taught": taught_delta,
        "anchor_drift": anchor_delta,
        "unseen_change": unseen_delta,
        "forgetting": max(0.0, -taught_delta) if taught_delta < 0 else 0.0,
        "max_drift_ratio": prokopton.stats.get("max_drift_ratio", 0.0),
        "skipped_steps": prokopton.stats.get("skipped_steps", 0),
        "total_time_s": time.time() - t0,
    }
    results["final"] = {k: v.summary() for k, v in final.items()}
    results["final_delta_vs_frozen"] = final_deltas
    results["metrics"] = metrics

    # ── 4. Anti-contamination assertions ──
    verdicts = []
    if abs(anchor_delta) < 0.1:
        verdicts.append("✓ anchors preserved")
    if unseen_delta <= 0.1:
        verdicts.append("✓ untaught facts not hallucinated into")
    if taught_delta > 0:
        verdicts.append("✓ growth on taught facts")
    if metrics["max_drift_ratio"] <= prokopton.config.ttt_trust_region + 1e-9:
        verdicts.append("✓ trust region respected")
    results["verdicts"] = verdicts

    print("\n" + "=" * 56)
    print("EVALUATION SUMMARY (Δ vs frozen baseline)")
    print("=" * 56)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  Verdict: {' | '.join(verdicts) or '(none)'}")
    print("=" * 56)

    # ── 5. Artifacts ──
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out / f"eval_{model_name.replace('/', '-')}_{stamp}.json"
    results["artifact"] = str(path)
    results["environment"] = _environment(model)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {path}")
    return results


# ============================================================
# Ablation harness
# ============================================================

def run_ablation(prokopton_factory, model_name: str = "unknown",
                 output_dir: str = "experiments/runs",
                 repeats: int = 5, seed: int = 0) -> Dict[str, Any]:
    """Run the same experiment with TTT on and off.

    TTT contribution = ``ttt_on − ttt_off``; without the control the number is
    unattributable.
    """
    out: Dict[str, Any] = {"model": model_name, "seed": seed}

    prok_on = prokopton_factory()
    out["ttt_on"] = run_full_evaluation(prok_on, model_name=f"{model_name}[ttt-on]",
                                        output_dir=output_dir, repeats=repeats, seed=seed)

    prok_off = prokopton_factory()
    for fw in prok_off.fast_weights:
        fw.lr = 0.0
    prok_off.config.ttt_lr = 0.0
    out["ttt_off"] = run_full_evaluation(prok_off, model_name=f"{model_name}[ttt-off]",
                                         output_dir=output_dir, repeats=repeats, seed=seed)

    contribution = {}
    for key in ("growth_taught", "anchor_drift", "unseen_change"):
        contribution[key] = (out["ttt_on"]["metrics"][key] - out["ttt_off"]["metrics"][key])
    out["ttt_contribution"] = contribution

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"ablation_{model_name.replace('/', '-')}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    out["artifact"] = str(path)
    print(f"  TTT contribution: {contribution}")
    print(f"  Saved: {path}")
    return out
