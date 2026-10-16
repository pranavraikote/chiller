"""Inference primitives: candidate scoring, Gradient×Input attribution, and feedback generation."""

import json
import re

import torch

from config import RAG_MIN_SCORE, RAG_TOP_K, USE_RAG
from feedback_proc import fb_mgr
from logging_setup import log
from models import mgr
from prompts import FB_RAG_SYSTEM, FB_RAG_USER, FEEDBACK_SYSTEM, FEEDBACK_USER


def _candidate(g: int, M: int) -> str:
    return json.dumps({"grade": g, "max_grade": M}, separators=(",", ":"))


# Closed-class words whose salience isn't meaningful to show (raw G×I otherwise ranks "the" highly).
_ATTR_STOP = frozenset(
    "the a an of to and or is are was were be been being in on at for with as by that this these those "
    "it its from into than then so we you they i he she him her his their our your not no if but can will "
    "would should could do does did has have had more most some any each".split()
)


def _is_noncontent(word: str) -> bool:
    core = re.sub(r"[^A-Za-z0-9]", "", word).lower()
    return core == "" or core in _ATTR_STOP


def _words_from_salience(pieces: list) -> list:
    """Turn (decoded_text, salience) pieces into per-word [word, score] salience for the UI heatmap.

    Splits on any whitespace (so compound tokens like "a)\\n\\nThe" become separate words, each inheriting
    its source piece's salience averaged over its characters), then zeroes stopword/punctuation-only words
    so the view highlights content, not "the". Keeps every content word so the answer text stays intact.
    """
    tokens, buf, sc = [], [], []
    for text, norm in pieces:
        for ch in text:
            if ch.isspace():
                if buf:
                    tokens.append(["".join(buf), sum(sc) / len(sc)])
                    buf, sc = [], []
            else:
                buf.append(ch)
                sc.append(float(norm))
    if buf:
        tokens.append(["".join(buf), sum(sc) / len(sc)])
    return [[w, round(0.0 if _is_noncontent(w) else s, 5)] for w, s in tokens]


# A few relational/operator macros map to ASCII so they stay readable instead of being dropped.
_MATH_SYMBOLS = {
    r"\geq": ">=", r"\ge": ">=", r"\leq": "<=", r"\le": "<=", r"\neq": "!=",
    r"\times": "x", r"\cdot": "*", r"\pm": "+/-", r"\approx": "~",
    r"\rightarrow": "->", r"\to": "->", r"\infty": "inf", r"\sum": "sum",
}


def _strip_latex(text: str) -> str:
    """Convert any LaTeX/math markup to readable plain text (the UI renders plain text).

    General, not pattern-specific: strips math delimiters, turns \\frac{a}{b} into (a)/(b), unwraps any
    \\command{...} to its contents (repeatedly, for nesting), maps a few symbol macros to ASCII, and
    drops anything else backslash-escaped. Citation brackets like [DAMI_2021_L10] are preserved — only
    backslash-escaped delimiters (\\[ \\]) and dollar signs are removed, never bare square brackets.
    """
    t = re.sub(r"\\[()\[\]]", "", text)          # \( \) \[ \] math delimiters
    t = t.replace("$$", "").replace("$", "")     # $…$ / $$…$$ delimiters
    t = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", t)
    prev = None                                  # \cmd{content} -> content, until stable (nesting)
    while prev != t:
        prev, t = t, re.sub(r"\\[a-zA-Z]+\s*\{([^{}]*)\}", r"\1", t)
    # Drop pure formatting/sizing macros first (they carry no content).
    t = re.sub(r"\\(?:left|right|bigg?|Bigg?|displaystyle|textstyle|scriptstyle|limits|nonumber"
               r"|quad|qquad)\b", "", t)
    # Map symbol macros to ASCII, word-bounded so \le doesn't match inside e.g. \left.
    for k, v in _MATH_SYMBOLS.items():
        t = re.sub(re.escape(k) + r"\b", v, t)
    t = t.replace(r"\|", "|")                    # norm bars
    # Keep the NAME of any other \command (so \max, \log, \alpha survive as words).
    t = re.sub(r"\\([a-zA-Z]+)", r"\1", t)
    t = re.sub(r"\\[^a-zA-Z]", "", t)            # drop spacing/escaped punctuation (\, \; \! \{ …)
    t = t.replace("\\", "")                      # drop lone backslashes
    return re.sub(r"[ \t]{2,}", " ", t).strip()


@torch.no_grad()
def _score(prompt: str, cont: str) -> float:
    enc_p  = mgr.tokenizer(prompt, return_tensors="pt")
    enc_c  = mgr.tokenizer(cont,   return_tensors="pt")
    ids    = torch.cat([enc_p["input_ids"], enc_c["input_ids"]], dim=1).to(mgr.model.device)
    labels = ids.clone()
    labels[:, :enc_p["input_ids"].shape[1]] = -100
    out    = mgr.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       labels=labels, use_cache=False)
    logits   = out.logits[:, :-1, :]
    next_ids = ids[:, 1:]
    lp = torch.log_softmax(logits, dim=-1).gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
    return (lp * (labels[:, 1:] != -100)).sum().item()


def _compute_attribution(prompt: str, response: str, grade: int, max_grade: int) -> list:
    M    = int(max_grade)
    cont = _candidate(grade, M)

    enc_p  = mgr.tokenizer(prompt, return_tensors="pt")
    enc_c  = mgr.tokenizer(cont,   return_tensors="pt")
    ids    = torch.cat([enc_p["input_ids"], enc_c["input_ids"]], dim=1).to(mgr.model.device)
    labels = ids.clone()
    labels[:, :enc_p["input_ids"].shape[1]] = -100

    actual_embeds = mgr.model.get_input_embeddings()(ids).detach().requires_grad_(True)  # [1, seq, hidden]
    attn     = torch.ones(ids.shape, device=mgr.model.device)
    next_ids = ids[:, 1:]
    mask     = (labels[:, 1:] != -100)

    # One backward pass: gradient of the chosen grade's log-prob w.r.t. the input embeddings.
    mgr.model.zero_grad()
    out = mgr.model(inputs_embeds=actual_embeds, attention_mask=attn, labels=labels, use_cache=False)
    lp  = torch.log_softmax(out.logits[:, :-1, :], dim=-1).gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
    (lp * mask).sum().backward()

    # Gradient x Input: grad ⊙ embedding, L2 over the hidden dim → per-token salience for this grade.
    log.info(f"GxI attribution: seq_len={ids.shape[1]}")
    gxi = (actual_embeds * actual_embeds.grad).detach()[0].norm(dim=-1).cpu().float().numpy()
    char_s   = prompt.find(response)
    if char_s == -1:
        log.warning("Attribution: response string not found in prompt — returning empty.")
        return []
    char_e   = char_s + len(response)
    enc_off  = mgr.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets  = enc_off["offset_mapping"][0].tolist()
    ans_mask = [(e > char_s and s < char_e and e > s) for s, e in offsets]

    answer_ids   = [ids[0, i].item() for i, m in enumerate(ans_mask) if m]
    answer_norms = [gxi[i]           for i, m in enumerate(ans_mask) if m]

    special = set(mgr.tokenizer.all_special_ids)
    pieces  = [(mgr.tokenizer.decode([tid]), nm)
               for tid, nm in zip(answer_ids, answer_norms) if tid not in special]
    return _words_from_salience(pieces)


def _gen(system: str, user: str, max_new_tokens: int = 220) -> str:
    """Deterministic chat generation (greedy) — reproducible feedback for the demo.

    Prefers the dedicated feedback model (larger instruct model in a killable subprocess): the grader's
    7B+LoRA can't reliably check "is X already in the answer?" and fabricates gaps. If no feedback model
    is configured, or the worker fails, falls back to the grader's BASE model with the adapter disabled —
    the LoRA is tuned for grade-token likelihood, not prose. Grading/attribution always keep the adapter on.
    """
    if fb_mgr.available:
        try:
            return _strip_latex(fb_mgr.generate(system, user, max_new_tokens,
                                                exclude_gpu=mgr.gpu if mgr.loaded else None))
        except Exception as e:
            log.error(f"Feedback worker failed ({type(e).__name__}): {e} — falling back to grader base.")
            fb_mgr.offload()

    # Fallback: grader base model, adapter disabled.
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = mgr.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = mgr.tokenizer(prompt, return_tensors="pt").to(mgr.model.device)
    with torch.no_grad(), mgr.model.disable_adapter():
        out = mgr.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=mgr.tokenizer.eos_token_id,
        )
    return _strip_latex(mgr.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


def _ungrounded(question, response, grade, max_grade, top_score=0.0):
    text = _gen(FEEDBACK_SYSTEM, FEEDBACK_USER.format(
        q=question, a=response, g=grade, max_g=max_grade))
    return {"feedback": text, "grounded": False, "top_score": round(top_score, 3), "excerpts": []}


def _generate_feedback(question: str, response: str, grade: int, max_grade: int) -> dict:
    """Grounded feedback: retrieve lecture excerpts, generate cited feedback, mark which were cited.

    Abstains to ungrounded feedback (no citations) when nothing is sufficiently relevant, so the
    model can't fabricate citations for material the corpus doesn't cover.
    """
    if not USE_RAG or mgr.retriever is None:
        return _ungrounded(question, response, grade, max_grade)

    chunks    = mgr.retriever.retrieve(f"{question} {response}", top_k=RAG_TOP_K)
    top_score = max((sc for _, _, sc in chunks), default=0.0)
    if top_score < RAG_MIN_SCORE:
        return _ungrounded(question, response, grade, max_grade, top_score)

    context = "\n\n".join(f"[{s}] {t}" for s, t, _ in chunks)
    text    = _gen(FB_RAG_SYSTEM, FB_RAG_USER.format(
        context=context, q=question, a=response, g=grade, max_g=max_grade))
    # Labels the model actually cited inside [...], splitting multi-cite groups like [a, b].
    in_brackets = {t.strip() for grp in re.findall(r"\[([^\]]+)\]", text) for t in grp.split(",")}
    excerpts = [{"label": s, "text": t, "score": round(sc, 3), "cited": s in in_brackets}
                for s, t, sc in chunks]
    return {"feedback": text, "grounded": True, "top_score": round(top_score, 3), "excerpts": excerpts}
