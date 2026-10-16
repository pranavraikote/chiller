"""Reference-corpus FAISS retriever for grounded feedback.

Retrieves lecture/reference chunks for an answer so feedback can cite its sources and
abstain when nothing is sufficiently relevant.
"""
import json


class LectureRetriever:
    """FAISS semantic retriever over a lecture-chunk corpus.

    chunks JSON = list of dicts with a `text` field and a source label (`label` or `file_name`).
    The embedder runs on CPU by default so it never competes with the grader for GPU memory
    and survives the model's idle-offload.
    """

    def __init__(self, index_path, chunks_path,
                 embed_model="sentence-transformers/all-mpnet-base-v2", device="cpu"):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.index = faiss.read_index(index_path)
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.embedder = SentenceTransformer(embed_model, device=device)

    def retrieve(self, query, top_k=6, min_chars=60, slide_floor=2, transcript_floor=2,
                 unique_labels=False):
        """Return list of (source_label, text, score), best first.

        Pulls a larger candidate pool, drops tiny/boilerplate chunks and exact-duplicate text,
        then guarantees the result mixes both slide and transcript sources (each up to its floor)
        so one source type can't sweep all `top_k` slots — the rest fill by score.
        """
        q = self.embedder.encode(query, normalize_embeddings=True).astype("float32").reshape(1, -1)
        scores, idx = self.index.search(q, max(top_k * 4, 32))
        cands, seen_text = [], set()
        for i, sc in zip(idx[0], scores[0]):
            if 0 <= i < len(self.chunks):
                c = self.chunks[i]
                text = c.get("text", "")
                if len(text) < min_chars or text in seen_text:      # drop exact-duplicate passages
                    continue
                seen_text.add(text)
                label = c.get("label") or c.get("file_name", "?").replace(".txt", "")
                if unique_labels and c.get("chunk_index") is not None:
                    label = f"{label}#{c.get('chunk_index')}"
                is_slide = "-slides" in label
                cands.append((label, text, float(sc), is_slide))
        chosen, seen = [], set()
        for is_slide, floor in [(True, slide_floor), (False, transcript_floor)]:
            for c in [x for x in cands if x[3] == is_slide][:floor]:
                chosen.append(c); seen.add(id(c))
        for c in cands:
            if len(chosen) >= top_k:
                break
            if id(c) not in seen:
                chosen.append(c); seen.add(id(c))
        chosen.sort(key=lambda x: x[2], reverse=True)
        return [(l, t, s) for l, t, s, _ in chosen[:top_k]]
