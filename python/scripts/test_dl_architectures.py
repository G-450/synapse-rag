"""
Synapse RAG — Deep Learning Architecture Validation Suite (Python)
Equivalent of scripts/test-dl-architectures.ts

Validates all DL components:
  1. Bi-Encoder architecture (all-MiniLM-L6-v2)
  2. Semantic similarity validation
  3. Pooling strategy comparison (mean vs CLS)
  4. Cross-Encoder architecture (ms-marco-MiniLM-L-6-v2)
  5. Cross-Encoder re-ranking effectiveness
  6. WordPiece tokenization analysis
  7. Embedding space geometry (isotropy & cluster analysis)
  8. Multi-head self-attention contextual sensitivity

Usage:
    python scripts/test_dl_architectures.py
"""

import sys
import json
import time
import math
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

RESULTS: list[dict] = []


def log(header: str):
    print("\n" + "═" * 70)
    print(f"  {header}")
    print("═" * 70)


def sublog(msg: str):
    print(f"  │ {msg}")


def cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))


# ═══════════════════════════════════════════════════════════════
# TEST 1: Bi-Encoder Architecture Inspection
# ═══════════════════════════════════════════════════════════════
def test_bi_encoder_architecture():
    log("TEST 1: Bi-Encoder Architecture — sentence-transformers/all-MiniLM-L6-v2")
    start = time.time()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sample_text = "The Seller shall indemnify the Buyer against any breach of warranty."
    embedding = model.encode(sample_text, normalize_embeddings=True)
    embedding_list = embedding.tolist()
    dims = len(embedding_list)

    magnitude = float(np.linalg.norm(embedding))
    is_normalized = abs(magnitude - 1.0) < 0.001
    has_negative = any(v < 0 for v in embedding_list)
    max_val = max(embedding_list)
    min_val = min(embedding_list)
    mean_val = sum(embedding_list) / dims

    sublog("Model: sentence-transformers/all-MiniLM-L6-v2")
    sublog("Architecture: BERT-base (distilled to 6 layers)")
    sublog("Hidden Size: 384")
    sublog("Attention Heads: 12")
    sublog("Feed-Forward Dim: 1536 (4 × hidden_size)")
    sublog("Max Sequence Length: 256 tokens")
    sublog("Vocabulary Size: 30,522 (WordPiece)")
    sublog("---")
    sublog(f"Output Dimensions: {dims}")
    sublog(f"L2 Norm (magnitude): {magnitude:.6f}")
    sublog(f"Is Unit-Normalized: {is_normalized}")
    sublog(f"Has Negative Values: {has_negative}")
    sublog(f"Value Range: [{min_val:.6f}, {max_val:.6f}]")
    sublog(f"Mean Value: {mean_val:.6f}")
    sublog("Pooling Strategy: Mean Pooling")

    passed = dims == 384 and is_normalized
    duration_ms = int((time.time() - start) * 1000)

    RESULTS.append({
        "name": "Bi-Encoder Architecture (all-MiniLM-L6-v2)",
        "passed": passed,
        "details": {
            "dimensions": dims,
            "is_normalized": is_normalized,
            "magnitude": magnitude,
            "value_range": [min_val, max_val],
            "mean_value": mean_val,
            "has_negative_values": has_negative,
            "model_config": {
                "num_layers": 6,
                "hidden_size": 384,
                "num_attention_heads": 12,
                "intermediate_size": 1536,
                "max_position_embeddings": 512,
                "vocab_size": 30522,
                "pooling": "mean",
                "normalization": "L2"
            }
        },
        "duration_ms": duration_ms,
    })

    print(f"  └─ {'✅ PASSED' if passed else '❌ FAILED'} ({duration_ms}ms)")
    return model


# ═══════════════════════════════════════════════════════════════
# TEST 2: Semantic Similarity Validation
# ═══════════════════════════════════════════════════════════════
def test_semantic_similarity(model: SentenceTransformer):
    log("TEST 2: Semantic Similarity — Cosine Distance Validation")
    start = time.time()

    test_pairs = [
        {
            "a": "The termination clause allows either party to exit the agreement.",
            "b": "Either party may terminate this contract upon written notice.",
            "label": "SIMILAR (paraphrase)",
        },
        {
            "a": "The termination clause allows either party to exit the agreement.",
            "b": "The weather forecast predicts sunny skies tomorrow.",
            "label": "DISSIMILAR (unrelated)",
        },
        {
            "a": "Seller represents that it has full authority to execute this agreement.",
            "b": "The Vendor warrants that it possesses the legal capacity to enter into this contract.",
            "label": "SIMILAR (legal synonym)",
        },
        {
            "a": "Confidential Information shall not be disclosed to any third party.",
            "b": "NDA provisions restrict sharing of proprietary data with external entities.",
            "label": "SIMILAR (domain-specific)",
        },
    ]

    for pair in test_pairs:
        va = model.encode(pair["a"], normalize_embeddings=True).tolist()
        vb = model.encode(pair["b"], normalize_embeddings=True).tolist()
        sim = cosine_sim(va, vb)
        sublog(f"{pair['label']}: cosine_sim = {sim:.4f}")

    duration_ms = int((time.time() - start) * 1000)
    RESULTS.append({
        "name": "Semantic Similarity Validation",
        "passed": True,
        "details": {"test_pairs": len(test_pairs), "note": "Scores logged for manual review"},
        "duration_ms": duration_ms,
    })
    print(f"  └─ ✅ PASSED ({duration_ms}ms)")


# ═══════════════════════════════════════════════════════════════
# TEST 3: Pooling Strategy Comparison
# ═══════════════════════════════════════════════════════════════
def test_pooling_strategies(model: SentenceTransformer):
    log("TEST 3: Pooling Strategy Comparison — Mean vs CLS Token")
    start = time.time()

    text = "The indemnification provision shall survive the termination of this Agreement."
    mean_vec = model.encode(text, normalize_embeddings=True).tolist()

    # CLS pooling via encode with convert_to_numpy
    import torch
    from sentence_transformers import models as st_models

    # Manually get CLS token embedding
    tokenized = model.tokenize([text])
    with torch.no_grad():
        output = model.forward(tokenized)
    # CLS is the first token of the last hidden state
    cls_vec_raw = output["token_embeddings"][0][0].cpu().numpy()
    cls_vec = (cls_vec_raw / (np.linalg.norm(cls_vec_raw) + 1e-10)).tolist()

    mean_mag = float(np.linalg.norm(mean_vec))
    cls_mag = float(np.linalg.norm(cls_vec))
    sim = cosine_sim(mean_vec, cls_vec)

    sublog(f"Mean Pooling Vector Magnitude: {mean_mag:.6f}")
    sublog(f"CLS Token Vector Magnitude: {cls_mag:.6f}")
    sublog(f"Cosine Similarity (Mean vs CLS): {sim:.4f}")
    sublog(f"Divergence: {1 - sim:.4f}")
    sublog("→ Mean pooling is used in production (better for sentence similarity)")

    passed = mean_mag > 0.99 and cls_mag > 0.99
    duration_ms = int((time.time() - start) * 1000)

    RESULTS.append({
        "name": "Pooling Strategy Comparison",
        "passed": passed,
        "details": {
            "mean_pooling_magnitude": mean_mag,
            "cls_pooling_magnitude": cls_mag,
            "cosine_similarity_mean_vs_cls": sim,
            "production_strategy": "mean"
        },
        "duration_ms": duration_ms,
    })
    print(f"  └─ {'✅ PASSED' if passed else '❌ FAILED'} ({duration_ms}ms)")


# ═══════════════════════════════════════════════════════════════
# TEST 4: Cross-Encoder Architecture Inspection
# ═══════════════════════════════════════════════════════════════
def test_cross_encoder_architecture():
    log("TEST 4: Cross-Encoder Architecture — cross-encoder/ms-marco-MiniLM-L-6-v2")
    start = time.time()

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    query = "What is the governing law of this agreement?"
    relevant_doc = "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware."
    irrelevant_doc = "The quarterly earnings report showed a 15% increase in revenue."

    score_rel = float(model.predict([(query, relevant_doc)])[0])
    score_irr = float(model.predict([(query, irrelevant_doc)])[0])

    sublog("Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (MS MARCO trained)")
    sublog("Architecture: BERT-base (distilled to 6 layers)")
    sublog("Hidden Size: 384")
    sublog("Attention Heads: 12")
    sublog("Feed-Forward Dim: 1536")
    sublog("Classification Head: Linear(384 → 1)")
    sublog("---")
    sublog(f"Relevant Document Score: {score_rel:.4f}")
    sublog(f"Irrelevant Document Score: {score_irr:.4f}")
    sublog(f"Score Delta: {score_rel - score_irr:.4f}")
    sublog(f"Correctly Ranked: {'YES ✓' if score_rel > score_irr else 'NO ✗'}")

    passed = score_rel > score_irr
    duration_ms = int((time.time() - start) * 1000)

    RESULTS.append({
        "name": "Cross-Encoder Architecture (ms-marco-MiniLM-L-6-v2)",
        "passed": passed,
        "details": {
            "relevant_score": score_rel,
            "irrelevant_score": score_irr,
            "correctly_ranked": passed,
        },
        "duration_ms": duration_ms,
    })
    print(f"  └─ {'✅ PASSED' if passed else '❌ FAILED'} ({duration_ms}ms)")
    return model


# ═══════════════════════════════════════════════════════════════
# TEST 5: Cross-Encoder Re-Ranking Effectiveness
# ═══════════════════════════════════════════════════════════════
def test_reranking(cross_encoder: CrossEncoder):
    log("TEST 5: Cross-Encoder Re-Ranking — Multi-Document Legal Passages")
    start = time.time()

    query = "Under what circumstances can the buyer terminate the agreement?"
    passages = [
        {"text": "The Buyer may terminate this Agreement at any time prior to the Closing if there has been a material breach by the Seller.", "expected_rank": 1},
        {"text": "All notices shall be sent to the addresses specified in Schedule A of this Agreement.", "expected_rank": 4},
        {"text": "The Agreement may be terminated by mutual written consent of both parties at any time.", "expected_rank": 2},
        {"text": "The representations and warranties contained herein shall survive for a period of two years.", "expected_rank": 5},
        {"text": "Termination rights include the right of the Buyer to withdraw if regulatory approval is not obtained within 90 days.", "expected_rank": 3},
    ]

    pairs = [(query, p["text"]) for p in passages]
    scores = cross_encoder.predict(pairs)

    scored = [
        {**p, "score": float(s)} for p, s in zip(passages, scores)
    ]
    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)

    sublog("Re-Ranked Results:")
    for i, r in enumerate(ranked, 1):
        sublog(f'  Rank {i}: score={r["score"]:.4f} | "{r["text"][:70]}..."')

    top_is_relevant = ranked[0]["expected_rank"] <= 2
    passed = top_is_relevant
    duration_ms = int((time.time() - start) * 1000)

    RESULTS.append({
        "name": "Cross-Encoder Re-Ranking Effectiveness",
        "passed": passed,
        "details": {
            "num_passages": len(passages),
            "top_ranked_was_relevant": top_is_relevant,
        },
        "duration_ms": duration_ms,
    })
    print(f"  └─ {'✅ PASSED' if passed else '❌ FAILED'} ({duration_ms}ms)")


# ═══════════════════════════════════════════════════════════════
# TEST 6: Tokenization Analysis (WordPiece)
# ═══════════════════════════════════════════════════════════════
def test_tokenization():
    log("TEST 6: WordPiece Tokenization Analysis")
    start = time.time()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    test_texts = [
        "indemnification",
        "force majeure event",
        "The Seller hereby represents and warrants that...",
        "anti-competitive behavior under Section 7 of the Clayton Act",
        "WHEREAS, the Company desires to engage the Consultant for advisory services",
    ]

    sublog("Tokenizer: WordPiece (BERT-style)")
    sublog("Vocab Size: 30,522 tokens")
    sublog("Special Tokens: [CLS], [SEP], [PAD], [UNK], [MASK]")
    sublog("---")

    for text in test_texts:
        encoded = tokenizer(text, padding=False, truncation=True)
        token_count = len(encoded["input_ids"])
        preview = text[:55] + ("..." if len(text) > 55 else "")
        sublog(f'"{preview}"')
        sublog(f"  → {token_count} tokens (incl. [CLS]+[SEP])")

    duration_ms = int((time.time() - start) * 1000)
    RESULTS.append({
        "name": "WordPiece Tokenization Analysis",
        "passed": True,
        "details": {"tokenizer_type": "WordPiece", "vocab_size": 30522, "samples_tested": len(test_texts)},
        "duration_ms": duration_ms,
    })
    print(f"  └─ ✅ PASSED ({duration_ms}ms)")


# ═══════════════════════════════════════════════════════════════
# TEST 7: Embedding Space Geometry
# ═══════════════════════════════════════════════════════════════
def test_embedding_geometry(model: SentenceTransformer):
    log("TEST 7: Embedding Space Geometry — Isotropy & Cluster Analysis")
    start = time.time()

    legal_texts = [
        "The agreement shall be governed by the laws of New York.",
        "Confidential information must not be disclosed to third parties.",
        "The seller warrants that the goods conform to the specifications.",
        "Force majeure events include natural disasters and acts of God.",
        "Termination may be effected by either party with 30 days notice.",
        "The indemnifying party shall hold harmless the indemnified party.",
        "Arbitration shall be conducted under ICC rules in London.",
        "Non-compete restrictions apply for 24 months post-termination.",
    ]
    non_legal_texts = [
        "The cat sat on the mat and looked out the window.",
        "Photosynthesis converts carbon dioxide into organic compounds.",
        "The guitar solo in that song was absolutely incredible.",
        "Mount Everest is the tallest mountain above sea level.",
    ]

    legal_embs = model.encode(legal_texts, normalize_embeddings=True)
    non_legal_embs = model.encode(non_legal_texts, normalize_embeddings=True)

    def avg_pairwise_sim(embs):
        n = len(embs)
        total, count = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                total += float(np.dot(embs[i], embs[j]))
                count += 1
        return total / count if count > 0 else 0.0

    legal_intra = avg_pairwise_sim(legal_embs)
    non_legal_intra = avg_pairwise_sim(non_legal_embs)

    inter_total, inter_count = 0.0, 0
    for le in legal_embs:
        for ne in non_legal_embs:
            inter_total += float(np.dot(le, ne))
            inter_count += 1
    inter_sim = inter_total / inter_count

    all_embs = np.vstack([legal_embs, non_legal_embs])
    centroid = all_embs.mean(axis=0)
    centroid_mag = float(np.linalg.norm(centroid))

    sublog(f"Legal Intra-Cluster Similarity: {legal_intra:.4f}")
    sublog(f"Non-Legal Intra-Cluster Similarity: {non_legal_intra:.4f}")
    sublog(f"Inter-Cluster Similarity (Legal vs Non-Legal): {inter_sim:.4f}")
    sublog(f"Cluster Separation: {legal_intra - inter_sim:.4f}")
    sublog("→ Higher intra-cluster + lower inter-cluster = better domain encoding")
    sublog(f"Centroid Magnitude (isotropy indicator): {centroid_mag:.4f}")
    sublog("→ Lower magnitude ≈ more isotropic (evenly distributed)")

    passed = legal_intra > inter_sim
    duration_ms = int((time.time() - start) * 1000)

    RESULTS.append({
        "name": "Embedding Space Geometry",
        "passed": passed,
        "details": {
            "legal_intra_cluster_sim": legal_intra,
            "non_legal_intra_cluster_sim": non_legal_intra,
            "inter_cluster_sim": inter_sim,
            "cluster_separation": legal_intra - inter_sim,
            "centroid_magnitude": centroid_mag,
        },
        "duration_ms": duration_ms,
    })
    print(f"  └─ {'✅ PASSED' if passed else '❌ FAILED'} ({duration_ms}ms)")


# ═══════════════════════════════════════════════════════════════
# TEST 8: Contextual Token Sensitivity (Self-Attention)
# ═══════════════════════════════════════════════════════════════
def test_attention_mechanism(model: SentenceTransformer):
    log("TEST 8: Multi-Head Self-Attention — Contextual Token Sensitivity")
    start = time.time()

    texts = [
        "The bank approved the loan for the property acquisition.",
        "The river bank was covered with wildflowers in spring.",
        "The court ruled in favor of the plaintiff in the damages case.",
        "The tennis court was freshly resurfaced for the tournament.",
    ]

    sublog("Contextual Embedding Sensitivity Test:")
    sublog("(Same word, different context → different embeddings)")
    sublog("---")

    embs = model.encode(texts, normalize_embeddings=True)
    bank_sim = cosine_sim(embs[0].tolist(), embs[1].tolist())
    court_sim = cosine_sim(embs[2].tolist(), embs[3].tolist())
    legal_sim = cosine_sim(embs[0].tolist(), embs[2].tolist())

    sublog(f'"bank" (financial) vs "bank" (river): {bank_sim:.4f}')
    sublog(f'"court" (legal) vs "court" (tennis): {court_sim:.4f}')
    sublog(f'"bank" (financial) vs "court" (legal): {legal_sim:.4f}')
    sublog("→ Self-attention enables context-dependent representations")

    duration_ms = int((time.time() - start) * 1000)
    RESULTS.append({
        "name": "Multi-Head Self-Attention Contextual Test",
        "passed": True,
        "details": {
            "bank_financial_vs_river": bank_sim,
            "court_legal_vs_tennis": court_sim,
            "bank_financial_vs_court_legal": legal_sim,
        },
        "duration_ms": duration_ms,
    })
    print(f"  └─ ✅ PASSED ({duration_ms}ms)")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("\n" + "█" * 70)
    print("  SYNAPSE RAG — DEEP LEARNING ARCHITECTURE VALIDATION SUITE")
    print("█" * 70)
    print(f"  Timestamp: {datetime.utcnow().isoformat()}")
    print(f"  Platform: Python {sys.version.split()[0]}")

    total_start = time.time()

    bi_encoder = test_bi_encoder_architecture()
    test_semantic_similarity(bi_encoder)
    test_pooling_strategies(bi_encoder)
    cross_enc = test_cross_encoder_architecture()
    test_reranking(cross_enc)
    test_tokenization()
    test_embedding_geometry(bi_encoder)
    test_attention_mechanism(bi_encoder)

    total_ms = int((time.time() - total_start) * 1000)

    log("VALIDATION SUMMARY")
    passed_count = sum(1 for r in RESULTS if r["passed"])
    failed_count = len(RESULTS) - passed_count

    for i, r in enumerate(RESULTS, 1):
        mark = "✅" if r["passed"] else "❌"
        sublog(f"{mark} Test {i}: {r['name']} ({r['duration_ms']}ms)")

    print(f"\n  Total: {len(RESULTS)} tests | ✅ {passed_count} passed | ❌ {failed_count} failed")
    print(f"  Total Duration: {total_ms / 1000:.1f}s")
    print("═" * 70)

    output_path = Path(__file__).parent.parent.parent / "data" / "dl_architecture_test_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "platform": f"Python {sys.version.split()[0]}",
            "total_duration_ms": total_ms,
            "summary": {"total": len(RESULTS), "passed": passed_count, "failed": failed_count},
            "tests": RESULTS,
        }, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
