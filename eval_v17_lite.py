#!/usr/bin/env python3
"""
LoCoMo Eval v17-lite — Quick validation on targeted categories

Runs on first 3 conversations, focusing on categories where 
Governance + Provenance should make the biggest difference:
- Category 2 (temporal): Provenance timestamps help
- Category 5 (adversarial): Safety filter helps
- Category 1 (single-hop factual): Statutory boost helps

Skips category 4 (open-domain, 841 questions) to save API cost.
"""
import json, re, math, time, sys, urllib.request, ssl
import numpy as np
from collections import Counter, defaultdict
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple

DEEPSEEK_KEY = "sk-b2278aa31f7144d0acb0943a40d0e62e"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
ZHIPU_KEY = "7b50a657c52e4c3fbb3c518ac206b483.AZTYQhOxC5IX6C9T"
ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE
stats = {'ds': 0, 'zp': 0, 'blocked': 0, 'contradictions_resolved': 0}

# --- Governance + Provenance ---
class GovernanceLayer(Enum):
    CONSTITUTIONAL = 3
    STATUTORY = 2
    OPERATIONAL = 1

class TrustLevel(Enum):
    HIGH = 0.95
    MEDIUM_HIGH = 0.8
    MEDIUM = 0.65
    MEDIUM_LOW = 0.5
    LOW = 0.3

FACTUAL_PATTERNS = [
    re.compile(r"(?:name|called|named)\b", re.IGNORECASE),
    re.compile(r"(?:work|job|employed|career|occupation)", re.IGNORECASE),
    re.compile(r"(?:wife|husband|girlfriend|boyfriend|partner|married|dating|relationship)", re.IGNORECASE),
    re.compile(r"(?:live|lives in|moved to|address|located)", re.IGNORECASE),
    re.compile(r"(?:likes?|loves?|hates?|favorite|prefer|hobby|hobbies)", re.IGNORECASE),
    re.compile(r"(?:pet|cat|dog|animal)", re.IGNORECASE),
    re.compile(r"(?:born|age|birthday|years old)", re.IGNORECASE),
    re.compile(r"(?:school|university|college|graduated|study|degree)", re.IGNORECASE),
    re.compile(r"(?:brother|sister|mother|father|parent|family|son|daughter)", re.IGNORECASE),
]

INJECTION_PATTERNS = [
    re.compile(r"(?i)(ignore|disregard|forget)\s+.{0,20}(previous|above|prior|all)\s+.{0,10}(instructions?|rules?|prompts?)"),
    re.compile(r"(?i)you\s+are\s+now\s+a?\s*(different|new|evil|unrestricted)"),
    re.compile(r"(?i)\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>"),
]

@dataclass
class MemoryNode:
    content: str
    session_id: int
    layer: GovernanceLayer
    trust: TrustLevel
    source_type: str

def classify_layer(content: str) -> GovernanceLayer:
    if any(p.search(content) for p in FACTUAL_PATTERNS):
        return GovernanceLayer.STATUTORY
    return GovernanceLayer.OPERATIONAL

def safety_check(content: str) -> bool:
    for p in INJECTION_PATTERNS:
        if p.search(content):
            stats['blocked'] += 1
            return False
    return True

# --- API ---
def embed_batch(texts):
    results = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
        payload = json.dumps({"model": "embedding-3", "input": batch}).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ZHIPU_KEY}"}
        for attempt in range(3):
            try:
                req = urllib.request.Request(ZHIPU_URL, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as resp:
                    data = json.loads(resp.read())
                    embs = [d['embedding'] for d in sorted(data['data'], key=lambda x: x['index'])]
                    results.extend(embs)
                    stats['zp'] += 1
                    break
            except Exception as e:
                if attempt < 2: time.sleep(2*(attempt+1))
                else: results.extend([[0.0]*2048]*len(batch))
        time.sleep(0.03)
    return np.array(results, dtype=np.float32)

def call_llm(messages, temp=0.1, max_tok=200):
    stats['ds'] += 1
    payload = json.dumps({"model": "deepseek-chat", "messages": messages,
                          "temperature": temp, "max_tokens": max_tok, "stream": False}).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY}"}
    for attempt in range(3):
        try:
            req = urllib.request.Request(DEEPSEEK_URL, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=60, context=ssl_ctx) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 2: time.sleep(3*(attempt+1))
    return ""

# --- Index ---
def build_index(conv_data) -> List[MemoryNode]:
    conversation = conv_data['conversation']
    observations = conv_data.get('observation', {})
    event_summaries = conv_data.get('event_summary', {})
    session_summaries = conv_data.get('session_summary', {})
    
    nodes = []
    session_dates = {}
    for key in conversation.keys():
        if '_date_time' in key:
            sn = int(key.replace('session_', '').replace('_date_time', ''))
            session_dates[sn] = conversation[key]
    
    for key, val in observations.items():
        if not isinstance(val, dict): continue
        m = re.search(r'session_(\d+)', key)
        if not m: continue
        sn = int(m.group(1))
        date_str = session_dates.get(sn, "")
        for spk, obs_list in val.items():
            if not isinstance(obs_list, list): continue
            for obs_item in obs_list:
                if isinstance(obs_item, list):
                    obs = obs_item[0] if obs_item else ""
                else:
                    obs = obs_item
                if not safety_check(obs): continue
                content = f"[Session {sn}, {date_str}] {spk}: {obs}"
                nodes.append(MemoryNode(content, sn, classify_layer(obs), TrustLevel.HIGH, "observation"))
    
    for key, val in event_summaries.items():
        if not isinstance(val, list): continue
        m = re.search(r'session_(\d+)', key)
        if not m: continue
        sn = int(m.group(1))
        date_str = session_dates.get(sn, "")
        for ev_item in val:
            if isinstance(ev_item, list):
                ev = str(ev_item[0]) if ev_item else ""
            else:
                ev = str(ev_item)
            if not ev or not safety_check(ev): continue
            content = f"[Session {sn}, {date_str}] Event: {ev}"
            nodes.append(MemoryNode(content, sn, classify_layer(ev), TrustLevel.MEDIUM_HIGH, "event_summary"))
    
    for key, val in session_summaries.items():
        if not isinstance(val, str): continue
        m = re.search(r'session_(\d+)', key)
        if not m: continue
        sn = int(m.group(1))
        date_str = session_dates.get(sn, "")
        if not safety_check(val): continue
        content = f"[Session {sn}, {date_str}] Summary: {val}"
        nodes.append(MemoryNode(content, sn, classify_layer(val), TrustLevel.MEDIUM, "session_summary"))
    
    return nodes

# --- Retrieval with governance boost ---
def retrieve(query, nodes, node_embeddings, top_k=12):
    if not nodes: return []
    q_emb = embed_batch([query])
    if q_emb.shape[0] == 0: return nodes[:top_k]
    
    norms_d = np.linalg.norm(node_embeddings, axis=1, keepdims=True)
    norms_d = np.where(norms_d == 0, 1, norms_d)
    norm_q = np.linalg.norm(q_emb, axis=1, keepdims=True)
    norm_q = np.where(norm_q == 0, 1, norm_q)
    sims = ((node_embeddings / norms_d) @ (q_emb / norm_q).T).flatten()
    
    is_factual = any(p.search(query) for p in FACTUAL_PATTERNS)
    
    scored = []
    for i, node in enumerate(nodes):
        score = float(sims[i])
        if is_factual and node.layer == GovernanceLayer.STATUTORY:
            score += 0.04
        score += node.trust.value * 0.015
        scored.append((score, i, node))
    
    scored.sort(key=lambda x: -x[0])
    
    # Contradiction resolution
    result = []
    seen_topics = {}
    for score, idx, node in scored[:top_k * 2]:
        topic = extract_topic(node.content)
        if topic and topic in seen_topics:
            existing = seen_topics[topic]
            if node.trust.value > existing.trust.value or \
               (node.trust.value == existing.trust.value and node.session_id > existing.session_id):
                result = [n for n in result if n is not existing]
                result.append(node)
                seen_topics[topic] = node
                stats['contradictions_resolved'] += 1
        else:
            if topic:
                seen_topics[topic] = node
            result.append(node)
        if len(result) >= top_k:
            break
    
    return result[:top_k]

def extract_topic(content: str) -> Optional[str]:
    patterns = [
        re.compile(r"(\w+)'s\s+(name|job|wife|husband|pet|hobby|favorite|age|school)", re.IGNORECASE),
        re.compile(r"(\w+)\s+(?:works?|lives?|likes?|moved?|started?)\s+(?:at|in|to)\s+(\w+)", re.IGNORECASE),
    ]
    for p in patterns:
        m = p.search(content)
        if m: return m.group(0).lower()[:30]
    return None

# --- Answer + Judge ---
def answer_and_judge(question, gold, retrieved_nodes, category):
    context_parts = []
    for node in retrieved_nodes:
        tag = "[✓] " if node.trust == TrustLevel.HIGH else ""
        context_parts.append(f"{tag}{node.content}")
    context = "\n".join(context_parts)
    
    sys_msg = """Answer questions based on memory context. Use ONLY provided context.
If info conflicts, prefer entries marked [✓] (highly reliable, direct from conversation).
For temporal questions, pay attention to session dates.
If uncertain, say "I don't know". Be concise."""
    
    user_msg = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    answer = call_llm([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}])
    
    if not answer or answer.lower().strip() in ["i don't know", "not mentioned", "unknown"]:
        return answer, False
    
    judge_msg = f"""Q: {question}
Gold: {gold}
Predicted: {answer}
Is predicted factually equivalent to gold? (temporal: ±1 day ok, partial: ok if key fact matches)
Reply ONLY "correct" or "incorrect"."""
    
    judgment = call_llm([{"role": "user", "content": judge_msg}], temp=0.0, max_tok=10)
    return answer, "correct" in judgment.lower()

# --- Main ---
def main():
    print("=" * 60)
    print("  LoCoMo Eval v17-lite (Governance + Provenance)")
    print("  Targeted: Cat 1 (factual) + Cat 2 (temporal) + Cat 5 (adversarial)")
    print("  Scope: First 3 conversations")
    print("=" * 60)
    
    with open("locomo10.json", "r") as f:
        dataset = json.load(f)
    
    # Run on first 3 conversations, targeted categories
    TARGET_CATS = {1, 2, 5}
    MAX_CONVS = 3
    
    all_results = []
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for conv_idx in range(min(MAX_CONVS, len(dataset))):
        conv_data = dataset[conv_idx]
        print(f"\n--- Conv {conv_idx+1}/{MAX_CONVS} ---")
        
        nodes = build_index(conv_data)
        layer_counts = Counter(n.layer.name for n in nodes)
        print(f"  Nodes: {len(nodes)} | Layers: {dict(layer_counts)}")
        
        node_texts = [n.content for n in nodes]
        if not node_texts: continue
        node_embs = embed_batch(node_texts)
        
        questions = conv_data.get('qa', [])
        target_qs = [q for q in questions if isinstance(q, dict) and q.get('category') in TARGET_CATS]
        print(f"  Target questions: {len(target_qs)}")
        
        for qi, q_data in enumerate(target_qs):
            question = q_data['question']
            gold = q_data.get('answer', q_data.get('answers', [''])[0] if isinstance(q_data.get('answers'), list) else q_data.get('answers', ''))
            category = q_data['category']
            
            retrieved = retrieve(question, nodes, node_embs, top_k=12)
            answer, correct = answer_and_judge(question, gold, retrieved, category)
            
            all_results.append({
                "question": question, "gold": gold, "predicted": answer,
                "correct": correct, "category": category
            })
            cat_stats[category]["total"] += 1
            if correct: cat_stats[category]["correct"] += 1
            
            if (qi + 1) % 10 == 0:
                tot = sum(c["total"] for c in cat_stats.values())
                cor = sum(c["correct"] for c in cat_stats.values())
                print(f"    [{qi+1}/{len(target_qs)}] Running acc: {cor}/{tot} = {cor/tot*100:.1f}%")
    
    # Final
    print("\n" + "=" * 60)
    print("  RESULTS — v17-lite")
    print("=" * 60)
    
    total = len(all_results)
    correct = sum(1 for r in all_results if r["correct"])
    print(f"\n  Overall: {correct}/{total} = {correct/total*100:.1f}%" if total > 0 else "  No results")
    
    cat_names = {1: "single-hop", 2: "temporal", 5: "adversarial"}
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        acc = s["correct"]/s["total"]*100 if s["total"] > 0 else 0
        print(f"    Cat {cat} ({cat_names.get(cat, '?'):12s}): {s['correct']:3d}/{s['total']:3d} = {acc:.1f}%")
    
    print(f"\n  Stats: LLM={stats['ds']} | Embed={stats['zp']} | Blocked={stats['blocked']} | ContraResolved={stats['contradictions_resolved']}")
    
    with open("eval_v17_lite_results.json", "w") as f:
        json.dump({"accuracy": correct/total if total else 0, "total": total, "correct": correct,
                   "categories": dict(cat_stats), "stats": stats, "results": all_results}, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to eval_v17_lite_results.json")

if __name__ == "__main__":
    main()
