"""
苏大姐 v7.1 — A-MAC Gate 离线效果模拟
验证 Novelty 过滤对索引质量的影响

思路:
1. 从 LoCoMo 对话中提取所有 observation (作为"应该记住的知识点")
2. 模拟 gate 准入过程, 统计过滤效果
3. 衡量: 过滤后保留的记忆是否覆盖了问题所需的事实
"""
import json
import re
import time
import numpy as np
from collections import Counter

# Load LoCoMo data
with open('locomo10.json') as f:
    data = json.load(f)

print("=" * 60)
print("  A-MAC Gate 离线效果模拟")
print("  基于 LoCoMo 10-conversation Dataset")
print("=" * 60)

# === Simplified A-MAC Gate ===
def evaluate_utility(content):
    patterns = [
        (r"(\w+)'s\s+(name|job|wife|husband|pet|hobby|favorite|age|school|girlfriend|boyfriend)", 0.9),
        (r"(名字|女朋友|男朋友|老婆|老公|工作|公司|学校|年龄)", 0.9),
        (r"(moved|started|works|lives|born|graduated)", 0.85),
        (r"(likes?|loves?|hates?|prefers?|always|never)", 0.8),
        (r"(will|plan|going to|next week|tomorrow)", 0.75),
    ]
    for p, s in patterns:
        if re.search(p, content, re.IGNORECASE):
            return s
    if len(content) < 10:
        return 0.2
    if len(content) > 50:
        return 0.6
    return 0.5

def evaluate_type_prior(content):
    if re.search(r"(name|wife|husband|friend|daughter|son|pet|girlfriend|boyfriend)", content, re.I):
        return 0.95
    if re.search(r"(likes?|loves?|hates?|prefers?|favorite|habit)", content, re.I):
        return 0.85
    if re.search(r"(works?|lives?|born|graduated|studies)", content, re.I):
        return 0.9
    if re.search(r"(plan|will|going|want|intend)", content, re.I):
        return 0.75
    if re.search(r"(feel|mood|happy|sad|stress|excit)", content, re.I):
        return 0.5
    return 0.4

def should_admit(content, source="conversation"):
    """Simplified admission decision"""
    utility = evaluate_utility(content)
    type_prior = evaluate_type_prior(content)
    
    # Quick rejects
    if utility < 0.3 and type_prior < 0.75:
        return False, "TRIVIAL"
    if len(content.strip()) < 5:
        return False, "TOO_SHORT"
    
    # Weighted score (no embedding-based novelty in offline mode)
    score = 0.30 * utility + 0.20 * 0.7 + 0.25 * 0.8 + 0.10 * 1.0 + 0.15 * type_prior
    if score < 0.45:
        return False, "BELOW_THRESHOLD"
    return True, None

# === Process each conversation ===
total_observations = 0
admitted_observations = 0
rejected_by_reason = Counter()

# Category analysis
cat_observations = Counter()  # by QA category
useful_observations = 0

for conv_idx, conv in enumerate(data):
    obs = conv.get('observation', {})
    qa_list = conv.get('qa', [])
    
    conv_admitted = 0
    conv_total = 0
    
    for session_key, obs_items in obs.items():
        if isinstance(obs_items, list):
            for item in obs_items:
                if isinstance(item, list):
                    text = item[0] if len(item) > 0 else ""
                elif isinstance(item, str):
                    text = item
                else:
                    continue
                
                conv_total += 1
                total_observations += 1
                
                admitted, reason = should_admit(text)
                if admitted:
                    admitted_observations += 1
                    conv_admitted += 1
                else:
                    rejected_by_reason[reason] += 1
    
    # Check QA coverage
    conv_questions = len(qa_list)
    
    print(f"  Conv {conv_idx+1}: {conv_admitted}/{conv_total} observations admitted "
          f"({conv_admitted/conv_total*100:.0f}%), {conv_questions} QA pairs")

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  Total observations: {total_observations}")
print(f"  Admitted: {admitted_observations} ({admitted_observations/total_observations*100:.1f}%)")
print(f"  Rejected: {total_observations - admitted_observations} ({(total_observations-admitted_observations)/total_observations*100:.1f}%)")
print(f"\n  Rejection breakdown:")
for reason, count in rejected_by_reason.most_common():
    print(f"    {reason}: {count} ({count/total_observations*100:.1f}%)")

# === Analyze what was rejected ===
print(f"\n{'='*60}")
print(f"  SAMPLE REJECTED OBSERVATIONS")
print(f"{'='*60}")
rejected_samples = []
for conv in data[:3]:
    obs = conv.get('observation', {})
    for session_key, obs_items in obs.items():
        if isinstance(obs_items, list):
            for item in obs_items:
                text = item[0] if isinstance(item, list) and len(item) > 0 else (item if isinstance(item, str) else "")
                admitted, reason = should_admit(text)
                if not admitted and len(rejected_samples) < 5:
                    rejected_samples.append((text[:80], reason))

for text, reason in rejected_samples:
    print(f"  [{reason}] {text}")

# === Value assessment ===
print(f"\n{'='*60}")
print(f"  QUALITY ASSESSMENT")
print(f"{'='*60}")
# Sample high-utility admitted observations
high_value = []
for conv in data[:3]:
    obs = conv.get('observation', {})
    for session_key, obs_items in obs.items():
        if isinstance(obs_items, list):
            for item in obs_items:
                text = item[0] if isinstance(item, list) and len(item) > 0 else (item if isinstance(item, str) else "")
                if should_admit(text)[0] and evaluate_utility(text) >= 0.8:
                    if len(high_value) < 8:
                        high_value.append(text[:80])

print("  High-value admitted (utility >= 0.8):")
for t in high_value:
    print(f"    ✓ {t}")

# === Estimate impact on eval ===
print(f"\n{'='*60}")
print(f"  ESTIMATED IMPACT ON EVAL")
print(f"{'='*60}")
filter_rate = (total_observations - admitted_observations) / total_observations
print(f"  Index size reduction: {filter_rate*100:.1f}%")
print(f"  Expected benefits:")
print(f"    - Reduced noise in retrieval context")
print(f"    - Higher precision @ top-k (fewer irrelevant results)")
print(f"    - Lower embedding compute cost ({filter_rate*100:.0f}% fewer embeddings)")
print(f"  Risks:")
print(f"    - False negatives (useful info incorrectly filtered)")
print(f"    - Need to verify against QA gold answers")
