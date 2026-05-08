"""
苏大姐 v7.1 — AdaptiveMemoryGate (A-MAC) 测试
验证五维准入控制逻辑的正确性
"""
import re
import math
import time
import numpy as np

# ============================================================
# Port of AdaptiveMemoryGate logic to Python for testing
# ============================================================

class DimensionWeights:
    def __init__(self, utility=0.30, confidence=0.20, novelty=0.25, recency=0.10, type_prior=0.15):
        self.utility = utility
        self.confidence = confidence
        self.novelty = novelty
        self.recency = recency
        self.type_prior = type_prior

class AdaptiveMemoryGate:
    def __init__(self):
        self.weights = DimensionWeights()
        self.admission_threshold = 0.45
        self.redundancy_threshold = 0.85
    
    def evaluate(self, candidate, candidate_emb=None, existing_embs=None, 
                 source="conversation", timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        if existing_embs is None:
            existing_embs = []
        
        utility = self.evaluate_utility(candidate)
        confidence = self.evaluate_confidence(candidate, source)
        novelty = self.evaluate_novelty(candidate_emb, existing_embs)
        recency = self.evaluate_recency(timestamp)
        type_prior = self.evaluate_type_prior(candidate)
        
        final_score = (self.weights.utility * utility +
                      self.weights.confidence * confidence +
                      self.weights.novelty * novelty +
                      self.weights.recency * recency +
                      self.weights.type_prior * type_prior)
        
        is_redundant = novelty < 0.15
        is_trivial = utility < 0.3 and type_prior < 0.75  # Trivial only if both low utility AND low type
        is_unreliable = confidence < 0.3  # NEW: too uncertain
        admitted = not is_redundant and not is_trivial and not is_unreliable and final_score >= self.admission_threshold
        
        return {
            'admitted': admitted,
            'final_score': final_score,
            'scores': {
                'utility': utility,
                'confidence': confidence,
                'novelty': novelty,
                'recency': recency,
                'type_prior': type_prior
            },
            'rejection_reason': 'REDUNDANT' if is_redundant else ('TRIVIAL' if is_trivial else ('UNRELIABLE' if is_unreliable else ('BELOW_THRESHOLD' if not admitted else None)))
        }
    
    def evaluate_utility(self, content):
        patterns = [
            (r'(\w+)的(名字|女朋友|男朋友|老婆|老公|工作|公司|学校|年龄|生日|爱好|宠物)', 0.9),
            (r'(明天|后天|下周|下个月|今天|昨天|上周).{0,10}(要|会|打算|计划|准备)', 0.85),
            (r'(喜欢|不喜欢|讨厌|最爱|偏好|习惯).{1,20}', 0.8),
            (r'(换了|辞了|分手|离婚|搬|毕业|入职|结婚|怀孕|生了)', 0.85),
        ]
        for pattern, score in patterns:
            if re.search(pattern, content):
                return score
        if len(content) < 5:
            return 0.1   # too short, almost no info
        elif len(content) < 10:
            return 0.25  # short, likely trivial
        elif len(content) > 100:
            return 0.6
        return 0.5
    
    def evaluate_confidence(self, content, source):
        base = {'user_direct': 0.95, 'conversation': 0.7, 'ai_summary': 0.6, 'external': 0.5}
        conf = base.get(source, 0.7)
        uncertainty = ['可能', '大概', '好像', '似乎', '也许', '不确定']
        contradiction = ['但是', '不过', '也不一定', '不太确定']
        if any(w in content for w in uncertainty):
            conf -= 0.2
        if any(w in content for w in contradiction):
            conf -= 0.1
        return max(0.1, min(1.0, conf))
    
    def evaluate_novelty(self, candidate_emb, existing_embs):
        if candidate_emb is None or len(existing_embs) == 0:
            return 0.8
        max_sim = max(cosine_sim(candidate_emb, e) for e in existing_embs)
        novelty = 1.0 - max_sim
        if 0.80 <= max_sim <= self.redundancy_threshold:
            novelty *= 1.5
        return novelty
    
    def evaluate_recency(self, timestamp):
        hours_ago = (time.time() - timestamp) / 3600.0
        return max(0.1, min(1.0, math.exp(-0.01 * hours_ago)))
    
    def evaluate_type_prior(self, content):
        if re.search(r'(名字|叫|是我的|女朋友|男朋友|老婆|老公|爸|妈|儿子|女儿|朋友|同事)', content):
            return 0.95
        if re.search(r'(喜欢|讨厌|习惯|总是|从不|每天)', content):
            return 0.85
        if re.search(r'(要去|打算|计划|准备|下周|明天)', content):
            return 0.75
        if re.search(r'(在.*工作|住在|毕业于|学的是)', content):
            return 0.9
        if re.search(r'(开心|难过|焦虑|压力|兴奋)', content):
            return 0.5
        return 0.4

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0

# ============================================================
# Tests
# ============================================================

def test_high_value_memory():
    """高价值记忆应该被准入"""
    gate = AdaptiveMemoryGate()
    
    cases = [
        ("小明的女朋友叫小红", "user_direct", True, "身份关系信息"),
        ("下周一要面试", "user_direct", True, "时间+事件"),
        ("喜欢喝美式咖啡", "user_direct", True, "偏好信息"),
        ("最近换了工作到字节", "user_direct", True, "重要状态变化"),
        ("住在北京朝阳区", "user_direct", True, "事实信息"),
    ]
    
    passed = 0
    for content, source, expected, desc in cases:
        result = gate.evaluate(content, source=source)
        ok = result['admitted'] == expected
        status = "✅" if ok else "❌"
        print(f"  {status} {desc}: admitted={result['admitted']}, score={result['final_score']:.3f}")
        if ok: passed += 1
    return passed, len(cases)

def test_low_value_memory():
    """低价值记忆应该被拒绝或边缘"""
    gate = AdaptiveMemoryGate()
    # 用低来源信任 + 不确定性词 来制造低分
    cases = [
        ("嗯", "ai_summary", False, "过短无信息"),
        ("好的", "ai_summary", False, "过短无信息2"),
        ("可能大概也许吧", "external", False, "充满不确定性"),
    ]
    
    passed = 0
    for content, source, expected, desc in cases:
        result = gate.evaluate(content, source=source)
        ok = result['admitted'] == expected
        status = "✅" if ok else "❌"
        print(f"  {status} {desc}: admitted={result['admitted']}, score={result['final_score']:.3f}, reason={result['rejection_reason']}")
        if ok: passed += 1
    return passed, len(cases)

def test_redundancy_detection():
    """高度相似的记忆应该被标记为冗余"""
    gate = AdaptiveMemoryGate()
    
    # 模拟: 已有记忆的 embedding
    existing = [np.array([0.8, 0.2, 0.1, 0.5, 0.3])]
    
    # 几乎相同的 embedding (cosine > 0.85)
    redundant_emb = np.array([0.81, 0.19, 0.11, 0.49, 0.31])
    result1 = gate.evaluate("小明的女朋友叫小红", candidate_emb=redundant_emb, 
                           existing_embs=existing, source="user_direct")
    
    # 完全不同的 embedding
    novel_emb = np.array([-0.5, 0.9, -0.3, 0.1, 0.8])
    result2 = gate.evaluate("小明的女朋友叫小红", candidate_emb=novel_emb,
                           existing_embs=existing, source="user_direct")
    
    passed = 0
    
    # Case 1: redundant should be rejected
    sim1 = cosine_sim(redundant_emb, existing[0])
    ok1 = result1['scores']['novelty'] < 0.15
    status1 = "✅" if ok1 else "❌"
    print(f"  {status1} Redundant (sim={sim1:.3f}): novelty={result1['scores']['novelty']:.3f}, admitted={result1['admitted']}")
    if ok1: passed += 1
    
    # Case 2: novel should pass
    sim2 = cosine_sim(novel_emb, existing[0])
    ok2 = result2['admitted'] == True
    status2 = "✅" if ok2 else "❌"
    print(f"  {status2} Novel (sim={sim2:.3f}): novelty={result2['scores']['novelty']:.3f}, admitted={result2['admitted']}")
    if ok2: passed += 1
    
    return passed, 2

def test_confidence_penalty():
    """不确定性和矛盾词应该降低 confidence"""
    gate = AdaptiveMemoryGate()
    
    c1 = gate.evaluate_confidence("小明在字节工作", "user_direct")
    c2 = gate.evaluate_confidence("小明可能在字节工作", "user_direct")
    c3 = gate.evaluate_confidence("小明好像在字节工作，但是也不一定", "user_direct")
    
    passed = 0
    ok1 = c1 > c2 > c3
    status = "✅" if ok1 else "❌"
    print(f"  {status} Confidence ordering: certain({c1:.2f}) > uncertain({c2:.2f}) > contradictory({c3:.2f})")
    if ok1: passed += 1
    
    # 来源影响
    c_user = gate.evaluate_confidence("今天天气不错", "user_direct")
    c_ai = gate.evaluate_confidence("今天天气不错", "ai_summary")
    c_ext = gate.evaluate_confidence("今天天气不错", "external")
    ok2 = c_user > c_ai > c_ext
    status2 = "✅" if ok2 else "❌"
    print(f"  {status2} Source ordering: user({c_user:.2f}) > ai({c_ai:.2f}) > external({c_ext:.2f})")
    if ok2: passed += 1
    
    return passed, 2

def test_type_prior():
    """不同类型内容应有不同先验权重"""
    gate = AdaptiveMemoryGate()
    
    cases = [
        ("我女朋友叫小红", 0.9, "身份关系"),
        ("我每天都喝咖啡", 0.8, "习惯偏好"),
        ("明天打算去跑步", 0.7, "计划"),
        ("在腾讯工作", 0.8, "事实"),
        ("今天好开心", 0.4, "情绪"),
        ("嗯嗯好的", 0.3, "闲聊"),
    ]
    
    passed = 0
    for content, min_expected, desc in cases:
        score = gate.evaluate_type_prior(content)
        ok = score >= min_expected
        status = "✅" if ok else "❌"
        print(f"  {status} {desc}: type_prior={score:.2f} (expected >= {min_expected})")
        if ok: passed += 1
    return passed, len(cases)

def test_recency():
    """新信息应该有更高 recency 分数"""
    gate = AdaptiveMemoryGate()
    
    now = time.time()
    score_now = gate.evaluate_recency(now)
    score_1h = gate.evaluate_recency(now - 3600)
    score_1d = gate.evaluate_recency(now - 86400)
    score_1w = gate.evaluate_recency(now - 604800)
    
    passed = 0
    ok = score_now > score_1h > score_1d > score_1w
    status = "✅" if ok else "❌"
    print(f"  {status} Recency decay: now({score_now:.3f}) > 1h({score_1h:.3f}) > 1d({score_1d:.3f}) > 1w({score_1w:.3f})")
    if ok: passed += 1
    
    # 即使一周前的也不应该降到太低
    ok2 = score_1w > 0.1
    status2 = "✅" if ok2 else "❌"
    print(f"  {status2} 1-week recency not too low: {score_1w:.3f} > 0.1")
    if ok2: passed += 1
    
    return passed, 2

def test_integrated_scenario():
    """完整场景: 模拟多轮对话中的准入决策"""
    gate = AdaptiveMemoryGate()
    
    # 模拟已有记忆
    existing_embs = [
        np.random.randn(128).astype(np.float32),  # 已有记忆1
        np.random.randn(128).astype(np.float32),  # 已有记忆2
    ]
    
    print("  --- Integrated Scenario ---")
    memories = [
        ("我叫小明，今年25岁", "user_direct", True),
        ("我女朋友叫小红，在一起3年了", "user_direct", True),
        ("好的", "conversation", False),
        ("可能大概也许是吧", "external", False),
        ("明天下午3点要开项目评审会", "user_direct", True),
    ]
    
    passed = 0
    for content, source, expected in memories:
        # 随机 embedding (不与 existing 重复)
        emb = np.random.randn(128).astype(np.float32)
        result = gate.evaluate(content, candidate_emb=emb, existing_embs=existing_embs, source=source)
        ok = result['admitted'] == expected
        status = "✅" if ok else "❌"
        short_content = content[:20]
        print(f"  {status} \"{short_content}\": admitted={result['admitted']}, score={result['final_score']:.3f}")
        if ok: passed += 1
    
    return passed, len(memories)

# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  AdaptiveMemoryGate (A-MAC) Test Suite")
    print("  苏大姐 v7.1 — 五维准入控制验证")
    print("=" * 60)
    
    total_passed = 0
    total_cases = 0
    
    tests = [
        ("1. 高价值记忆准入", test_high_value_memory),
        ("2. 低价值记忆拒绝", test_low_value_memory),
        ("3. 冗余检测 (Novelty)", test_redundancy_detection),
        ("4. Confidence 降分", test_confidence_penalty),
        ("5. Type Prior 分类", test_type_prior),
        ("6. Recency 衰减", test_recency),
        ("7. 集成场景", test_integrated_scenario),
    ]
    
    for name, test_fn in tests:
        print(f"\n[{name}]")
        p, t = test_fn()
        total_passed += p
        total_cases += t
    
    print(f"\n{'=' * 60}")
    print(f"  RESULT: {total_passed}/{total_cases} passed")
    rate = total_passed / total_cases * 100
    print(f"  PASS RATE: {rate:.1f}%")
    if total_passed == total_cases:
        print("  🎉 ALL TESTS PASSED")
    else:
        print(f"  ⚠️  {total_cases - total_passed} tests failed")
    print("=" * 60)
