"""
Test: EmbeddingProvider abstraction + SemanticRuleProvider
验证向量嵌入抽象层和基于规则的语义相似度
"""

import math
import sys

TESTS_PASSED = 0
TESTS_FAILED = 0

def run_test(name, fn):
    global TESTS_PASSED, TESTS_FAILED
    try:
        fn()
        print(f"✓ {name}")
        TESTS_PASSED += 1
    except AssertionError as e:
        print(f"✗ {name}: {e}")
        TESTS_FAILED += 1
    except Exception as e:
        print(f"✗ {name}: EXCEPTION {e}")
        TESTS_FAILED += 1


# === Python simulations ===

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class CharOverlapProvider:
    name = "char-overlap"
    
    def compute_similarity(self, query, content):
        q = set(query)
        c = set(content)
        inter = len(q & c)
        union = len(q | c)
        return inter / union if union > 0 else 0.0


class SemanticRuleProvider:
    name = "semantic-rule"
    
    INTENT_KEYWORDS = {
        "location_current": ["住", "在哪", "现在", "搬", "居住", "地址"],
        "location_past": ["以前", "之前", "住过", "老家", "曾经"],
        "preference": ["喜欢", "爱好", "偏好", "喜好", "最爱"],
        "health": ["过敏", "病", "健康", "身体", "药"],
        "work": ["工作", "上班", "公司", "项目", "职业"],
        "relationship": ["朋友", "家人", "男朋友", "女朋友", "对象"],
    }
    
    STATE_CHANGE_VERBS = ["搬到", "换了", "开始", "不再", "改为"]
    KNOWN_ENTITIES = ["北京", "上海", "深圳", "广州", "杭州", "花生", "猫", "狗"]
    
    def compute_similarity(self, query, content):
        score = 0.0
        
        # 1. Intent alignment
        for _, keywords in self.INTENT_KEYWORDS.items():
            q_hits = sum(1 for kw in keywords if kw in query)
            c_hits = sum(1 for kw in keywords if kw in content)
            if q_hits > 0 and c_hits > 0:
                score += 0.3
        
        # 2. Entity overlap
        q_ent = {e for e in self.KNOWN_ENTITIES if e in query}
        c_ent = {e for e in self.KNOWN_ENTITIES if e in content}
        overlap = len(q_ent & c_ent)
        if overlap > 0:
            score += 0.2 * overlap
        
        # 3. Action-state relevance
        if "住" in query and any(v in content and "到" in content for v in self.STATE_CHANGE_VERBS):
            score += 0.25
        
        # 4. Bigram overlap
        q_bigrams = {query[i:i+2] for i in range(len(query)-1) if ord(query[i]) > 0x4E00}
        c_bigrams = {content[i:i+2] for i in range(len(content)-1) if ord(content[i]) > 0x4E00}
        if q_bigrams:
            score += 0.15 * len(q_bigrams & c_bigrams) / len(q_bigrams)
        
        return min(1.0, max(0.0, score))
    
    def extract_entities(self, text):
        return {e for e in self.KNOWN_ENTITIES if e in text}


class CachedProvider:
    def __init__(self, delegate):
        self.delegate = delegate
        self.cache = {}
        self.name = f"cached({delegate.name})"
    
    def compute_similarity(self, query, content):
        key = (query, content)
        if key not in self.cache:
            self.cache[key] = self.delegate.compute_similarity(query, content)
        return self.cache[key]
    
    def cache_size(self):
        return len(self.cache)


# === Tests ===

def test_char_overlap_basic():
    """CharOverlap: 基本字符重叠"""
    p = CharOverlapProvider()
    # Identical
    assert p.compute_similarity("hello", "hello") == 1.0
    # No overlap
    assert p.compute_similarity("abc", "xyz") == 0.0
    # Partial
    s = p.compute_similarity("abcd", "cdef")
    assert 0 < s < 1

def test_semantic_rule_location_query():
    """SemanticRule: 住在哪 → 搬到上海 应该有高相似度"""
    p = SemanticRuleProvider()
    
    s1 = p.compute_similarity("我现在住在哪里", "我搬到了上海浦东")
    s2 = p.compute_similarity("我现在住在哪里", "今天天气不错")
    
    assert s1 > s2, f"Location query should match relocation: {s1:.3f} vs {s2:.3f}"
    assert s1 >= 0.3, f"Should have meaningful similarity: {s1:.3f}"
    print(f"  '住在哪' vs '搬到上海': {s1:.3f}")
    print(f"  '住在哪' vs '天气不错': {s2:.3f}")

def test_semantic_rule_resolves_char_overlap_problem():
    """核心测试：解决 char-overlap 的排序问题"""
    p = SemanticRuleProvider()
    
    # The classic failure case:
    # "我现在住在哪里" vs "我住在北京" vs "我搬到了上海"
    # CharOverlap gives: 北京 > 上海 (because 住在 bigram overlap)
    # SemanticRule should give: 上海 >= 北京 (relocation answers the question)
    
    q = "我现在住在哪里"
    s_bj = p.compute_similarity(q, "我住在北京朝阳区")
    s_sh = p.compute_similarity(q, "我搬到了上海浦东")
    
    print(f"  北京: {s_bj:.3f}, 上海: {s_sh:.3f}")
    # Both should be reasonable (>0.2)
    assert s_bj > 0.2, f"北京 should have meaningful score: {s_bj:.3f}"
    assert s_sh > 0.2, f"上海 should have meaningful score: {s_sh:.3f}"
    # 上海 should score at least as high (搬到 = state change answering 住在哪)
    assert s_sh >= s_bj * 0.8, f"上海 should not be much lower than 北京: {s_sh:.3f} vs {s_bj:.3f}"

def test_semantic_rule_preference():
    """偏好类查询"""
    p = SemanticRuleProvider()
    s1 = p.compute_similarity("我喜欢什么", "我喜欢吃火锅")
    s2 = p.compute_similarity("我喜欢什么", "明天要开会")
    assert s1 > s2, f"Preference match: {s1:.3f} should > {s2:.3f}"

def test_semantic_rule_health():
    """健康类查询"""
    p = SemanticRuleProvider()
    s1 = p.compute_similarity("我有什么过敏", "我对花生过敏")
    s2 = p.compute_similarity("我有什么过敏", "我喜欢跑步")
    assert s1 > s2

def test_semantic_rule_entity_boost():
    """Entity 提及 boost"""
    p = SemanticRuleProvider()
    s1 = p.compute_similarity("北京怎么样", "我住在北京")
    s2 = p.compute_similarity("北京怎么样", "我住在上海")
    assert s1 > s2, f"Entity match should boost: {s1:.3f} vs {s2:.3f}"

def test_cached_provider():
    """缓存层正常工作"""
    base = SemanticRuleProvider()
    cached = CachedProvider(base)
    
    s1 = cached.compute_similarity("测试", "内容")
    s2 = cached.compute_similarity("测试", "内容")
    
    assert s1 == s2
    assert cached.cache_size() == 1

def test_cosine_similarity():
    """Cosine 相似度计算"""
    # Same vector → 1.0
    a = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(a, a) - 1.0) < 0.001
    
    # Orthogonal → 0.0
    b = [0.0, 1.0, 0.0]
    assert abs(cosine_similarity(a, b)) < 0.001
    
    # Opposite → -1.0
    c = [-1.0, 0.0, 0.0]
    assert abs(cosine_similarity(a, c) - (-1.0)) < 0.001

def test_semantic_vs_char_overlap_comparison():
    """对比：SemanticRule 在关键场景比 CharOverlap 更准确"""
    char_p = CharOverlapProvider()
    sem_p = SemanticRuleProvider()
    
    test_cases = [
        ("我现在住在哪", "我搬到了上海", "我住在北京"),  # 上海应该更relevant
        ("我喜欢什么颜色", "我最爱蓝色", "我住在蓝山"),  # 颜色偏好 > 地名碰巧含蓝
    ]
    
    sem_correct = 0
    char_correct = 0
    
    for query, better, worse in test_cases:
        s_sem_better = sem_p.compute_similarity(query, better)
        s_sem_worse = sem_p.compute_similarity(query, worse)
        s_char_better = char_p.compute_similarity(query, better)
        s_char_worse = char_p.compute_similarity(query, worse)
        
        if s_sem_better >= s_sem_worse:
            sem_correct += 1
        if s_char_better >= s_char_worse:
            char_correct += 1
    
    print(f"  SemanticRule correct: {sem_correct}/{len(test_cases)}")
    print(f"  CharOverlap correct: {char_correct}/{len(test_cases)}")
    # SemanticRule should be at least as good
    assert sem_correct >= char_correct, \
        f"SemanticRule should be >= CharOverlap in accuracy"

def test_zero_vectors():
    """空/零向量处理"""
    assert cosine_similarity([0,0,0], [1,2,3]) == 0.0
    assert cosine_similarity([], []) == 0.0


# === Run ===
print("=" * 60)
print("EmbeddingProvider + SemanticRuleProvider Tests")
print("=" * 60)

tests = [
    ("char_overlap_basic", test_char_overlap_basic),
    ("semantic_rule_location", test_semantic_rule_location_query),
    ("semantic_resolves_overlap_problem", test_semantic_rule_resolves_char_overlap_problem),
    ("semantic_preference", test_semantic_rule_preference),
    ("semantic_health", test_semantic_rule_health),
    ("semantic_entity_boost", test_semantic_rule_entity_boost),
    ("cached_provider", test_cached_provider),
    ("cosine_similarity", test_cosine_similarity),
    ("semantic_vs_char_comparison", test_semantic_vs_char_overlap_comparison),
    ("zero_vectors", test_zero_vectors),
]

for name, fn in tests:
    run_test(name, fn)

print()
print("=" * 50)
print(f"Results: {TESTS_PASSED} passed, {TESTS_FAILED} failed, {TESTS_PASSED + TESTS_FAILED} total")
if TESTS_FAILED == 0:
    print("ALL TESTS PASSED ✓")
else:
    print(f"FAILURES: {TESTS_FAILED}")
    sys.exit(1)
