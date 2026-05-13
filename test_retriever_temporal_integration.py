"""
Integration test: MultiSignalRetriever + TemporalReasoner
验证 temporal reasoning 通过 retriever 管线正确生效
"""

import subprocess
import json
import sys
import time

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

# === Simulated Kotlin classes (matching MultiSignalRetriever v2 logic) ===

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class MemoryEntry:
    id: str
    content: str
    timestamp: int  # epoch millis
    entities: List[str] = field(default_factory=list)
    source: str = "USER"  # USER/AGENT/SYSTEM

@dataclass
class RetrievalCandidate:
    id: str
    content: str
    semantic_score: float
    entry: MemoryEntry

@dataclass
class SignalScores:
    semantic: float
    keyword: float
    entity: float
    recency: float
    source_weight: float

@dataclass
class RankedMemory:
    id: str
    content: str
    score: float
    signals: SignalScores
    entry: Optional[MemoryEntry] = None

# TemporalReasoner (same as test_temporal_reasoner.py)
class TemporalReasoner:
    PAST_PATTERNS = [
        (r'以前', 90), (r'之前', 30), (r'过去', 90),
        (r'曾经', 180), (r'小时候', 365*10),
        (r'(\d+)天前', None), (r'(\d+)周前', None), (r'(\d+)个?月前', None),
        (r'上周', 7), (r'上个?月', 30), (r'去年', 365),
        (r'前天', 2), (r'大前天', 3), (r'昨天', 1),
    ]
    PRESENT_PATTERNS = [
        (r'现在', 0), (r'目前', 0), (r'当前', 0),
        (r'最近', 7), (r'这几天', 3), (r'今天', 0),
        (r'这周', 3), (r'这个?月', 15),
    ]
    FUTURE_PATTERNS = [
        (r'明天', -1), (r'后天', -2), (r'下周', -7),
        (r'下个?月', -30), (r'明年', -365),
        (r'即将', -7), (r'打算', -30), (r'计划', -30),
    ]
    
    def __init__(self, half_life_days=30.0, min_score=0.1):
        self.half_life_days = half_life_days
        self.min_score = min_score
    
    def analyze(self, query, reference_time):
        anchors = []
        DAY_MS = 86400000
        
        for pattern, default_days in self.PAST_PATTERNS:
            m = re.search(pattern, query)
            if m:
                if default_days is None:
                    num = int(m.group(1))
                    if '周' in pattern: num *= 7
                    elif '月' in pattern: num *= 30
                    days = num
                else:
                    days = default_days
                anchor_time = reference_time - days * DAY_MS
                anchors.append(('PAST', anchor_time, days))
        
        for pattern, default_days in self.PRESENT_PATTERNS:
            m = re.search(pattern, query)
            if m:
                anchor_time = reference_time - default_days * DAY_MS
                anchors.append(('PRESENT', anchor_time, default_days))
        
        for pattern, neg_days in self.FUTURE_PATTERNS:
            m = re.search(pattern, query)
            if m:
                days = abs(neg_days)
                anchor_time = reference_time + days * DAY_MS
                anchors.append(('FUTURE', anchor_time, days))
        
        if not anchors:
            return {'direction': 'NONE', 'anchor': reference_time, 'anchors': []}
        
        # Primary = first detected
        primary = anchors[0]
        return {
            'direction': primary[0],
            'anchor': primary[1],
            'anchors': anchors
        }
    
    def compute_temporal_score(self, memory_timestamp, temporal_context):
        anchor = temporal_context['anchor']
        direction = temporal_context['direction']
        
        distance_ms = abs(memory_timestamp - anchor)
        distance_days = distance_ms / 86400000.0
        
        decay = math.exp(-0.693 * distance_days / self.half_life_days)
        score = max(self.min_score, decay)
        
        # Direction bonus
        if direction == 'PAST' and memory_timestamp < anchor:
            score = min(1.0, score + 0.1)
        elif direction == 'FUTURE' and memory_timestamp > temporal_context.get('reference_time', anchor):
            score = min(1.0, score + 0.1)
        
        return score


class MultiSignalRetrieverV2:
    """Python simulation of MultiSignalRetriever v2 with TemporalReasoner"""
    
    def __init__(self, semantic_weight=0.5, keyword_weight=0.2, entity_weight=0.15, temporal_weight=0.15,
                 half_life_days=30.0, min_recency_score=0.1):
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.entity_weight = entity_weight
        self.temporal_weight = temporal_weight
        self.half_life_days = half_life_days
        self.min_recency_score = min_recency_score
        self.temporal_reasoner = TemporalReasoner(half_life_days, min_recency_score)
        
        self.source_weights = {"USER": 1.0, "AGENT": 0.9, "SYSTEM": 0.7}
    
    def rank(self, query, candidates, query_entities=None, current_time=None, context_entities=None):
        if not candidates:
            return []
        
        query_entities = query_entities or []
        context_entities = context_entities or []
        current_time = current_time or int(time.time() * 1000)
        
        # Analyze temporal intent ONCE
        temporal_context = self.temporal_reasoner.analyze(query, current_time)
        
        results = []
        for c in candidates:
            signals = self._compute_signals(c, query, query_entities, context_entities, current_time, temporal_context)
            final_score = self._combine_signals(signals)
            results.append(RankedMemory(
                id=c.id, content=c.content, score=final_score, signals=signals, entry=c.entry
            ))
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results
    
    def _compute_signals(self, candidate, query, query_entities, context_entities, current_time, temporal_context):
        semantic = candidate.semantic_score
        keyword = self._keyword_score(query, candidate.content)
        entity = self._entity_score(query_entities, context_entities, candidate.entry.entities)
        
        if temporal_context['direction'] != 'NONE':
            temporal = self.temporal_reasoner.compute_temporal_score(candidate.entry.timestamp, temporal_context)
        else:
            temporal = self._recency_score(candidate.entry.timestamp, current_time)
        
        source = self.source_weights.get(candidate.entry.source, 1.0)
        
        return SignalScores(semantic=semantic, keyword=keyword, entity=entity, recency=temporal, source_weight=source)
    
    def _combine_signals(self, s):
        weighted = (s.semantic * self.semantic_weight + s.keyword * self.keyword_weight +
                    s.entity * self.entity_weight + s.recency * self.temporal_weight)
        return weighted * s.source_weight
    
    def _keyword_score(self, query, content):
        q_tokens = self._tokenize(query)
        c_tokens = set(self._tokenize(content))
        if not q_tokens:
            return 0.0
        return sum(1 for t in q_tokens if t in c_tokens) / len(q_tokens)
    
    def _entity_score(self, query_entities, context_entities, memory_entities):
        if not memory_entities:
            return 0.0
        all_q = set(query_entities + context_entities)
        if not all_q:
            return 0.3
        match = sum(1 for e in memory_entities if e in all_q)
        return min(1.0, match / max(1, len(all_q)))
    
    def _recency_score(self, ts, now):
        age_days = (now - ts) / 86400000.0
        decay = math.exp(-0.693 * age_days / self.half_life_days)
        return max(self.min_recency_score, decay)
    
    def _tokenize(self, text):
        tokens = []
        for i in range(len(text) - 1):
            if ord(text[i]) > 0x4E00:
                tokens.append(text[i:i+2])
        tokens.extend([w for w in re.split(r'[\s,，。！？、；：]+', text) if len(w) >= 2])
        return tokens


# === Test Helpers ===

DAY = 86400000  # ms
NOW = 1747180800000  # 2025-05-14 00:00 UTC approx

def make_candidate(id, content, semantic, days_ago, entities=None, source="USER"):
    return RetrievalCandidate(
        id=id, content=content, semantic_score=semantic,
        entry=MemoryEntry(id=id, content=content, timestamp=NOW - days_ago * DAY,
                          entities=entities or [], source=source)
    )


# === Tests ===

def test_no_temporal_backward_compat():
    """无时间意图时，v2 行为应与 v1 一致（标准 recency decay）"""
    retriever = MultiSignalRetrieverV2()
    candidates = [
        make_candidate("new", "我喜欢吃火锅", 0.9, 1),
        make_candidate("old", "我喜欢吃烤鸭", 0.9, 100),
    ]
    results = retriever.rank("我喜欢吃什么", candidates, current_time=NOW)
    assert results[0].id == "new", f"Expected 'new' first, got '{results[0].id}'"
    # Recency should be standard decay
    new_rec = results[0].signals.recency
    old_rec = results[1].signals.recency
    assert new_rec > 0.9, f"New memory recency should be >0.9, got {new_rec:.3f}"
    assert old_rec < 0.2, f"Old memory recency should be <0.2, got {old_rec:.3f}"

def test_present_query_favors_recent():
    """'现在住哪' → 最近的记忆应排第一"""
    retriever = MultiSignalRetrieverV2()
    candidates = [
        make_candidate("bj", "我住在北京朝阳区", 0.85, 100, ["北京"]),
        make_candidate("sh", "我搬到了上海浦东", 0.92, 1, ["上海"]),
    ]
    results = retriever.rank("我现在住在哪里？", candidates, current_time=NOW)
    assert results[0].id == "sh", f"Expected '上海' first for '现在住哪', got '{results[0].content}'"
    print(f"  现在: {results[0].content}={results[0].score:.3f}, {results[1].content}={results[1].score:.3f}")

def test_past_query_favors_old():
    """'以前住哪' → 旧记忆应排第一（anchor 移到 90 天前）"""
    retriever = MultiSignalRetrieverV2()
    candidates = [
        make_candidate("bj", "我住在北京朝阳区", 0.85, 100, ["北京"]),
        make_candidate("sh", "我搬到了上海浦东", 0.85, 1, ["上海"]),  # same semantic to isolate temporal
    ]
    results = retriever.rank("我以前住在哪里？", candidates, current_time=NOW)
    assert results[0].id == "bj", f"Expected '北京' first for '以前住哪', got '{results[0].content}'"
    print(f"  以前: {results[0].content}={results[0].score:.3f}, {results[1].content}={results[1].score:.3f}")

def test_same_data_different_views():
    """核心哲学验证：同一批数据，不同时间视图产生不同排序"""
    retriever = MultiSignalRetrieverV2()
    candidates = [
        make_candidate("bj", "我住在北京", 0.85, 100, ["北京"]),
        make_candidate("sh", "我搬到了上海", 0.85, 1, ["上海"]),
    ]
    
    present_results = retriever.rank("我现在住在哪？", candidates, current_time=NOW)
    past_results = retriever.rank("我以前住在哪？", candidates, current_time=NOW)
    
    assert present_results[0].id == "sh", "现在 → 上海"
    assert past_results[0].id == "bj", "以前 → 北京"
    print(f"  ✓ Same data, different views — 信息不分类别，视图函数决定呈现")

def test_relative_time_query():
    """'3天前吃了什么' → anchor 在 3 天前，3 天前的记忆得分最高"""
    retriever = MultiSignalRetrieverV2()
    candidates = [
        make_candidate("today", "今天吃了寿司", 0.8, 0),
        make_candidate("3d", "吃了麻辣烫", 0.8, 3),
        make_candidate("30d", "吃了烤全羊", 0.8, 30),
    ]
    results = retriever.rank("3天前吃了什么？", candidates, current_time=NOW)
    assert results[0].id == "3d", f"Expected '3d' first, got '{results[0].id}'"
    print(f"  3天前: best={results[0].content}({results[0].score:.3f})")

def test_temporal_does_not_override_semantic():
    """时间信号不应完全覆盖语义相关性 (semantic weight=0.5 >> temporal weight=0.15)"""
    retriever = MultiSignalRetrieverV2()
    candidates = [
        make_candidate("relevant_old", "我住在北京", 0.95, 100, ["北京"]),
        make_candidate("irrelevant_new", "今天天气不错", 0.2, 1),
    ]
    # Even with "现在" pushing toward recent, semantic=0.95 vs 0.2 should dominate
    results = retriever.rank("我现在住在哪？", candidates, current_time=NOW)
    assert results[0].id == "relevant_old", \
        f"Semantic should dominate over temporal. Got '{results[0].content}' first"

def test_agent_source_with_temporal():
    """Agent source (weight=0.9) + temporal reasoning should work together"""
    retriever = MultiSignalRetrieverV2()
    candidates = [
        make_candidate("user_old", "我喜欢红色", 0.8, 100, source="USER"),
        make_candidate("agent_new", "用户似乎开始喜欢蓝色", 0.8, 1, source="AGENT"),
    ]
    results = retriever.rank("现在喜欢什么颜色？", candidates, current_time=NOW)
    # Agent source (0.9) + very high temporal (recent + "现在") should still beat old USER
    assert results[0].id == "agent_new", f"Agent new should win, got '{results[0].id}'"

def test_future_query():
    """'明天打算做什么' → future anchor, recent plans should rank high"""
    retriever = MultiSignalRetrieverV2()
    candidates = [
        make_candidate("plan", "明天要去看牙医", 0.85, 0),
        make_candidate("old_plan", "下个月去旅行", 0.85, 20),
    ]
    results = retriever.rank("明天打算做什么？", candidates, current_time=NOW)
    assert results[0].id == "plan", f"Expected today's plan first, got '{results[0].id}'"

def test_multiple_temporal_signals():
    """多个时间信号时取第一个（优先级）"""
    retriever = MultiSignalRetrieverV2()
    # "以前" will be detected first → PAST direction
    candidates = [
        make_candidate("old", "旧记忆", 0.8, 90),
        make_candidate("new", "新记忆", 0.8, 1),
    ]
    results = retriever.rank("以前到现在有什么变化？", candidates, current_time=NOW)
    # "以前" detected first → PAST → old memory closer to anchor
    assert results[0].id == "old", f"Expected 'old' first (以前 anchor), got '{results[0].id}'"

def test_entity_boost_with_temporal():
    """Entity matching + temporal reasoning 叠加"""
    retriever = MultiSignalRetrieverV2()
    candidates = [
        make_candidate("match", "北京的房子", 0.8, 90, ["北京"]),
        make_candidate("nomatch", "上海的工作", 0.8, 90, ["上海"]),
    ]
    results = retriever.rank("以前在北京住的怎么样？", candidates, 
                              query_entities=["北京"], current_time=NOW)
    assert results[0].id == "match", f"Entity+temporal should favor '北京', got '{results[0].id}'"


# === Run all tests ===

print("=" * 60)
print("Integration: MultiSignalRetriever v2 + TemporalReasoner")
print("=" * 60)

tests = [
    ("backward_compat_no_temporal", test_no_temporal_backward_compat),
    ("present_query_favors_recent", test_present_query_favors_recent),
    ("past_query_favors_old", test_past_query_favors_old),
    ("same_data_different_views", test_same_data_different_views),
    ("relative_time_3days_ago", test_relative_time_query),
    ("semantic_dominates_temporal", test_temporal_does_not_override_semantic),
    ("agent_source_with_temporal", test_agent_source_with_temporal),
    ("future_query", test_future_query),
    ("multiple_temporal_signals", test_multiple_temporal_signals),
    ("entity_boost_with_temporal", test_entity_boost_with_temporal),
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
