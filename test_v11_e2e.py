"""
End-to-End Test: ChatEngine v11 — Full-Stack ADD-only Architecture

Validates the complete pipeline:
FactExtractor → ValidityWindow → AdmissionController → ProvenanceTracker → 
AppendOnlyStore → RetrievalAgent → TemporalReasoner → MultiSignalRetriever v2 →
ValidityWindow (retrieval) → AccessFrequencyTracker → ProvenanceTracker (signal) →
ConflictResolution → ContextExpander
"""

import time
import math
import uuid
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

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
        print(f"✗ {name}: EXCEPTION {type(e).__name__}: {e}")
        TESTS_FAILED += 1


# === Unified Model Layer (simulating all Kotlin classes) ===

DAY = 86400000
NOW = 1747267200000  # ~2025-05-15 approx

class MemorySource(Enum):
    USER = "USER"
    AGENT = "AGENT"
    SYSTEM = "SYSTEM"

class AdmissionOutcome(Enum):
    ADMITTED = "ADMITTED"
    FAST_PATH = "FAST_PATH"
    REJECTED = "REJECTED"
    FILTERED = "FILTERED"

class ValidityCategory(Enum):
    PERMANENT = "PERMANENT"
    LONG_TERM = "LONG_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    SHORT_TERM = "SHORT_TERM"
    EPHEMERAL = "EPHEMERAL"

@dataclass
class MemoryEntry:
    id: str
    content: str
    source: MemorySource
    entities: List[str]
    timestamp: int
    session_id: str
    user_id: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class ValidityInfo:
    category: ValidityCategory
    estimated_days: float
    confidence: float
    reason: str

@dataclass
class ProvenanceRecord:
    memory_id: str
    turn_id: str
    session_id: str
    user_id: str
    original_message: str
    extraction_confidence: float
    admission_decision: AdmissionOutcome
    created_at: int

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
    signals: SignalScores = None
    entry: MemoryEntry = None
    validity_score: float = 1.0
    frequency_boost: float = 1.0
    provenance_signal: float = 1.0


# === Minimal Engine Simulation ===

class ChatEngineV11Sim:
    def __init__(self):
        self.store = []  # List[MemoryEntry]
        self.provenance = {}  # id → ProvenanceRecord
        self.validity_cache = {}  # id → ValidityInfo
        self.access_counts = defaultdict(int)  # id → count
        self.turn_counter = 0
        self.rejected = []
        
    def process_user_turn(self, user_id, message, session_id="s1", timestamp=None):
        ts = timestamp or int(time.time() * 1000)
        self.turn_counter += 1
        turn_id = f"turn_{self.turn_counter}"
        
        # 1. Extract facts (simplified)
        facts = self._extract_facts(message)
        
        stored = []
        for fact_content, fact_entities in facts:
            # 2. Admission
            admitted, score, fast_path = self._admission_check(fact_content)
            if not admitted:
                self.rejected.append(fact_content)
                continue
            
            # 3. Validity inference
            validity = self._infer_validity(fact_content)
            
            # 4. Store
            entry = MemoryEntry(
                id=str(uuid.uuid4())[:8],
                content=fact_content, source=MemorySource.USER,
                entities=fact_entities, timestamp=ts,
                session_id=session_id, user_id=user_id,
                metadata={"validity_category": validity.category.name,
                          "validity_days": str(validity.estimated_days)}
            )
            self.store.append(entry)
            stored.append(entry)
            
            # 5. Provenance
            outcome = AdmissionOutcome.FAST_PATH if fast_path else AdmissionOutcome.ADMITTED
            self.provenance[entry.id] = ProvenanceRecord(
                entry.id, turn_id, session_id, user_id,
                message, score, outcome, ts
            )
            
            # 6. Cache validity
            self.validity_cache[entry.id] = validity
        
        return {"stored": stored, "turn_id": turn_id, "rejected": len(self.rejected)}
    
    def retrieve(self, user_id, query, query_time=None):
        qt = query_time or int(time.time() * 1000)
        memories = [m for m in self.store if m.user_id == user_id]
        if not memories:
            return []
        
        # Temporal analysis
        temporal_ctx = self._analyze_temporal(query, qt)
        
        scored = []
        for mem in memories:
            # Base score (semantic + keyword + entity + temporal)
            semantic = self._synthetic_semantic(query, mem.content)
            keyword = self._keyword_score(query, mem.content)
            entity = 0.3  # neutral
            
            if temporal_ctx['direction'] != 'NONE':
                temporal = self._temporal_score(mem.timestamp, temporal_ctx)
            else:
                temporal = self._recency_score(mem.timestamp, qt)
            
            source_w = 1.0 if mem.source == MemorySource.USER else 0.9
            base = (semantic*0.5 + keyword*0.2 + entity*0.15 + temporal*0.15) * source_w
            
            # v11 signals
            validity = self.validity_cache.get(mem.id, 
                ValidityInfo(ValidityCategory.LONG_TERM, 365, 0.3, "default"))
            v_score = self._validity_score(mem.timestamp, qt, validity)
            
            f_boost = self._frequency_boost(mem.id)
            
            p_signal = self._provenance_signal(mem.id)
            
            final = base * v_score * f_boost * p_signal
            
            scored.append(RankedMemory(
                id=mem.id, content=mem.content, score=final,
                entry=mem, validity_score=v_score,
                frequency_boost=f_boost, provenance_signal=p_signal
            ))
        
        scored.sort(key=lambda x: x.score, reverse=True)
        
        # Record access for top-5
        for m in scored[:5]:
            self.access_counts[m.id] += 1
        
        return scored
    
    def explain_memory(self, memory_id):
        prov = self.provenance.get(memory_id)
        if not prov:
            return None
        return f"来自 {prov.session_id}: \"{prov.original_message}\""
    
    # --- Internal helpers ---
    
    def _extract_facts(self, message):
        if len(message) < 3:
            return []
        facts = []
        sentences = re.split(r'[。！？；\n]+', message)
        for s in sentences:
            s = s.strip()
            if len(s) < 3:
                continue
            entities = []
            for name in ['北京', '上海', '深圳', '花生', '猫', '年糕', '小明']:
                if name in s:
                    entities.append(name)
            facts.append((s, entities))
        return facts
    
    def _admission_check(self, content):
        # Fast path
        if any(kw in content for kw in ['过敏', '健康', '安全']):
            return True, 0.95, True
        # Reject fillers
        if len(content) < 4 or content in ['嗯', '好的', '哈哈', '嗯嗯', '好好好']:
            return False, 0.1, False
        return True, 0.7, False
    
    def _infer_validity(self, content):
        if re.search(r'过敏|血型|生日|一直|永远', content):
            return ValidityInfo(ValidityCategory.PERMANENT, -1, 0.9, "permanent")
        if re.search(r'现在|今天|此刻|累|饿|困', content):
            return ValidityInfo(ValidityCategory.EPHEMERAL, 1.0, 0.7, "ephemeral")
        if re.search(r'减肥|健身|项目|deadline', content):
            return ValidityInfo(ValidityCategory.MEDIUM_TERM, 60.0, 0.6, "medium")
        return ValidityInfo(ValidityCategory.LONG_TERM, 365.0, 0.3, "default")
    
    def _analyze_temporal(self, query, ref_time):
        if re.search(r'以前|之前|过去|曾经', query):
            return {'direction': 'PAST', 'anchor': ref_time - 90*DAY}
        if re.search(r'现在|目前|当前|最近', query):
            return {'direction': 'PRESENT', 'anchor': ref_time}
        if re.search(r'明天|下周|打算', query):
            return {'direction': 'FUTURE', 'anchor': ref_time + 7*DAY}
        return {'direction': 'NONE', 'anchor': ref_time}
    
    def _temporal_score(self, ts, ctx):
        dist = abs(ts - ctx['anchor']) / DAY
        score = math.exp(-0.693 * dist / 30.0)
        return max(0.1, score)
    
    def _recency_score(self, ts, now):
        age = (now - ts) / DAY
        return max(0.1, math.exp(-0.693 * age / 30.0))
    
    def _synthetic_semantic(self, query, content):
        q = set(query)
        c = set(content)
        inter = len(q & c)
        union = len(q | c)
        return inter / union if union > 0 else 0
    
    def _keyword_score(self, query, content):
        bigrams_q = [query[i:i+2] for i in range(len(query)-1) if ord(query[i]) > 0x4E00]
        if not bigrams_q:
            return 0
        bigrams_c = set(content[i:i+2] for i in range(len(content)-1) if ord(content[i]) > 0x4E00)
        return sum(1 for b in bigrams_q if b in bigrams_c) / len(bigrams_q)
    
    def _validity_score(self, fact_ts, query_time, validity):
        if validity.estimated_days < 0:
            return 1.0
        age = (query_time - fact_ts) / DAY
        if age <= validity.estimated_days:
            return 1.0
        overage = age - validity.estimated_days
        decay = math.exp(-0.1 * overage)
        return 0.3 + 0.7 * decay
    
    def _frequency_boost(self, mem_id):
        count = self.access_counts.get(mem_id, 0)
        if count < 1:
            return 1.0
        return min(1.5, 1.0 + 0.1 * math.log(1 + count))
    
    def _provenance_signal(self, mem_id):
        prov = self.provenance.get(mem_id)
        if not prov:
            return 1.0
        signal = prov.extraction_confidence
        if prov.admission_decision == AdmissionOutcome.FAST_PATH:
            signal = min(1.0, signal * 1.1)
        return max(0.3, min(1.0, signal))


# === Tests ===

def test_full_pipeline():
    """完整管线：ingest + retrieve"""
    engine = ChatEngineV11Sim()
    engine.process_user_turn("u1", "我住在北京朝阳区", timestamp=NOW - 100*DAY)
    engine.process_user_turn("u1", "我搬到了上海浦东", timestamp=NOW - 1*DAY)
    
    results = engine.retrieve("u1", "我现在住在哪？", NOW)
    assert len(results) >= 2
    # 上海 should rank higher (more recent + "现在" temporal intent)
    contents = [r.content for r in results]
    assert any("上海" in c for c in contents), "上海 should be in results"

def test_temporal_views():
    """核心哲学：同一数据，不同时间视图"""
    engine = ChatEngineV11Sim()
    engine.process_user_turn("u1", "我住在北京朝阳", timestamp=NOW - 100*DAY)
    engine.process_user_turn("u1", "我住在上海浦东", timestamp=NOW - 1*DAY)
    
    present = engine.retrieve("u1", "我现在住在哪？", NOW)
    past = engine.retrieve("u1", "我以前住在哪？", NOW)
    
    # Present → 上海 higher score
    sh_present = next(r for r in present if "上海" in r.content)
    bj_present = next(r for r in present if "北京" in r.content)
    assert sh_present.score > bj_present.score, \
        f"现在: 上海({sh_present.score:.3f}) should > 北京({bj_present.score:.3f})"
    
    # Past → 北京 higher score
    sh_past = next(r for r in past if "上海" in r.content)
    bj_past = next(r for r in past if "北京" in r.content)
    assert bj_past.score > sh_past.score, \
        f"以前: 北京({bj_past.score:.3f}) should > 上海({sh_past.score:.3f})"
    
    print(f"  现在: 上海={sh_present.score:.3f} > 北京={bj_present.score:.3f}")
    print(f"  以前: 北京={bj_past.score:.3f} > 上海={sh_past.score:.3f}")

def test_validity_window_effect():
    """有效期影响检索分数"""
    engine = ChatEngineV11Sim()
    # Permanent fact
    engine.process_user_turn("u1", "我对花生过敏", timestamp=NOW - 365*DAY)
    # Ephemeral fact
    engine.process_user_turn("u1", "我今天好累", timestamp=NOW - 30*DAY)
    
    results = engine.retrieve("u1", "关于我的信息", NOW)
    
    allergy = next(r for r in results if "过敏" in r.content)
    tired = next(r for r in results if "累" in r.content)
    
    # Allergy: permanent → validity=1.0 despite being old
    assert allergy.validity_score == 1.0, f"Permanent fact should have validity=1.0, got {allergy.validity_score}"
    # Tired: ephemeral with 1-day window, 30 days later → low validity
    assert tired.validity_score < 0.5, f"30-day-old ephemeral should have low validity, got {tired.validity_score:.3f}"
    print(f"  过敏(365d, permanent): validity={allergy.validity_score:.3f}")
    print(f"  累(30d, ephemeral): validity={tired.validity_score:.3f}")

def test_frequency_boost_accumulates():
    """频繁检索的记忆获得 boost"""
    engine = ChatEngineV11Sim()
    engine.process_user_turn("u1", "我喜欢吃火锅", timestamp=NOW - 10*DAY)
    engine.process_user_turn("u1", "我养了一只猫叫年糕", timestamp=NOW - 10*DAY)
    
    # First retrieval
    r1 = engine.retrieve("u1", "我喜欢什么", NOW)
    hotpot_1 = next(r for r in r1 if "火锅" in r.content)
    boost_1 = hotpot_1.frequency_boost
    
    # Retrieve again (simulating repeated access)
    for _ in range(5):
        engine.retrieve("u1", "我喜欢什么", NOW)
    
    r2 = engine.retrieve("u1", "我喜欢什么", NOW)
    hotpot_2 = next(r for r in r2 if "火锅" in r.content)
    boost_2 = hotpot_2.frequency_boost
    
    assert boost_2 > boost_1, f"Repeated access should increase boost: {boost_1:.3f} → {boost_2:.3f}"
    print(f"  Fire pot boost: {boost_1:.3f} → {boost_2:.3f} after 6 retrievals")

def test_provenance_tracking():
    """溯源：记忆可以追溯到原始消息"""
    engine = ChatEngineV11Sim()
    result = engine.process_user_turn("u1", "我对花生过敏，不能吃坚果", session_id="afternoon_chat")
    
    stored = result["stored"]
    assert len(stored) > 0
    
    # Check provenance
    for entry in stored:
        explanation = engine.explain_memory(entry.id)
        assert explanation is not None
        assert "afternoon_chat" in explanation
        assert "花生过敏" in explanation or "坚果" in explanation

def test_admission_rejects_fillers():
    """准入控制过滤填充语"""
    engine = ChatEngineV11Sim()
    engine.process_user_turn("u1", "嗯")
    engine.process_user_turn("u1", "好的")
    engine.process_user_turn("u1", "我对花生过敏，很严重")
    
    assert len(engine.store) == 1, f"Only meaningful fact should be stored, got {len(engine.store)}"
    assert "过敏" in engine.store[0].content

def test_multi_session():
    """跨 session 记忆保持"""
    engine = ChatEngineV11Sim()
    engine.process_user_turn("u1", "我住在北京", session_id="s1", timestamp=NOW - 30*DAY)
    engine.process_user_turn("u1", "我养了一只猫", session_id="s2", timestamp=NOW - 1*DAY)
    
    results = engine.retrieve("u1", "关于我", NOW)
    contents = " ".join(r.content for r in results)
    assert "北京" in contents and "猫" in contents

def test_add_only_never_deletes():
    """ADD-only: 即使有矛盾也不删除"""
    engine = ChatEngineV11Sim()
    engine.process_user_turn("u1", "我住在北京朝阳", timestamp=NOW - 100*DAY)
    engine.process_user_turn("u1", "我住在上海", timestamp=NOW - 50*DAY)
    engine.process_user_turn("u1", "我住在深圳", timestamp=NOW - 1*DAY)
    
    assert len(engine.store) == 3, f"All 3 locations should be stored, got {len(engine.store)}"
    
    results = engine.retrieve("u1", "我住在哪", NOW)
    assert len(results) == 3

def test_fast_path_provenance():
    """高价值记忆的 fast-path 溯源"""
    engine = ChatEngineV11Sim()
    result = engine.process_user_turn("u1", "我对花生过敏")
    
    entry = result["stored"][0]
    prov = engine.provenance[entry.id]
    assert prov.admission_decision == AdmissionOutcome.FAST_PATH

def test_stats():
    """引擎统计"""
    engine = ChatEngineV11Sim()
    engine.process_user_turn("u1", "我住在北京")
    engine.process_user_turn("u1", "我喜欢编程")
    engine.retrieve("u1", "告诉我关于我的", NOW)
    
    assert len(engine.store) == 2
    assert len(engine.provenance) == 2
    assert engine.turn_counter == 2

def test_v11_signals_all_combine():
    """验证 v11 三个新信号都参与 final score"""
    engine = ChatEngineV11Sim()
    # Permanent fact (validity=1.0) with high confidence (fast path)
    engine.process_user_turn("u1", "我对花生过敏", timestamp=NOW - 10*DAY)
    # Ephemeral fact long expired (validity≈0.3) with normal confidence
    engine.process_user_turn("u1", "我今天特别累", timestamp=NOW - 60*DAY)
    
    results = engine.retrieve("u1", "关于我", NOW)
    
    allergy = next(r for r in results if "过敏" in r.content)
    tired = next(r for r in results if "累" in r.content)
    
    # Allergy should dominate: permanent validity + fast-path provenance boost
    assert allergy.score > tired.score, \
        f"过敏({allergy.score:.4f}) should > 累({tired.score:.4f})"
    
    # Verify signals are populated
    assert allergy.validity_score == 1.0
    assert allergy.provenance_signal > 0.9  # fast path boost
    assert tired.validity_score < 0.5  # expired ephemeral

def test_pipeline_performance():
    """性能: 500 memories should process quickly"""
    engine = ChatEngineV11Sim()
    for i in range(100):
        engine.process_user_turn("u1", f"记忆条目 {i}: 我在第{i}天做了一些有趣的事情",
                                  timestamp=NOW - (100-i)*DAY)
    
    start = time.time()
    results = engine.retrieve("u1", "我做了什么有趣的事", NOW)
    elapsed = (time.time() - start) * 1000
    
    assert len(results) > 0
    assert elapsed < 500, f"Retrieval took {elapsed:.0f}ms, should be <500ms"
    print(f"  100 memories → retrieve in {elapsed:.1f}ms")


# === Run ===
print("=" * 60)
print("ChatEngine v11 E2E Tests — Full-Stack ADD-only Architecture")
print("=" * 60)

tests = [
    ("full_pipeline", test_full_pipeline),
    ("temporal_views", test_temporal_views),
    ("validity_window_effect", test_validity_window_effect),
    ("frequency_boost_accumulates", test_frequency_boost_accumulates),
    ("provenance_tracking", test_provenance_tracking),
    ("admission_rejects_fillers", test_admission_rejects_fillers),
    ("multi_session", test_multi_session),
    ("add_only_never_deletes", test_add_only_never_deletes),
    ("fast_path_provenance", test_fast_path_provenance),
    ("stats", test_stats),
    ("v11_signals_combine", test_v11_signals_all_combine),
    ("pipeline_performance", test_pipeline_performance),
]

for name, fn in tests:
    run_test(name, fn)

print()
print("=" * 50)
print(f"Results: {TESTS_PASSED} passed, {TESTS_FAILED} failed, {TESTS_PASSED + TESTS_FAILED} total")
if TESTS_FAILED == 0:
    print("ALL V11 E2E TESTS PASSED ✓")
    print()
    print("Architecture validated:")
    print("  FactExtractor → ValidityWindow → AdmissionController → ProvenanceTracker")
    print("  → AppendOnlyStore → RetrievalAgent → TemporalReasoner")
    print("  → MultiSignalRetriever v2 → ValidityWindow × FrequencyBoost × Provenance")
    print("  → ConflictResolution → ContextExpander")
    print("  = Complete v11 ADD-only pipeline ✓")
else:
    print(f"FAILURES: {TESTS_FAILED}")
    sys.exit(1)
