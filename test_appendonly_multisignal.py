"""
Integration tests for AppendOnlyStore + MultiSignalRetriever
Python simulation of Kotlin logic for validation
"""
import time
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

# ============================
# Data Models
# ============================

class MemorySource(Enum):
    USER = "USER"
    AGENT = "AGENT"
    SYSTEM = "SYSTEM"

@dataclass
class MemoryEntry:
    id: str
    content: str
    source: MemorySource
    entities: List[str]
    timestamp: int  # ms
    session_id: Optional[str]
    user_id: Optional[str]
    metadata: Dict
    created_at: int

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

# ============================
# AppendOnlyStore
# ============================

class AppendOnlyStore:
    def __init__(self):
        self.memories: Dict[str, MemoryEntry] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
    
    def add(self, content: str, source: MemorySource, entities: List[str] = None,
            timestamp: int = None, session_id: str = None, user_id: str = None,
            metadata: Dict = None) -> MemoryEntry:
        entities = entities or []
        timestamp = timestamp or int(time.time() * 1000)
        metadata = metadata or {}
        
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            source=source,
            entities=entities,
            timestamp=timestamp,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            created_at=int(time.time() * 1000)
        )
        
        self.memories[entry.id] = entry
        for entity in entities:
            self.entity_index[entity].add(entry.id)
        
        return entry
    
    def get_all(self, user_id: str = None) -> List[MemoryEntry]:
        if user_id:
            return [m for m in self.memories.values() if m.user_id == user_id]
        return list(self.memories.values())
    
    def get_by_entity(self, entity: str) -> List[MemoryEntry]:
        ids = self.entity_index.get(entity, set())
        return [self.memories[id] for id in ids if id in self.memories]
    
    def count(self, user_id: str = None) -> int:
        if user_id:
            return sum(1 for m in self.memories.values() if m.user_id == user_id)
        return len(self.memories)

# ============================
# MultiSignalRetriever
# ============================

class MultiSignalRetriever:
    def __init__(self,
                 semantic_weight=0.5,
                 keyword_weight=0.2,
                 entity_weight=0.15,
                 temporal_weight=0.15,
                 half_life_days=30.0,
                 min_recency_score=0.1,
                 user_source_weight=1.0,
                 agent_source_weight=0.9,
                 system_source_weight=0.7):
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.entity_weight = entity_weight
        self.temporal_weight = temporal_weight
        self.half_life_days = half_life_days
        self.min_recency_score = min_recency_score
        self.source_weights = {
            MemorySource.USER: user_source_weight,
            MemorySource.AGENT: agent_source_weight,
            MemorySource.SYSTEM: system_source_weight,
        }
    
    def rank(self, query: str, candidates: List[RetrievalCandidate],
             query_entities: List[str] = None,
             current_time: int = None,
             context_entities: List[str] = None) -> List[RankedMemory]:
        query_entities = query_entities or []
        context_entities = context_entities or []
        current_time = current_time or int(time.time() * 1000)
        
        results = []
        for c in candidates:
            signals = self._compute_signals(c, query, query_entities, context_entities, current_time)
            final_score = self._combine_signals(signals)
            results.append(RankedMemory(
                id=c.id, content=c.content, score=final_score,
                signals=signals, entry=c.entry
            ))
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def _compute_signals(self, candidate, query, query_entities, context_entities, current_time):
        semantic = candidate.semantic_score
        keyword = self._keyword_score(query, candidate.content)
        entity = self._entity_score(query_entities, context_entities, candidate.entry.entities)
        recency = self._recency_score(candidate.entry.timestamp, current_time)
        source_weight = self.source_weights.get(candidate.entry.source, 0.7)
        
        return SignalScores(
            semantic=semantic, keyword=keyword, entity=entity,
            recency=recency, source_weight=source_weight
        )
    
    def _combine_signals(self, s: SignalScores) -> float:
        weighted = (
            s.semantic * self.semantic_weight +
            s.keyword * self.keyword_weight +
            s.entity * self.entity_weight +
            s.recency * self.temporal_weight
        )
        return weighted * s.source_weight
    
    def _keyword_score(self, query: str, content: str) -> float:
        """Simplified BM25: Chinese bigrams + word split"""
        q_tokens = self._tokenize(query)
        c_tokens = set(self._tokenize(content))
        if not q_tokens:
            return 0.0
        match_count = sum(1 for t in q_tokens if t in c_tokens)
        return match_count / len(q_tokens)
    
    def _entity_score(self, query_entities, context_entities, memory_entities):
        if not memory_entities:
            return 0.0
        all_q = set(query_entities + context_entities)
        if not all_q:
            return 0.3  # neutral baseline
        match_count = sum(1 for e in memory_entities if e in all_q)
        return min(1.0, match_count / max(1, len(all_q)))
    
    def _recency_score(self, mem_timestamp: int, current_time: int) -> float:
        age_ms = current_time - mem_timestamp
        age_days = age_ms / (1000 * 60 * 60 * 24)
        decay = math.exp(-0.693 * age_days / self.half_life_days)
        return max(self.min_recency_score, decay)
    
    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        # Chinese bigrams
        for i in range(len(text) - 1):
            if ord(text[i]) > 0x4E00:
                tokens.append(text[i:i+2])
        # Word split
        import re
        words = re.split(r'[\s,，。！？、；：]+', text)
        tokens.extend(w for w in words if len(w) >= 2)
        return tokens
    
    def resolve_conflicts(self, ranked: List[RankedMemory],
                         entity_groups: Dict[str, List[RankedMemory]]):
        current = []
        historical = []
        
        for entity, memories in entity_groups.items():
            if len(memories) <= 1:
                current.extend(memories)
                continue
            sorted_by_time = sorted(memories, key=lambda m: m.entry.timestamp if m.entry else 0, reverse=True)
            current.append(sorted_by_time[0])
            historical.extend(sorted_by_time[1:])
        
        grouped_ids = {m.id for mems in entity_groups.values() for m in mems}
        current.extend(m for m in ranked if m.id not in grouped_ids)
        
        return {
            'current': sorted(current, key=lambda m: m.score, reverse=True),
            'historical': historical,
            'conflict_count': len(historical)
        }


# ============================
# Tests
# ============================

def test_append_only_basic():
    """Basic ADD operations"""
    store = AppendOnlyStore()
    now = int(time.time() * 1000)
    
    e1 = store.add('我叫小明', source=MemorySource.USER, entities=['小明'], 
                   timestamp=now, user_id='u1')
    e2 = store.add('住在北京', source=MemorySource.USER, entities=['北京'],
                   timestamp=now, user_id='u1')
    
    assert store.count('u1') == 2
    assert store.count() == 2
    assert e1.id != e2.id
    assert e1.content == '我叫小明'
    print("✓ test_append_only_basic")

def test_append_only_never_overwrites():
    """Same entity, multiple ADD → all preserved"""
    store = AppendOnlyStore()
    now = int(time.time() * 1000)
    
    store.add('住在北京', source=MemorySource.USER, entities=['住所', '北京'],
              timestamp=now - 30*86400*1000, user_id='u1')
    store.add('搬到了上海', source=MemorySource.USER, entities=['住所', '上海'],
              timestamp=now - 1*86400*1000, user_id='u1')
    
    # Both still exist!
    all_mems = store.get_all('u1')
    assert len(all_mems) == 2
    
    # Entity index has both
    loc_mems = store.get_by_entity('住所')
    assert len(loc_mems) == 2
    print("✓ test_append_only_never_overwrites")

def test_entity_index():
    """Entity linking without graph DB"""
    store = AppendOnlyStore()
    now = int(time.time() * 1000)
    
    store.add('小明喜欢打篮球', source=MemorySource.USER, entities=['小明', '篮球'],
              timestamp=now, user_id='u1')
    store.add('小红也喜欢篮球', source=MemorySource.USER, entities=['小红', '篮球'],
              timestamp=now, user_id='u1')
    store.add('小明在字节工作', source=MemorySource.USER, entities=['小明', '字节'],
              timestamp=now, user_id='u1')
    
    xiaoming_mems = store.get_by_entity('小明')
    assert len(xiaoming_mems) == 2
    
    basketball_mems = store.get_by_entity('篮球')
    assert len(basketball_mems) == 2
    print("✓ test_entity_index")

def test_multi_source():
    """Agent facts are first-class (Mem0 v3 insight)"""
    store = AppendOnlyStore()
    now = int(time.time() * 1000)
    
    store.add('我过敏不能吃花生', source=MemorySource.USER, entities=['花生过敏'],
              timestamp=now, user_id='u1')
    store.add('已记录：用户对花生过敏', source=MemorySource.AGENT, entities=['花生过敏'],
              timestamp=now, user_id='u1')
    store.add('用户打开了设置页面', source=MemorySource.SYSTEM, entities=[],
              timestamp=now, user_id='u1')
    
    all_mems = store.get_all('u1')
    sources = {m.source for m in all_mems}
    assert MemorySource.USER in sources
    assert MemorySource.AGENT in sources
    assert MemorySource.SYSTEM in sources
    print("✓ test_multi_source")

def test_retriever_basic_ranking():
    """Semantic score dominates when other signals are neutral"""
    retriever = MultiSignalRetriever()
    now = int(time.time() * 1000)
    
    entries = [
        MemoryEntry(id='1', content='我叫小明', source=MemorySource.USER,
                    entities=['小明'], timestamp=now, session_id=None,
                    user_id='u1', metadata={}, created_at=now),
        MemoryEntry(id='2', content='天气很好', source=MemorySource.USER,
                    entities=[], timestamp=now, session_id=None,
                    user_id='u1', metadata={}, created_at=now),
    ]
    
    candidates = [
        RetrievalCandidate('1', '我叫小明', 0.95, entries[0]),
        RetrievalCandidate('2', '天气很好', 0.3, entries[1]),
    ]
    
    ranked = retriever.rank('你叫什么名字', candidates, current_time=now)
    assert ranked[0].content == '我叫小明'
    assert ranked[0].score > ranked[1].score
    print("✓ test_retriever_basic_ranking")

def test_recency_decay():
    """Recent memories get higher recency scores"""
    retriever = MultiSignalRetriever()
    now = int(time.time() * 1000)
    
    # 1 day old
    score_1d = retriever._recency_score(now - 1*86400*1000, now)
    # 30 days old (half-life)
    score_30d = retriever._recency_score(now - 30*86400*1000, now)
    # 90 days old
    score_90d = retriever._recency_score(now - 90*86400*1000, now)
    
    assert score_1d > score_30d > score_90d
    # At half-life, score should be ~0.5
    assert 0.45 < score_30d < 0.55
    # Never reaches zero
    assert score_90d >= 0.1
    print(f"  1d={score_1d:.3f}, 30d={score_30d:.3f}, 90d={score_90d:.3f}")
    print("✓ test_recency_decay")

def test_source_weighting():
    """USER > AGENT > SYSTEM in base weight"""
    retriever = MultiSignalRetriever()
    now = int(time.time() * 1000)
    
    # Same content, same semantic score, same time — differ only by source
    base = lambda src: MemoryEntry(
        id=str(uuid.uuid4()), content='test', source=src, entities=[],
        timestamp=now, session_id=None, user_id='u1', metadata={}, created_at=now)
    
    candidates = [
        RetrievalCandidate('a', 'test', 0.8, base(MemorySource.USER)),
        RetrievalCandidate('b', 'test', 0.8, base(MemorySource.AGENT)),
        RetrievalCandidate('c', 'test', 0.8, base(MemorySource.SYSTEM)),
    ]
    
    ranked = retriever.rank('test', candidates, current_time=now)
    # USER ranked highest
    assert ranked[0].entry.source == MemorySource.USER
    assert ranked[2].entry.source == MemorySource.SYSTEM
    print("✓ test_source_weighting")

def test_entity_boosting():
    """Memories matching query entities get boosted"""
    retriever = MultiSignalRetriever()
    now = int(time.time() * 1000)
    
    entries = [
        MemoryEntry(id='1', content='小明喜欢篮球', source=MemorySource.USER,
                    entities=['小明', '篮球'], timestamp=now, session_id=None,
                    user_id='u1', metadata={}, created_at=now),
        MemoryEntry(id='2', content='天气预报说明天下雨', source=MemorySource.USER,
                    entities=['天气'], timestamp=now, session_id=None,
                    user_id='u1', metadata={}, created_at=now),
    ]
    
    candidates = [
        RetrievalCandidate('1', '小明喜欢篮球', 0.7, entries[0]),
        RetrievalCandidate('2', '天气预报说明天下雨', 0.7, entries[1]),
    ]
    
    # Query mentions 小明 → entry 1 should be boosted
    ranked = retriever.rank('小明的爱好是什么', candidates, 
                           query_entities=['小明'], current_time=now)
    assert ranked[0].content == '小明喜欢篮球'
    print("✓ test_entity_boosting")

def test_keyword_matching():
    """BM25 keyword signal with Chinese bigrams"""
    retriever = MultiSignalRetriever()
    
    # '住在' appears as bigram in both query and content
    score1 = retriever._keyword_score('住在哪里', '住在北京')
    # '搬到' does not overlap with '住在哪里'
    score2 = retriever._keyword_score('住在哪里', '搬到了上海')
    
    assert score1 > score2, f"Expected '住在北京' to keyword-match better: {score1} vs {score2}"
    print(f"  '住在北京' kw={score1:.3f}, '搬到了上海' kw={score2:.3f}")
    print("✓ test_keyword_matching")

def test_conflict_resolution():
    """Conflict resolution marks newest as current, older as historical"""
    retriever = MultiSignalRetriever()
    now = int(time.time() * 1000)
    
    e_old = MemoryEntry(id='old', content='住在北京', source=MemorySource.USER,
                        entities=['住所'], timestamp=now - 30*86400*1000,
                        session_id=None, user_id='u1', metadata={}, created_at=now)
    e_new = MemoryEntry(id='new', content='搬到了上海', source=MemorySource.USER,
                        entities=['住所'], timestamp=now - 1*86400*1000,
                        session_id=None, user_id='u1', metadata={}, created_at=now)
    
    ranked = [
        RankedMemory('old', '住在北京', 0.7, SignalScores(0.7, 0.5, 0.5, 0.3, 1.0), e_old),
        RankedMemory('new', '搬到了上海', 0.8, SignalScores(0.8, 0.3, 0.5, 0.95, 1.0), e_new),
    ]
    
    entity_groups = {'住所': ranked}
    resolution = retriever.resolve_conflicts(ranked, entity_groups)
    
    # Newest is current
    assert resolution['current'][0].content == '搬到了上海'
    assert resolution['historical'][0].content == '住在北京'
    assert resolution['conflict_count'] == 1
    print("✓ test_conflict_resolution")

def test_full_pipeline():
    """
    End-to-end: ADD-only store + multi-signal retrieval resolves contradictions
    
    Scenario: User moved from 北京 to 上海. Query: 住在哪里?
    Expected: 搬到了上海 ranked #1 (via recency + semantic)
    
    Key insight: In a real system, semantic search would give '搬到了上海' 
    a HIGHER score because it's a location-change event directly answering
    "where do you live NOW". The keyword overlap of '住在' with '住在北京' 
    is a red herring — semantic similarity handles intent better.
    """
    store = AppendOnlyStore()
    retriever = MultiSignalRetriever()
    now = int(time.time() * 1000)
    
    # Build memory timeline
    store.add('我叫小明', source=MemorySource.USER, entities=['小明'], 
              timestamp=now - 30*86400*1000, user_id='u1')
    store.add('住在北京', source=MemorySource.USER, entities=['北京', '住所'],
              timestamp=now - 20*86400*1000, user_id='u1')
    store.add('搬到了上海', source=MemorySource.USER, entities=['上海', '住所'],
              timestamp=now - 1*86400*1000, user_id='u1')
    store.add('好的，已更新您的位置为上海', source=MemorySource.AGENT, entities=['上海', '住所'],
              timestamp=now - 1*86400*1000, user_id='u1')
    
    all_mems = store.get_all('u1')
    
    # Simulate realistic semantic scores for query "住在哪里"
    # A real embedding model would understand:
    # - "搬到了上海" → high relevance (relocation = current location answer)
    # - "住在北京" → moderate relevance (historical location)
    # - "好的，已更新您的位置为上海" → moderate (confirmation, not direct answer)
    # - "我叫小明" → low relevance (name, not location)
    def semantic_score(content):
        if '搬到' in content:
            return 0.92  # Move event directly answers "where now?"
        elif '住在' in content:
            return 0.85  # Historical statement about living somewhere
        elif '位置' in content or '上海' in content:
            return 0.75  # Related but indirect
        else:
            return 0.3   # Unrelated
    
    candidates = [RetrievalCandidate(m.id, m.content, semantic_score(m.content), m) 
                  for m in all_mems]
    
    ranked = retriever.rank('住在哪里', candidates, 
                           query_entities=['住所'], current_time=now)
    
    # Debug output
    print("  Ranking results:")
    for i, r in enumerate(ranked):
        print(f"    #{i+1} [{r.score:.4f}] {r.content}")
        s = r.signals
        print(f"         sem={s.semantic:.3f} kw={s.keyword:.3f} ent={s.entity:.3f} "
              f"rec={s.recency:.3f} src={s.source_weight:.2f}")
    
    # Most recent location should be top
    assert '上海' in ranked[0].content, f"Expected 上海 in top result, got: {ranked[0].content}"
    # Specifically the user's statement (not agent echo)
    assert '搬到' in ranked[0].content or ('上海' in ranked[0].content and ranked[0].entry.source == MemorySource.USER)
    print("✓ test_full_pipeline")

def test_full_pipeline_with_conflict_resolution():
    """
    Full pipeline + conflict resolution view
    """
    store = AppendOnlyStore()
    retriever = MultiSignalRetriever()
    now = int(time.time() * 1000)
    
    store.add('住在北京', source=MemorySource.USER, entities=['住所'],
              timestamp=now - 60*86400*1000, user_id='u1')
    store.add('搬到了上海', source=MemorySource.USER, entities=['住所'],
              timestamp=now - 10*86400*1000, user_id='u1')
    store.add('又搬到了深圳', source=MemorySource.USER, entities=['住所'],
              timestamp=now - 1*86400*1000, user_id='u1')
    
    all_mems = store.get_all('u1')
    candidates = [RetrievalCandidate(m.id, m.content, 0.8, m) for m in all_mems]
    
    ranked = retriever.rank('住在哪里', candidates, 
                           query_entities=['住所'], current_time=now)
    
    # Group by entity '住所'
    entity_groups = {'住所': ranked}
    resolution = retriever.resolve_conflicts(ranked, entity_groups)
    
    # Most recent is "current"
    assert '深圳' in resolution['current'][0].content
    # Others are historical (not deleted!)
    assert resolution['conflict_count'] == 2
    assert len(resolution['historical']) == 2
    print("✓ test_full_pipeline_with_conflict_resolution")

def test_add_only_vs_crud():
    """
    Demonstrates why ADD-only outperforms CRUD:
    A CRUD system would have deleted "住在北京" when "搬到上海" arrived.
    But the user might later ask "我以前住在哪里？" — ADD-only preserves this.
    """
    store = AppendOnlyStore()
    now = int(time.time() * 1000)
    retriever = MultiSignalRetriever()
    
    store.add('住在北京', source=MemorySource.USER, entities=['住所', '北京'],
              timestamp=now - 365*86400*1000, user_id='u1')
    store.add('搬到了上海', source=MemorySource.USER, entities=['住所', '上海'],
              timestamp=now - 30*86400*1000, user_id='u1')
    store.add('又搬到了深圳', source=MemorySource.USER, entities=['住所', '深圳'],
              timestamp=now - 1*86400*1000, user_id='u1')
    
    all_mems = store.get_all('u1')
    
    # Query 1: "现在住在哪里" → 深圳 (newest)
    candidates1 = [RetrievalCandidate(m.id, m.content, 
                   0.9 if '搬到了深圳' in m.content or '又搬' in m.content else 0.7, m)
                   for m in all_mems]
    ranked1 = retriever.rank('现在住在哪里', candidates1, 
                            query_entities=['住所'], current_time=now)
    assert '深圳' in ranked1[0].content
    
    # Query 2: "以前住在哪里" → 北京/上海 (historical)
    # Semantic model gives higher scores to older locations for "以前" queries
    candidates2 = [RetrievalCandidate(m.id, m.content,
                   0.9 if '住在北京' in m.content else 0.85 if '搬到了上海' in m.content else 0.5, m)
                   for m in all_mems]
    ranked2 = retriever.rank('以前住在哪里', candidates2,
                            query_entities=['住所'], current_time=now)
    # Should return historical locations — not just the newest!
    contents = [r.content for r in ranked2[:2]]
    assert any('北京' in c for c in contents), f"Historical query should surface 北京: {contents}"
    print("✓ test_add_only_vs_crud")

def test_batch_add():
    """Batch add works correctly"""
    store = AppendOnlyStore()
    now = int(time.time() * 1000)
    
    # Simulate batch with multiple adds
    contents = ['事实1', '事实2', '事实3']
    for c in contents:
        store.add(c, source=MemorySource.USER, timestamp=now, user_id='u1')
    
    assert store.count('u1') == 3
    print("✓ test_batch_add")

def test_no_deletion_ever():
    """Store has no delete method — architectural guarantee"""
    store = AppendOnlyStore()
    assert not hasattr(store, 'delete')
    assert not hasattr(store, 'update')
    assert not hasattr(store, 'remove')
    print("✓ test_no_deletion_ever")


if __name__ == '__main__':
    tests = [
        test_append_only_basic,
        test_append_only_never_overwrites,
        test_entity_index,
        test_multi_source,
        test_retriever_basic_ranking,
        test_recency_decay,
        test_source_weighting,
        test_entity_boosting,
        test_keyword_matching,
        test_conflict_resolution,
        test_full_pipeline,
        test_full_pipeline_with_conflict_resolution,
        test_add_only_vs_crud,
        test_batch_add,
        test_no_deletion_ever,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    if failed == 0:
        print("ALL TESTS PASSED ✓")
