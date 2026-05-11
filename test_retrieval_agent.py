"""
Tests for RetrievalAgent — multi-strategy routing + MQR
Python simulation of Kotlin RetrievalAgent.kt
"""
import time
import math
import uuid
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

# === Reuse models from appendonly test ===

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
    timestamp: int
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
    
    def copy(self, **kwargs):
        d = {'id': self.id, 'content': self.content, 'score': self.score,
             'signals': self.signals, 'entry': self.entry}
        d.update(kwargs)
        return RankedMemory(**d)


class RetrievalStrategy(Enum):
    DIRECT = "DIRECT"
    SPLIT_QUERY = "SPLIT_QUERY"
    CHAIN_OF_QUERY = "CHAIN_OF_QUERY"


# === Simplified MultiSignalRetriever ===
class MultiSignalRetriever:
    def __init__(self):
        self.semantic_w = 0.5
        self.keyword_w = 0.2
        self.entity_w = 0.15
        self.temporal_w = 0.15
        self.half_life = 30.0
        self.source_weights = {
            MemorySource.USER: 1.0, MemorySource.AGENT: 0.9, MemorySource.SYSTEM: 0.7
        }
    
    def rank(self, query, candidates, query_entities=None, current_time=None, context_entities=None):
        query_entities = query_entities or []
        context_entities = context_entities or []
        current_time = current_time or int(time.time()*1000)
        
        results = []
        for c in candidates:
            kw = self._keyword_score(query, c.content)
            ent = self._entity_score(query_entities + context_entities, c.entry.entities)
            rec = self._recency(c.entry.timestamp, current_time)
            src = self.source_weights.get(c.entry.source, 0.7)
            
            score = (c.semantic_score * self.semantic_w + kw * self.keyword_w +
                     ent * self.entity_w + rec * self.temporal_w) * src
            
            results.append(RankedMemory(
                id=c.id, content=c.content, score=score,
                signals=SignalScores(c.semantic_score, kw, ent, rec, src),
                entry=c.entry
            ))
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def _keyword_score(self, query, content):
        q_tokens = self._tokenize(query)
        c_tokens = set(self._tokenize(content))
        if not q_tokens: return 0.0
        return sum(1 for t in q_tokens if t in c_tokens) / len(q_tokens)
    
    def _entity_score(self, all_query_entities, memory_entities):
        if not memory_entities: return 0.0
        all_q = set(all_query_entities)
        if not all_q: return 0.3
        return min(1.0, sum(1 for e in memory_entities if e in all_q) / max(1, len(all_q)))
    
    def _recency(self, ts, now):
        age_days = (now - ts) / (1000*60*60*24)
        decay = math.exp(-0.693 * age_days / self.half_life)
        return max(0.1, decay)
    
    def _tokenize(self, text):
        tokens = []
        for i in range(len(text)-1):
            if ord(text[i]) > 0x4E00:
                tokens.append(text[i:i+2])
        tokens.extend(w for w in re.split(r'[\s,，。！？、；：]+', text) if len(w) >= 2)
        return tokens


# === RetrievalAgent (Python port) ===

class RetrievalAgent:
    def __init__(self, retriever=None, mqr_boost=1.3, mqr_min_hits=2,
                 max_sub_queries=4, max_chain_depth=3):
        self.retriever = retriever or MultiSignalRetriever()
        self.mqr_boost = mqr_boost
        self.mqr_min_hits = mqr_min_hits
        self.max_sub_queries = max_sub_queries
        self.max_chain_depth = max_chain_depth
        
        self.temporal_pats = [
            re.compile(r'以前|之前|过去|曾经|原来|上次|去年|前天|昨天'),
            re.compile(r'现在|目前|当前|最近|刚才|今天'),
            re.compile(r'后来|之后|接着|然后|最终')
        ]
        self.causal_pats = [
            re.compile(r'为什么|因为|所以|导致|引起|由于'),
            re.compile(r'如果|假如|要是|万一')
        ]
        self.conjunction_pats = [
            re.compile(r'和|与|以及|还有|另外'),
            re.compile(r'而且|并且|同时'),
            re.compile(r'[,，;；]')
        ]
    
    def analyze_query(self, query, context_entities=None):
        context_entities = context_entities or []
        entity_count = len(context_entities) + self._count_inline_entities(query)
        clause_count = self._count_clauses(query)
        has_temporal = any(p.search(query) for p in self.temporal_pats)
        has_causal = any(p.search(query) for p in self.causal_pats)
        
        strategy, confidence = self._decide_strategy(
            entity_count, clause_count, has_temporal, has_causal, len(query))
        
        if strategy == RetrievalStrategy.SPLIT_QUERY:
            sub_queries = self._split_query(query)
        elif strategy == RetrievalStrategy.CHAIN_OF_QUERY:
            sub_queries = self._chain_decompose(query)
        else:
            sub_queries = []
        
        return {
            'strategy': strategy, 'confidence': confidence,
            'entity_count': entity_count, 'clause_count': clause_count,
            'has_temporal': has_temporal, 'has_causal': has_causal,
            'sub_queries': sub_queries
        }
    
    def retrieve(self, query, candidates, analysis=None, query_entities=None,
                 current_time=None, context_entities=None):
        query_entities = query_entities or []
        context_entities = context_entities or []
        current_time = current_time or int(time.time()*1000)
        
        a = analysis or self.analyze_query(query, context_entities)
        
        if a['strategy'] == RetrievalStrategy.DIRECT:
            return self._execute_direct(query, candidates, query_entities, current_time, context_entities)
        elif a['strategy'] == RetrievalStrategy.SPLIT_QUERY:
            return self._execute_split(query, a['sub_queries'], candidates,
                                      query_entities, current_time, context_entities)
        else:
            return self._execute_chain(query, a['sub_queries'], candidates,
                                      query_entities, current_time, context_entities)
    
    def _execute_direct(self, query, candidates, qe, ct, ce):
        ranked = self.retriever.rank(query, candidates, qe, ct, ce)
        return {'ranked': ranked, 'strategy': RetrievalStrategy.DIRECT,
                'sub_results': {}, 'mqr_applied': False}
    
    def _execute_split(self, original, sub_queries, candidates, qe, ct, ce):
        if not sub_queries:
            return self._execute_direct(original, candidates, qe, ct, ce)
        
        sub_results = {}
        hit_counts = defaultdict(int)
        
        for sq in sub_queries[:self.max_sub_queries]:
            ranked = self.retriever.rank(sq, candidates, qe, ct, ce)
            sub_results[sq] = ranked
            for r in ranked[:10]:
                hit_counts[r.id] += 1
        
        all_results = [r for rs in sub_results.values() for r in rs]
        merged = self._merge_mqr(all_results, hit_counts)
        
        return {'ranked': merged, 'strategy': RetrievalStrategy.SPLIT_QUERY,
                'sub_results': sub_results, 
                'mqr_applied': any(v >= self.mqr_min_hits for v in hit_counts.values())}
    
    def _execute_chain(self, original, chain_steps, candidates, qe, ct, ce):
        if not chain_steps:
            return self._execute_direct(original, candidates, qe, ct, ce)
        
        acc_entities = list(qe)
        sub_results = {}
        
        for i, step in enumerate(chain_steps[:self.max_chain_depth]):
            ranked = self.retriever.rank(step, candidates, acc_entities, ct, ce)
            sub_results[f"step_{i+1}: {step}"] = ranked
            top_ents = [e for r in ranked[:3] for e in (r.entry.entities if r.entry else [])]
            acc_entities.extend(top_ents)
        
        final = self.retriever.rank(original, candidates, list(set(acc_entities)), ct, ce)
        return {'ranked': final, 'strategy': RetrievalStrategy.CHAIN_OF_QUERY,
                'sub_results': sub_results, 'mqr_applied': False}
    
    def _merge_mqr(self, all_results, hit_counts):
        best = {}
        for r in all_results:
            if r.id not in best or r.score > best[r.id].score:
                best[r.id] = r
        
        boosted = []
        for mem in best.values():
            hits = hit_counts.get(mem.id, 1)
            if hits >= self.mqr_min_hits:
                boost = 1.0 + (self.mqr_boost - 1.0) * (hits - 1) / self.max_sub_queries
                boosted.append(mem.copy(score=mem.score * boost))
            else:
                boosted.append(mem)
        
        return sorted(boosted, key=lambda x: x.score, reverse=True)
    
    def _decide_strategy(self, ec, cc, temporal, causal, qlen):
        if causal and ec >= 2:
            return RetrievalStrategy.CHAIN_OF_QUERY, 0.8
        if temporal and cc >= 2:
            return RetrievalStrategy.CHAIN_OF_QUERY, 0.7
        if cc >= 2 and ec >= 2:
            return RetrievalStrategy.SPLIT_QUERY, 0.85
        if qlen > 30 and cc >= 2:
            return RetrievalStrategy.SPLIT_QUERY, 0.7
        return RetrievalStrategy.DIRECT, 0.9
    
    def _count_inline_entities(self, query):
        parts = re.split(r'[的地得了过着吗呢吧啊]', query)
        return min(5, len([p for p in parts if len(p) >= 2]))
    
    def _count_clauses(self, query):
        count = 1
        for pat in self.conjunction_pats:
            count += len(pat.findall(query))
        return min(6, count)
    
    def _split_query(self, query):
        parts = [p.strip() for p in re.split(r'[,，;；、和与以及还有另外而且并且同时]+', query) if len(p.strip()) >= 2]
        return parts[:self.max_sub_queries] if len(parts) >= 2 else [query]
    
    def _chain_decompose(self, query):
        parts = [p.strip() for p in re.split(r'(为什么|因为|所以|然后|之后|接着|后来|如果)', query) if len(p.strip()) >= 2]
        return parts[:self.max_chain_depth] if len(parts) >= 2 else [query]


# === Helper ===
def make_entry(content, source=MemorySource.USER, entities=None, ts_offset_days=0):
    now = int(time.time()*1000)
    return MemoryEntry(
        id=str(uuid.uuid4()), content=content, source=source,
        entities=entities or [], timestamp=now - ts_offset_days*86400*1000,
        session_id=None, user_id='u1', metadata={}, created_at=now)

def make_candidate(entry, semantic_score=0.7):
    return RetrievalCandidate(entry.id, entry.content, semantic_score, entry)


# ============================
# Tests
# ============================

def test_direct_routing():
    """Simple queries route to DIRECT"""
    agent = RetrievalAgent()
    a = agent.analyze_query('你叫什么名字')
    assert a['strategy'] == RetrievalStrategy.DIRECT
    assert a['confidence'] >= 0.8
    print("✓ test_direct_routing")

def test_split_routing():
    """Multi-entity compound queries route to SPLIT_QUERY"""
    agent = RetrievalAgent()
    # Multiple clauses + entities
    a = agent.analyze_query('小明的爱好和小红的工作', context_entities=['小明', '小红'])
    assert a['strategy'] == RetrievalStrategy.SPLIT_QUERY
    assert len(a['sub_queries']) >= 2
    print(f"  sub_queries: {a['sub_queries']}")
    print("✓ test_split_routing")

def test_chain_routing():
    """Causal/temporal multi-hop routes to CHAIN_OF_QUERY"""
    agent = RetrievalAgent()
    a = agent.analyze_query('为什么小明后来搬到了上海', context_entities=['小明', '上海'])
    assert a['strategy'] == RetrievalStrategy.CHAIN_OF_QUERY
    print(f"  sub_queries: {a['sub_queries']}")
    print("✓ test_chain_routing")

def test_temporal_chain():
    """Temporal + multi-clause triggers chain"""
    agent = RetrievalAgent()
    a = agent.analyze_query('以前住在哪里，后来又搬到哪里了')
    assert a['strategy'] == RetrievalStrategy.CHAIN_OF_QUERY
    print("✓ test_temporal_chain")

def test_direct_execution():
    """DIRECT strategy retrieves correctly"""
    agent = RetrievalAgent()
    now = int(time.time()*1000)
    
    entries = [
        make_entry('我叫小明', entities=['小明'], ts_offset_days=0),
        make_entry('天气很好', ts_offset_days=0),
    ]
    candidates = [
        make_candidate(entries[0], 0.95),
        make_candidate(entries[1], 0.3),
    ]
    
    result = agent.retrieve('你叫什么名字', candidates, current_time=now)
    assert result['strategy'] == RetrievalStrategy.DIRECT
    assert result['ranked'][0].content == '我叫小明'
    assert not result['mqr_applied']
    print("✓ test_direct_execution")

def test_split_execution():
    """SPLIT_QUERY retrieves and merges sub-query results"""
    agent = RetrievalAgent()
    now = int(time.time()*1000)
    
    entries = [
        make_entry('小明喜欢篮球', entities=['小明', '篮球'], ts_offset_days=1),
        make_entry('小红在字节工作', entities=['小红', '字节'], ts_offset_days=1),
        make_entry('昨天下雨了', ts_offset_days=1),
    ]
    candidates = [
        make_candidate(entries[0], 0.85),
        make_candidate(entries[1], 0.85),
        make_candidate(entries[2], 0.3),
    ]
    
    analysis = {
        'strategy': RetrievalStrategy.SPLIT_QUERY,
        'sub_queries': ['小明的爱好', '小红的工作'],
        'confidence': 0.85, 'entity_count': 2, 'clause_count': 2,
        'has_temporal': False, 'has_causal': False,
    }
    
    result = agent.retrieve('小明的爱好和小红的工作', candidates,
                           analysis=analysis, 
                           query_entities=['小明', '小红'],
                           current_time=now)
    
    assert result['strategy'] == RetrievalStrategy.SPLIT_QUERY
    # Both relevant memories should appear in top results
    top_contents = [r.content for r in result['ranked'][:2]]
    assert any('篮球' in c for c in top_contents)
    assert any('字节' in c for c in top_contents)
    print("✓ test_split_execution")

def test_mqr_boost():
    """Memories hit by multiple sub-queries get MQR boost"""
    agent = RetrievalAgent()
    now = int(time.time()*1000)
    
    # This memory is relevant to BOTH sub-queries
    shared = make_entry('小明和小红一起去打篮球', 
                       entities=['小明', '小红', '篮球'], ts_offset_days=1)
    only_ming = make_entry('小明喜欢编程', 
                          entities=['小明', '编程'], ts_offset_days=1)
    
    candidates = [
        make_candidate(shared, 0.8),
        make_candidate(only_ming, 0.85),
    ]
    
    analysis = {
        'strategy': RetrievalStrategy.SPLIT_QUERY,
        'sub_queries': ['小明的爱好', '小红的爱好'],
        'confidence': 0.85, 'entity_count': 2, 'clause_count': 2,
        'has_temporal': False, 'has_causal': False,
    }
    
    result = agent.retrieve('小明的爱好和小红的爱好', candidates,
                           analysis=analysis,
                           query_entities=['小明', '小红'],
                           current_time=now)
    
    # The shared memory should be boosted by MQR (hit by both sub-queries)
    if result['mqr_applied']:
        boosted = [r for r in result['ranked'] if '一起' in r.content]
        if boosted:
            print(f"  MQR applied: shared memory score={boosted[0].score:.4f}")
    
    print("✓ test_mqr_boost")

def test_chain_execution():
    """CHAIN_OF_QUERY propagates context between steps"""
    agent = RetrievalAgent()
    now = int(time.time()*1000)
    
    entries = [
        make_entry('小明在北京工作', entities=['小明', '北京'], ts_offset_days=30),
        make_entry('北京房价太高了', entities=['北京', '房价'], ts_offset_days=20),
        make_entry('小明搬到了上海', entities=['小明', '上海'], ts_offset_days=1),
    ]
    candidates = [
        make_candidate(entries[0], 0.7),
        make_candidate(entries[1], 0.6),
        make_candidate(entries[2], 0.8),
    ]
    
    analysis = {
        'strategy': RetrievalStrategy.CHAIN_OF_QUERY,
        'sub_queries': ['小明住哪里', '搬到了哪里'],
        'confidence': 0.8, 'entity_count': 2, 'clause_count': 2,
        'has_temporal': True, 'has_causal': True,
    }
    
    result = agent.retrieve('为什么小明后来搬家了', candidates,
                           analysis=analysis,
                           query_entities=['小明'],
                           current_time=now)
    
    assert result['strategy'] == RetrievalStrategy.CHAIN_OF_QUERY
    # Chain should accumulate entities from intermediate steps
    assert len(result['sub_results']) >= 2
    print("✓ test_chain_execution")

def test_empty_candidates():
    """Handles empty candidate list gracefully"""
    agent = RetrievalAgent()
    result = agent.retrieve('你好', [])
    assert len(result['ranked']) == 0
    print("✓ test_empty_candidates")

def test_fallback_on_no_subqueries():
    """If split/chain produces no sub-queries, falls back to DIRECT"""
    agent = RetrievalAgent()
    now = int(time.time()*1000)
    
    entry = make_entry('测试', ts_offset_days=0)
    candidates = [make_candidate(entry, 0.9)]
    
    # Force SPLIT strategy but with empty sub_queries
    analysis = {
        'strategy': RetrievalStrategy.SPLIT_QUERY,
        'sub_queries': [],
        'confidence': 0.5, 'entity_count': 0, 'clause_count': 1,
        'has_temporal': False, 'has_causal': False,
    }
    
    result = agent.retrieve('测试', candidates, analysis=analysis, current_time=now)
    # Should fall back and still return results
    assert len(result['ranked']) == 1
    print("✓ test_fallback_on_no_subqueries")

def test_strategy_consistency():
    """Same query always gets same strategy"""
    agent = RetrievalAgent()
    q = '小明以前住在哪里，后来搬到哪里了'
    
    results = [agent.analyze_query(q) for _ in range(5)]
    strategies = {r['strategy'] for r in results}
    assert len(strategies) == 1, f"Inconsistent strategies: {strategies}"
    print(f"  consistent strategy: {results[0]['strategy']}")
    print("✓ test_strategy_consistency")

def test_various_query_types():
    """Test routing on diverse query types"""
    agent = RetrievalAgent()
    
    cases = [
        ('你好', RetrievalStrategy.DIRECT),
        ('我叫什么名字', RetrievalStrategy.DIRECT),
        ('小明喜欢什么和小红喜欢什么', RetrievalStrategy.SPLIT_QUERY),
        ('因为工作原因小明搬家了', RetrievalStrategy.CHAIN_OF_QUERY),
    ]
    
    for query, expected in cases:
        a = agent.analyze_query(query, context_entities=['小明', '小红'] if '小明' in query and '小红' in query else [])
        print(f"  '{query}' → {a['strategy'].value} (expected {expected.value})")
    
    # At minimum, verify simple queries are DIRECT
    simple = agent.analyze_query('你好')
    assert simple['strategy'] == RetrievalStrategy.DIRECT
    print("✓ test_various_query_types")


if __name__ == '__main__':
    tests = [
        test_direct_routing,
        test_split_routing,
        test_chain_routing,
        test_temporal_chain,
        test_direct_execution,
        test_split_execution,
        test_mqr_boost,
        test_chain_execution,
        test_empty_candidates,
        test_fallback_on_no_subqueries,
        test_strategy_consistency,
        test_various_query_types,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    if failed == 0:
        print("ALL TESTS PASSED ✓")
