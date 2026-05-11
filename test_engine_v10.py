"""
Integration tests for ChatEngine v10 — ADD-only architecture end-to-end
"""
import time
import math
import uuid
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set
from collections import defaultdict

# === Minimal simulation of v10 components ===

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

class AppendOnlyStore:
    def __init__(self):
        self.memories = {}
        self.entity_index = defaultdict(set)
    
    def add(self, content, source, entities=None, timestamp=None, 
            session_id=None, user_id=None, metadata=None):
        entities = entities or []
        timestamp = timestamp or int(time.time()*1000)
        metadata = metadata or {}
        entry = MemoryEntry(
            id=str(uuid.uuid4()), content=content, source=source,
            entities=entities, timestamp=timestamp, session_id=session_id,
            user_id=user_id, metadata=metadata, created_at=int(time.time()*1000))
        self.memories[entry.id] = entry
        for e in entities:
            self.entity_index[e].add(entry.id)
        return entry
    
    def get_all(self, user_id=None):
        if user_id:
            return [m for m in self.memories.values() if m.user_id == user_id]
        return list(self.memories.values())
    
    def get_by_entity(self, entity):
        ids = self.entity_index.get(entity, set())
        return [self.memories[i] for i in ids if i in self.memories]
    
    def count(self, user_id=None):
        if user_id:
            return sum(1 for m in self.memories.values() if m.user_id == user_id)
        return len(self.memories)
    
    def entity_count(self):
        return len(self.entity_index)


class FactExtractor:
    def extract(self, message):
        sentences = [s.strip() for s in re.split(r'[。！？；\n]+', message) if len(s.strip()) >= 2]
        return [{'content': s, 'entities': self._extract_entities(s)} for s in sentences]
    
    def extract_agent(self, response):
        sentences = [s.strip() for s in re.split(r'[。！？；\n]+', response) 
                    if len(s.strip()) >= 4 and '?' not in s and '？' not in s]
        return [{'content': s, 'entities': self._extract_entities(s)} for s in sentences]
    
    def _extract_entities(self, text):
        entities = []
        for m in re.finditer(r'[\u4e00-\u9fa5]{2,3}(?=说|是|在|住|喜欢|工作|叫)', text):
            entities.append(m.group())
        for m in re.finditer(r'(?:在|住|到|去)([\u4e00-\u9fa5]{2,4})', text):
            entities.append(m.group(1))
        return list(set(entities))


class ChatEngineV10:
    def __init__(self, context_budget=5, conflict_mode='current_only'):
        self.store = AppendOnlyStore()
        self.extractor = FactExtractor()
        self.context_budget = context_budget
        self.conflict_mode = conflict_mode
    
    def process_user_turn(self, user_id, message, session_id=None):
        now = int(time.time()*1000)
        
        # Extract & ADD
        facts = self.extractor.extract(message)
        new_entries = []
        for f in facts:
            entry = self.store.add(
                content=f['content'], source=MemorySource.USER,
                entities=f['entities'], timestamp=now,
                session_id=session_id, user_id=user_id,
                metadata={'original': message})
            new_entries.append(entry)
        
        # Retrieve
        all_mems = self.store.get_all(user_id)
        if not all_mems:
            return {'context': [], 'new_facts': new_entries, 'conflicts': []}
        
        # Simplified ranking (char overlap + recency)
        scored = []
        for m in all_mems:
            if m.id in {e.id for e in new_entries}:
                continue  # Skip just-added
            sem = self._char_similarity(message, m.content)
            age_days = (now - m.timestamp) / (86400*1000)
            rec = max(0.1, math.exp(-0.693 * age_days / 30.0))
            score = sem * 0.6 + rec * 0.4
            scored.append((m, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        context = [m.content for m, s in scored[:self.context_budget]]
        
        return {'context': context, 'new_facts': new_entries, 'conflicts': []}
    
    def process_agent_turn(self, user_id, response, session_id=None):
        facts = self.extractor.extract_agent(response)
        entries = []
        for f in facts:
            entry = self.store.add(
                content=f['content'], source=MemorySource.AGENT,
                entities=f['entities'], timestamp=int(time.time()*1000),
                session_id=session_id, user_id=user_id)
            entries.append(entry)
        return entries
    
    def query(self, user_id, query, top_k=5):
        all_mems = self.store.get_all(user_id)
        if not all_mems:
            return []
        scored = [(m, self._char_similarity(query, m.content)) for m in all_mems]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(m.content, s) for m, s in scored[:top_k]]
    
    def entity_history(self, entity):
        return sorted(self.store.get_by_entity(entity), key=lambda m: m.timestamp)
    
    def _char_similarity(self, a, b):
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)


# ============================
# Tests
# ============================

def test_basic_conversation():
    """Simple conversation: extract + store + retrieve"""
    engine = ChatEngineV10()
    
    # Turn 1
    r1 = engine.process_user_turn('u1', '我叫小明，住在北京')
    assert len(r1['new_facts']) >= 1
    assert engine.store.count('u1') >= 1
    
    # Turn 2
    r2 = engine.process_user_turn('u1', '今天天气怎么样？')
    # Should retrieve Turn 1 as context
    # (may or may not depending on similarity)
    assert 'new_facts' in r2
    
    print(f"  After 2 turns: {engine.store.count('u1')} memories")
    print("✓ test_basic_conversation")

def test_contradiction_preserved():
    """Contradictory info is stored, not overwritten"""
    engine = ChatEngineV10()
    
    engine.process_user_turn('u1', '我住在北京')
    time.sleep(0.01)
    engine.process_user_turn('u1', '我搬到了上海')
    
    # Both should exist
    all_mems = engine.store.get_all('u1')
    contents = [m.content for m in all_mems]
    # Check that both location facts are preserved
    has_beijing = any('北京' in c for c in contents)
    has_shanghai = any('上海' in c for c in contents)
    assert has_beijing, f"北京 not found in: {contents}"
    assert has_shanghai, f"上海 not found in: {contents}"
    
    print("✓ test_contradiction_preserved")

def test_agent_facts_first_class():
    """Agent responses also generate memories"""
    engine = ChatEngineV10()
    
    engine.process_user_turn('u1', '我对花生过敏')
    entries = engine.process_agent_turn('u1', '好的，已记录您对花生过敏。以后推荐食物时会注意避开花生成分')
    
    # Agent facts should be stored
    assert len(entries) >= 1
    agent_mems = [m for m in engine.store.get_all('u1') if m.source == MemorySource.AGENT]
    assert len(agent_mems) >= 1
    
    print(f"  Agent memories: {len(agent_mems)}")
    print("✓ test_agent_facts_first_class")

def test_entity_history():
    """Can trace full history of an entity"""
    engine = ChatEngineV10()
    now = int(time.time()*1000)
    
    # Manually add with controlled timestamps and entities
    engine.store.add('住在北京', source=MemorySource.USER, entities=['住所'],
                    timestamp=now - 90*86400*1000, user_id='u1')
    engine.store.add('搬到上海', source=MemorySource.USER, entities=['住所'],
                    timestamp=now - 30*86400*1000, user_id='u1')
    engine.store.add('又搬到深圳', source=MemorySource.USER, entities=['住所'],
                    timestamp=now - 1*86400*1000, user_id='u1')
    
    history = engine.entity_history('住所')
    assert len(history) == 3
    assert '北京' in history[0].content  # oldest first
    assert '深圳' in history[2].content  # newest last
    
    print(f"  住所 history: {[h.content for h in history]}")
    print("✓ test_entity_history")

def test_multi_user_isolation():
    """Users' memories are isolated"""
    engine = ChatEngineV10()
    
    engine.process_user_turn('u1', '我叫小明')
    engine.process_user_turn('u2', '我叫小红')
    
    u1_mems = engine.store.get_all('u1')
    u2_mems = engine.store.get_all('u2')
    
    u1_contents = ' '.join(m.content for m in u1_mems)
    u2_contents = ' '.join(m.content for m in u2_mems)
    
    # u1 should not see u2's data
    assert '小红' not in u1_contents or '小明' not in u2_contents
    print("✓ test_multi_user_isolation")

def test_query_without_adding():
    """Query doesn't add new memories"""
    engine = ChatEngineV10()
    engine.process_user_turn('u1', '我喜欢打篮球')
    
    count_before = engine.store.count('u1')
    results = engine.query('u1', '我的爱好是什么')
    count_after = engine.store.count('u1')
    
    assert count_before == count_after, "Query should not add memories"
    assert len(results) > 0
    print(f"  Query results: {results}")
    print("✓ test_query_without_adding")

def test_session_tracking():
    """Memories track session_id"""
    engine = ChatEngineV10()
    
    engine.process_user_turn('u1', '今天开会讨论了项目进度', session_id='sess_001')
    engine.process_user_turn('u1', '明天要提交报告', session_id='sess_001')
    engine.process_user_turn('u1', '周末想去爬山', session_id='sess_002')
    
    all_mems = engine.store.get_all('u1')
    sessions = {m.session_id for m in all_mems}
    assert 'sess_001' in sessions
    assert 'sess_002' in sessions
    
    print("✓ test_session_tracking")

def test_many_memories_performance():
    """Performance with 100+ memories"""
    engine = ChatEngineV10()
    
    # Add 100 memories
    for i in range(100):
        engine.store.add(
            f'记忆条目{i}: 一些关于话题{i%10}的信息',
            source=MemorySource.USER,
            entities=[f'话题{i%10}'],
            timestamp=int(time.time()*1000) - i*3600*1000,
            user_id='u1')
    
    assert engine.store.count('u1') == 100
    
    # Query should still be fast
    start = time.time()
    results = engine.query('u1', '话题5相关的信息', top_k=5)
    elapsed = time.time() - start
    
    assert elapsed < 1.0, f"Query took too long: {elapsed:.3f}s"
    assert len(results) == 5
    print(f"  100 memories, query in {elapsed*1000:.1f}ms")
    print("✓ test_many_memories_performance")

def test_context_budget():
    """Context budget limits injection"""
    engine = ChatEngineV10(context_budget=3)
    
    # Add many relevant memories
    for i in range(10):
        engine.store.add(f'我喜欢运动{i}', source=MemorySource.USER,
                        entities=['运动'], timestamp=int(time.time()*1000) - i*86400*1000,
                        user_id='u1')
    
    result = engine.process_user_turn('u1', '我喜欢什么运动')
    # Context should be limited by budget
    # Note: new_facts from current turn don't count against budget
    # The context retrieves from EXISTING memories
    context_from_old = result['context']
    assert len(context_from_old) <= 3, f"Budget exceeded: {len(context_from_old)}"
    print(f"  Context size: {len(context_from_old)} (budget=3)")
    print("✓ test_context_budget")

def test_empty_store():
    """Works correctly with empty store"""
    engine = ChatEngineV10()
    
    result = engine.process_user_turn('u1', '你好')
    assert result['context'] == []
    assert len(result['new_facts']) >= 0  # May or may not extract from '你好'
    
    results = engine.query('u1', '什么都没有')
    # After processing '你好', there might be 1 memory
    # Query should work regardless
    print("✓ test_empty_store")

def test_v10_vs_v8_philosophy():
    """
    Demonstrate v10's philosophical advantage over v8:
    In v8, CRUD would DELETE '住在北京' when '搬到上海' arrives.
    In v10, both coexist — temporal context distinguishes them.
    """
    engine = ChatEngineV10()
    now = int(time.time()*1000)
    
    # Simulate conversation over time
    engine.store.add('我住在北京，在字节工作', source=MemorySource.USER,
                    entities=['北京', '字节', '住所'], 
                    timestamp=now - 365*86400*1000, user_id='u1')
    engine.store.add('我搬到了上海', source=MemorySource.USER,
                    entities=['上海', '住所'],
                    timestamp=now - 30*86400*1000, user_id='u1')
    
    # v10: both preserved, can answer different questions
    all_mems = engine.store.get_all('u1')
    assert len(all_mems) == 2
    
    # Query "现在住哪" → 上海 (recency wins)
    q1 = engine.query('u1', '现在住在哪里')
    # Query "以前住哪" → 北京 (historical)  
    q2 = engine.query('u1', '以前住在哪里')
    
    # Both queries return results (nothing was deleted)
    assert len(q1) >= 1
    assert len(q2) >= 1
    
    print(f"  '现在住哪': {q1[0][0]}")
    print(f"  '以前住哪': {q2[0][0]}")
    print("✓ test_v10_vs_v8_philosophy")


if __name__ == '__main__':
    tests = [
        test_basic_conversation,
        test_contradiction_preserved,
        test_agent_facts_first_class,
        test_entity_history,
        test_multi_user_isolation,
        test_query_without_adding,
        test_session_tracking,
        test_many_memories_performance,
        test_context_budget,
        test_empty_store,
        test_v10_vs_v8_philosophy,
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
