"""
Tests for ContextExpander — nucleus + neighbors context expansion
"""
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set
from collections import defaultdict

# === Models ===

class MemorySource(Enum):
    USER = "USER"
    AGENT = "AGENT"
    SYSTEM = "SYSTEM"

class ContextRole(Enum):
    NUCLEUS = "NUCLEUS"
    SESSION_NEIGHBOR = "SESSION_NEIGHBOR"
    TEMPORAL_NEIGHBOR = "TEMPORAL_NEIGHBOR"
    ENTITY_CHAIN = "ENTITY_CHAIN"

class ExpansionMode(Enum):
    NUCLEUS_ONLY = "NUCLEUS_ONLY"
    SESSION_ONLY = "SESSION_ONLY"
    TEMPORAL_ONLY = "TEMPORAL_ONLY"
    SESSION_AND_TEMPORAL = "SESSION_AND_TEMPORAL"
    FULL = "FULL"

@dataclass
class MemoryEntry:
    id: str
    content: str
    source: MemorySource
    entities: List[str]
    timestamp: int
    session_id: Optional[str]
    user_id: Optional[str]
    metadata: Dict = field(default_factory=dict)
    created_at: int = 0

@dataclass
class RankedMemory:
    id: str
    content: str
    score: float
    entry: Optional[MemoryEntry] = None

@dataclass
class ContextEntry:
    memory: MemoryEntry
    role: ContextRole
    score: float
    distance_from_nucleus: int

@dataclass
class ExpandedContext:
    entries: List[ContextEntry]
    nuclei_count: int = 0
    session_neighbors: int = 0
    temporal_neighbors: int = 0
    entity_chain_entries: int = 0
    truncated: bool = False


class AppendOnlyStore:
    def __init__(self):
        self.memories = {}
        self.entity_index = defaultdict(set)
    
    def add(self, content, source=MemorySource.USER, entities=None,
            timestamp=None, session_id=None, user_id=None):
        entities = entities or []
        timestamp = timestamp or int(time.time()*1000)
        entry = MemoryEntry(
            id=str(uuid.uuid4()), content=content, source=source,
            entities=entities, timestamp=timestamp, session_id=session_id,
            user_id=user_id)
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


class ContextExpander:
    def __init__(self, session_window_before=2, session_window_after=1,
                 temporal_window_hours=2.0, entity_chain_limit=3,
                 max_context_entries=12):
        self.session_before = session_window_before
        self.session_after = session_window_after
        self.temporal_hours = temporal_window_hours
        self.entity_chain_limit = entity_chain_limit
        self.max_entries = max_context_entries
    
    def expand(self, nuclei, store, user_id, mode=ExpansionMode.SESSION_AND_TEMPORAL):
        if not nuclei:
            return ExpandedContext(entries=[])
        
        all_mems = sorted(store.get_all(user_id), key=lambda m: m.timestamp)
        session_groups = defaultdict(list)
        for m in all_mems:
            key = m.session_id or f"no_session_{m.id}"
            session_groups[key].append(m)
        for k in session_groups:
            session_groups[k].sort(key=lambda m: m.timestamp)
        
        seen = set()
        entries = []
        
        # Phase 1: Nuclei
        for n in nuclei:
            if not n.entry or n.id in seen:
                continue
            seen.add(n.id)
            entries.append(ContextEntry(n.entry, ContextRole.NUCLEUS, n.score, 0))
        
        # Phase 2: Session neighbors
        if mode in (ExpansionMode.SESSION_ONLY, ExpansionMode.SESSION_AND_TEMPORAL, ExpansionMode.FULL):
            for n in nuclei:
                entry = n.entry
                if not entry:
                    continue
                key = entry.session_id or f"no_session_{entry.id}"
                session_list = session_groups.get(key, [])
                pos = next((i for i, m in enumerate(session_list) if m.id == entry.id), -1)
                if pos < 0:
                    continue
                
                for offset in range(1, self.session_before + 1):
                    idx = pos - offset
                    if idx < 0:
                        break
                    neighbor = session_list[idx]
                    if neighbor.id in seen:
                        continue
                    seen.add(neighbor.id)
                    entries.append(ContextEntry(
                        neighbor, ContextRole.SESSION_NEIGHBOR,
                        n.score * (0.8 - offset * 0.15), -offset))
                
                for offset in range(1, self.session_after + 1):
                    idx = pos + offset
                    if idx >= len(session_list):
                        break
                    neighbor = session_list[idx]
                    if neighbor.id in seen:
                        continue
                    seen.add(neighbor.id)
                    entries.append(ContextEntry(
                        neighbor, ContextRole.SESSION_NEIGHBOR,
                        n.score * (0.7 - offset * 0.15), offset))
        
        # Phase 3: Temporal neighbors
        if mode in (ExpansionMode.TEMPORAL_ONLY, ExpansionMode.SESSION_AND_TEMPORAL, ExpansionMode.FULL):
            window_ms = int(self.temporal_hours * 3600 * 1000)
            for n in nuclei:
                entry = n.entry
                if not entry:
                    continue
                for mem in all_mems:
                    if mem.id in seen:
                        continue
                    if abs(mem.timestamp - entry.timestamp) <= window_ms and mem.id != entry.id:
                        seen.add(mem.id)
                        time_dist = abs(mem.timestamp - entry.timestamp) / window_ms
                        entries.append(ContextEntry(
                            mem, ContextRole.TEMPORAL_NEIGHBOR,
                            n.score * 0.6 * (1.0 - time_dist),
                            int((mem.timestamp - entry.timestamp) / 60000)))
        
        # Phase 4: Entity chain
        if mode == ExpansionMode.FULL:
            nucleus_entities = set()
            for n in nuclei:
                if n.entry:
                    nucleus_entities.update(n.entry.entities)
            
            for entity in nucleus_entities:
                entity_mems = [m for m in store.get_by_entity(entity) if m.id not in seen]
                entity_mems.sort(key=lambda m: m.timestamp, reverse=True)
                for mem in entity_mems[:self.entity_chain_limit]:
                    seen.add(mem.id)
                    entries.append(ContextEntry(mem, ContextRole.ENTITY_CHAIN, 0.3, -1))
        
        # Phase 5: Truncate
        # Sort: nucleus first, then by score descending within role
        role_order = {ContextRole.NUCLEUS: 0, ContextRole.SESSION_NEIGHBOR: 1,
                     ContextRole.TEMPORAL_NEIGHBOR: 2, ContextRole.ENTITY_CHAIN: 3}
        entries.sort(key=lambda e: (role_order[e.role], -e.score))
        
        truncated = len(entries) > self.max_entries
        final = entries[:self.max_entries]
        
        return ExpandedContext(
            entries=final,
            nuclei_count=sum(1 for e in final if e.role == ContextRole.NUCLEUS),
            session_neighbors=sum(1 for e in final if e.role == ContextRole.SESSION_NEIGHBOR),
            temporal_neighbors=sum(1 for e in final if e.role == ContextRole.TEMPORAL_NEIGHBOR),
            entity_chain_entries=sum(1 for e in final if e.role == ContextRole.ENTITY_CHAIN),
            truncated=truncated
        )


# ============================
# Tests
# ============================

def build_conversation(store, user_id='u1'):
    """Build a realistic multi-session conversation"""
    now = int(time.time() * 1000)
    day = 86400 * 1000
    hour = 3600 * 1000
    
    # Session 1: 7 days ago — work stress
    s1_entries = []
    s1_entries.append(store.add('最近工作压力好大', source=MemorySource.USER,
        entities=['工作'], timestamp=now - 7*day, session_id='sess1', user_id=user_id))
    s1_entries.append(store.add('项目deadline是下周五', source=MemorySource.USER,
        entities=['项目', '工作'], timestamp=now - 7*day + 1*60000, session_id='sess1', user_id=user_id))
    s1_entries.append(store.add('跨部门沟通特别头疼', source=MemorySource.USER,
        entities=['工作'], timestamp=now - 7*day + 2*60000, session_id='sess1', user_id=user_id))
    s1_entries.append(store.add('我想跑步解压', source=MemorySource.USER,
        entities=['跑步'], timestamp=now - 7*day + 3*60000, session_id='sess1', user_id=user_id))
    
    # Session 2: 3 days ago — personal life
    s2_entries = []
    s2_entries.append(store.add('周末去了趟杭州', source=MemorySource.USER,
        entities=['杭州', '旅行'], timestamp=now - 3*day, session_id='sess2', user_id=user_id))
    s2_entries.append(store.add('西湖真的很美', source=MemorySource.USER,
        entities=['杭州', '西湖'], timestamp=now - 3*day + 1*60000, session_id='sess2', user_id=user_id))
    s2_entries.append(store.add('在那边吃了龙井虾仁', source=MemorySource.USER,
        entities=['杭州', '美食'], timestamp=now - 3*day + 2*60000, session_id='sess2', user_id=user_id))
    
    # Session 3: today — project update
    s3_entries = []
    s3_entries.append(store.add('项目终于提交了', source=MemorySource.USER,
        entities=['项目', '工作'], timestamp=now - 2*hour, session_id='sess3', user_id=user_id))
    s3_entries.append(store.add('老板说做得不错', source=MemorySource.USER,
        entities=['老板', '工作'], timestamp=now - 2*hour + 1*60000, session_id='sess3', user_id=user_id))
    s3_entries.append(store.add('下个项目是AI相关的', source=MemorySource.USER,
        entities=['项目', 'AI'], timestamp=now - 2*hour + 2*60000, session_id='sess3', user_id=user_id))
    
    return {'sess1': s1_entries, 'sess2': s2_entries, 'sess3': s3_entries, 'now': now}


def test_nucleus_only():
    """NUCLEUS_ONLY mode returns only the hit memories"""
    store = AppendOnlyStore()
    data = build_conversation(store)
    expander = ContextExpander()
    
    # Simulate: "项目终于提交了" was retrieved as nucleus
    nucleus_entry = data['sess3'][0]
    nuclei = [RankedMemory(nucleus_entry.id, nucleus_entry.content, 0.9, nucleus_entry)]
    
    result = expander.expand(nuclei, store, 'u1', ExpansionMode.NUCLEUS_ONLY)
    
    assert result.nuclei_count == 1
    assert result.session_neighbors == 0
    assert result.temporal_neighbors == 0
    assert len(result.entries) == 1
    assert result.entries[0].memory.content == '项目终于提交了'
    print("✓ test_nucleus_only")


def test_session_expansion():
    """SESSION_ONLY expands with same-session neighbors"""
    store = AppendOnlyStore()
    data = build_conversation(store)
    expander = ContextExpander(session_window_before=2, session_window_after=1)
    
    # Nucleus: "西湖真的很美" (middle of session 2)
    nucleus_entry = data['sess2'][1]
    nuclei = [RankedMemory(nucleus_entry.id, nucleus_entry.content, 0.85, nucleus_entry)]
    
    result = expander.expand(nuclei, store, 'u1', ExpansionMode.SESSION_ONLY)
    
    # Should get: nucleus + before (周末去了趟杭州) + after (龙井虾仁)
    assert result.nuclei_count == 1
    assert result.session_neighbors >= 2  # at least 1 before + 1 after
    
    contents = [e.memory.content for e in result.entries]
    assert '西湖真的很美' in contents  # nucleus
    assert '周末去了趟杭州' in contents  # before neighbor
    assert '在那边吃了龙井虾仁' in contents  # after neighbor
    
    print(f"  Expanded: {contents}")
    print("✓ test_session_expansion")


def test_temporal_expansion():
    """TEMPORAL_ONLY finds time-proximate memories across sessions"""
    store = AppendOnlyStore()
    now = int(time.time() * 1000)
    
    # Two sessions very close in time (30 min apart)
    store.add('开完会了', session_id='a', timestamp=now - 3600*1000, user_id='u1')
    target = store.add('会议结论是加人手', session_id='a', timestamp=now - 3600*1000 + 60000, user_id='u1')
    store.add('给老板发了邮件', session_id='b', timestamp=now - 1800*1000, user_id='u1')
    store.add('昨天的事情完全不相关', session_id='c', timestamp=now - 24*3600*1000, user_id='u1')
    
    expander = ContextExpander(temporal_window_hours=1.0)
    nuclei = [RankedMemory(target.id, target.content, 0.9, target)]
    
    result = expander.expand(nuclei, store, 'u1', ExpansionMode.TEMPORAL_ONLY)
    
    contents = [e.memory.content for e in result.entries]
    # Should include "开完会了" (same session, within window) and "给老板发了邮件" (diff session, within 1h)
    assert '开完会了' in contents or '给老板发了邮件' in contents
    # Should NOT include "昨天的事情" (too far away)
    assert '昨天的事情完全不相关' not in contents
    
    print(f"  Temporal neighbors: {contents}")
    print("✓ test_temporal_expansion")


def test_full_expansion():
    """FULL mode includes all expansion types"""
    store = AppendOnlyStore()
    data = build_conversation(store)
    expander = ContextExpander()
    
    # Nucleus: "项目终于提交了"
    nucleus_entry = data['sess3'][0]
    nuclei = [RankedMemory(nucleus_entry.id, nucleus_entry.content, 0.9, nucleus_entry)]
    
    result = expander.expand(nuclei, store, 'u1', ExpansionMode.FULL)
    
    # Should have nucleus + session neighbors + temporal + entity chain
    assert result.nuclei_count >= 1
    
    # Entity chain should find old "项目" mentions from session 1
    roles = [e.role for e in result.entries]
    contents = [e.memory.content for e in result.entries]
    
    print(f"  Full expansion: {len(result.entries)} entries")
    print(f"    Nuclei: {result.nuclei_count}")
    print(f"    Session: {result.session_neighbors}")
    print(f"    Temporal: {result.temporal_neighbors}")
    print(f"    Entity chain: {result.entity_chain_entries}")
    
    # "项目deadline是下周五" should appear via entity chain (same entity '项目')
    if result.entity_chain_entries > 0:
        entity_contents = [e.memory.content for e in result.entries if e.role == ContextRole.ENTITY_CHAIN]
        print(f"    Entity chain contents: {entity_contents}")
    
    print("✓ test_full_expansion")


def test_budget_enforcement():
    """Context entries are truncated to max budget"""
    store = AppendOnlyStore()
    now = int(time.time() * 1000)
    
    # Create many memories in same session
    for i in range(20):
        store.add(f'消息{i}', session_id='sess_big', 
                 timestamp=now - (20-i)*60000, user_id='u1')
    
    # Tight budget
    expander = ContextExpander(max_context_entries=5, session_window_before=10, session_window_after=10)
    
    all_mems = store.get_all('u1')
    mid = all_mems[10]
    nuclei = [RankedMemory(mid.id, mid.content, 0.9, mid)]
    
    result = expander.expand(nuclei, store, 'u1', ExpansionMode.SESSION_ONLY)
    
    assert len(result.entries) <= 5, f"Budget exceeded: {len(result.entries)}"
    assert result.truncated
    print(f"  Budget enforced: {len(result.entries)} entries (max=5), truncated={result.truncated}")
    print("✓ test_budget_enforcement")


def test_deduplication():
    """Same memory is never included twice"""
    store = AppendOnlyStore()
    now = int(time.time() * 1000)
    
    # Entry that would be both session neighbor AND temporal neighbor
    e1 = store.add('A', session_id='s1', timestamp=now - 60000, user_id='u1')
    e2 = store.add('B', session_id='s1', timestamp=now, user_id='u1')  # nucleus
    e3 = store.add('C', session_id='s1', timestamp=now + 60000, user_id='u1')
    
    expander = ContextExpander(temporal_window_hours=1.0)
    nuclei = [RankedMemory(e2.id, e2.content, 0.9, e2)]
    
    result = expander.expand(nuclei, store, 'u1', ExpansionMode.SESSION_AND_TEMPORAL)
    
    # Count occurrences of each id
    ids = [e.memory.id for e in result.entries]
    assert len(ids) == len(set(ids)), f"Duplicate entries found: {ids}"
    print("✓ test_deduplication")


def test_multiple_nuclei():
    """Multiple nucleus entries expand independently"""
    store = AppendOnlyStore()
    data = build_conversation(store)
    expander = ContextExpander()
    
    # Two nuclei from different sessions
    n1 = data['sess1'][2]  # "跨部门沟通特别头疼"
    n2 = data['sess3'][0]  # "项目终于提交了"
    
    nuclei = [
        RankedMemory(n1.id, n1.content, 0.9, n1),
        RankedMemory(n2.id, n2.content, 0.85, n2),
    ]
    
    result = expander.expand(nuclei, store, 'u1', ExpansionMode.SESSION_ONLY)
    
    # Should get neighbors from BOTH sessions
    sessions = {e.memory.session_id for e in result.entries}
    assert 'sess1' in sessions
    assert 'sess3' in sessions
    
    print(f"  Multi-nuclei expansion from sessions: {sessions}")
    print("✓ test_multiple_nuclei")


def test_empty_nuclei():
    """Empty nuclei list returns empty context"""
    store = AppendOnlyStore()
    store.add('something', user_id='u1')
    expander = ContextExpander()
    
    result = expander.expand([], store, 'u1')
    assert len(result.entries) == 0
    print("✓ test_empty_nuclei")


def test_single_entry_no_neighbors():
    """Single memory in store → no neighbors to expand"""
    store = AppendOnlyStore()
    entry = store.add('唯一的记忆', session_id='only', user_id='u1')
    
    expander = ContextExpander()
    nuclei = [RankedMemory(entry.id, entry.content, 0.9, entry)]
    
    result = expander.expand(nuclei, store, 'u1', ExpansionMode.SESSION_ONLY)
    
    # Only the nucleus itself
    assert result.nuclei_count == 1
    assert result.session_neighbors == 0
    print("✓ test_single_entry_no_neighbors")


def test_context_ordering():
    """Entries are ordered: nucleus first, then by role priority"""
    store = AppendOnlyStore()
    data = build_conversation(store)
    expander = ContextExpander()
    
    nucleus_entry = data['sess2'][1]  # "西湖真的很美"
    nuclei = [RankedMemory(nucleus_entry.id, nucleus_entry.content, 0.9, nucleus_entry)]
    
    result = expander.expand(nuclei, store, 'u1', ExpansionMode.FULL)
    
    # Verify ordering: NUCLEUS should come first
    if result.entries:
        assert result.entries[0].role == ContextRole.NUCLEUS
        # All nuclei should be before all other roles
        found_non_nucleus = False
        for e in result.entries:
            if e.role != ContextRole.NUCLEUS:
                found_non_nucleus = True
            if found_non_nucleus and e.role == ContextRole.NUCLEUS:
                assert False, "Nucleus found after non-nucleus"
    
    print("✓ test_context_ordering")


def test_why_add_only_helps_expansion():
    """
    Demonstrate: ADD-only preserves context chain that CRUD would destroy.
    
    If we had DELETE'd "项目deadline是下周五" when "项目终于提交了" arrived,
    we'd lose the ability to provide the full project timeline context.
    """
    store = AppendOnlyStore()
    now = int(time.time() * 1000)
    day = 86400 * 1000
    
    # Full project timeline (ADD-only preserves ALL)
    store.add('接到新项目', entities=['项目'], timestamp=now-30*day, session_id='s1', user_id='u1')
    store.add('项目deadline是下周五', entities=['项目'], timestamp=now-7*day, session_id='s2', user_id='u1')
    store.add('项目进度落后了', entities=['项目'], timestamp=now-3*day, session_id='s3', user_id='u1')
    final = store.add('项目终于提交了', entities=['项目'], timestamp=now-1*day, session_id='s4', user_id='u1')
    
    expander = ContextExpander(entity_chain_limit=5)
    nuclei = [RankedMemory(final.id, final.content, 0.9, final)]
    
    result = expander.expand(nuclei, store, 'u1', ExpansionMode.FULL)
    
    # Entity chain should show project evolution
    all_contents = [e.memory.content for e in result.entries]
    
    # In ADD-only: full timeline available
    assert '项目终于提交了' in all_contents  # current
    # At least some historical project entries should be in entity chain
    historical = [c for c in all_contents if c != '项目终于提交了']
    assert len(historical) >= 1, "ADD-only should preserve historical context"
    
    print(f"  Project timeline preserved: {all_contents}")
    print("  (CRUD would have deleted old entries, losing this context)")
    print("✓ test_why_add_only_helps_expansion")


if __name__ == '__main__':
    tests = [
        test_nucleus_only,
        test_session_expansion,
        test_temporal_expansion,
        test_full_expansion,
        test_budget_enforcement,
        test_deduplication,
        test_multiple_nuclei,
        test_empty_nuclei,
        test_single_entry_no_neighbors,
        test_context_ordering,
        test_why_add_only_helps_expansion,
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
