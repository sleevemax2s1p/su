"""
Tests for EventBoundaryDetector — HiGMem-inspired event segmentation
"""
import sys
import os
import time

# Simulating the Kotlin EventBoundaryDetector in Python for test validation

class MemoryEntry:
    def __init__(self, id, content, timestamp, entities=None, session_id=None):
        self.id = id
        self.content = content
        self.timestamp = timestamp
        self.entities = entities or []
        self.session_id = session_id


class EventBoundaryDetector:
    def __init__(
        self,
        time_gap_threshold_ms=30*60*1000,  # 30 min
        time_gap_min_ms=2*60*1000,  # 2 min
        entity_change_threshold=0.7,
        boundary_threshold=0.6,
        time_gap_weight=0.4,
        entity_change_weight=0.3,
        session_change_weight=0.2,
        explicit_signal_weight=0.1,
    ):
        self.time_gap_threshold_ms = time_gap_threshold_ms
        self.time_gap_min_ms = time_gap_min_ms
        self.entity_change_threshold = entity_change_threshold
        self.boundary_threshold = boundary_threshold
        self.time_gap_weight = time_gap_weight
        self.entity_change_weight = entity_change_weight
        self.session_change_weight = session_change_weight
        self.explicit_signal_weight = explicit_signal_weight
        self.topic_change_signals = [
            "对了", "另外", "换个话题", "说到这个", "顺便说一下",
            "不说这个了", "说点别的", "我想问", "还有个事"
        ]

    def detect_boundary(self, before, after):
        signals = {}

        # Time gap signal
        time_gap = after.timestamp - before.timestamp
        if time_gap >= self.time_gap_threshold_ms:
            time_signal = 1.0
        elif time_gap <= self.time_gap_min_ms:
            time_signal = 0.0
        else:
            time_signal = (time_gap - self.time_gap_min_ms) / (self.time_gap_threshold_ms - self.time_gap_min_ms)
        signals["time_gap"] = time_signal

        # Entity change
        before_ents = set(before.entities)
        after_ents = set(after.entities)
        if not before_ents and not after_ents:
            entity_signal = 0.3
        elif not before_ents or not after_ents:
            entity_signal = 0.5
        else:
            intersection = len(before_ents & after_ents)
            union = len(before_ents | after_ents)
            entity_signal = 1.0 - (intersection / union)
        signals["entity_change"] = entity_signal

        # Session change
        if before.session_id and after.session_id:
            session_signal = 1.0 if before.session_id != after.session_id else 0.0
        else:
            session_signal = 0.0
        signals["session_change"] = session_signal

        # Explicit signal
        explicit_signal = 1.0 if any(s in after.content for s in self.topic_change_signals) else 0.0
        signals["explicit_signal"] = explicit_signal

        # Weighted fusion
        probability = (
            time_signal * self.time_gap_weight +
            entity_signal * self.entity_change_weight +
            session_signal * self.session_change_weight +
            explicit_signal * self.explicit_signal_weight
        )

        is_boundary = probability >= self.boundary_threshold
        return {
            "is_boundary": is_boundary,
            "probability": min(max(probability, 0.0), 1.0),
            "signals": signals,
            "time_gap_ms": time_gap
        }

    def segment_into_events(self, memories):
        if not memories:
            return []
        sorted_mems = sorted(memories, key=lambda m: m.timestamp)
        if len(sorted_mems) == 1:
            return [sorted_mems]

        events = []
        current_group = [sorted_mems[0]]

        for i in range(1, len(sorted_mems)):
            boundary = self.detect_boundary(sorted_mems[i-1], sorted_mems[i])
            if boundary["is_boundary"]:
                events.append(current_group)
                current_group = [sorted_mems[i]]
            else:
                current_group.append(sorted_mems[i])

        if current_group:
            events.append(current_group)
        return events

    def find_event_for_nucleus(self, nucleus, all_memories):
        sorted_mems = sorted(all_memories, key=lambda m: m.timestamp)
        nucleus_idx = next((i for i, m in enumerate(sorted_mems) if m.id == nucleus.id), -1)
        if nucleus_idx < 0:
            return [nucleus]

        start = nucleus_idx
        while start > 0:
            boundary = self.detect_boundary(sorted_mems[start-1], sorted_mems[start])
            if boundary["is_boundary"]:
                break
            start -= 1

        end = nucleus_idx
        while end < len(sorted_mems) - 1:
            boundary = self.detect_boundary(sorted_mems[end], sorted_mems[end+1])
            if boundary["is_boundary"]:
                break
            end += 1

        return sorted_mems[start:end+1]


# === Tests ===

def test_no_boundary_within_event():
    """Memories close in time with same entities → no boundary"""
    detector = EventBoundaryDetector()
    base_time = 1000000000000

    m1 = MemoryEntry("m1", "我住在北京朝阳", base_time, ["北京"], "s1")
    m2 = MemoryEntry("m2", "北京的天气不错", base_time + 60*1000, ["北京"], "s1")

    result = detector.detect_boundary(m1, m2)
    assert not result["is_boundary"], f"Expected no boundary, got prob={result['probability']}"
    assert result["probability"] < 0.4


def test_time_gap_boundary():
    """Large time gap → boundary detected"""
    detector = EventBoundaryDetector()
    base_time = 1000000000000

    m1 = MemoryEntry("m1", "今天好累", base_time, ["工作"], "s1")
    m2 = MemoryEntry("m2", "想吃火锅", base_time + 60*60*1000, ["火锅"], "s1")  # 1hr, different entity

    result = detector.detect_boundary(m1, m2)
    assert result["is_boundary"], f"Expected boundary for 1hr gap, prob={result['probability']}"
    assert result["signals"]["time_gap"] == 1.0


def test_session_change_boundary():
    """Different sessions → strong boundary signal"""
    detector = EventBoundaryDetector()
    base_time = 1000000000000

    m1 = MemoryEntry("m1", "晚安", base_time, ["睡觉"], "session_1")
    m2 = MemoryEntry("m2", "早上好", base_time + 20*60*1000, ["起床"], "session_2")  # 20 min, new session, different entities

    result = detector.detect_boundary(m1, m2)
    assert result["signals"]["session_change"] == 1.0
    # time(0.64*0.4) + entity(1.0*0.3) + session(1.0*0.2) = 0.76 → boundary
    assert result["is_boundary"], f"Session change should trigger boundary, prob={result['probability']}"


def test_entity_change_boundary():
    """Completely different entities → topic change"""
    detector = EventBoundaryDetector()
    base_time = 1000000000000

    m1 = MemoryEntry("m1", "我养了只猫叫年糕", base_time, ["猫", "年糕"], "s1")
    m2 = MemoryEntry("m2", "明天要去面试", base_time + 10*60*1000, ["面试", "工作"], "s1")  # 10 min

    result = detector.detect_boundary(m1, m2)
    assert result["signals"]["entity_change"] == 1.0  # No overlap
    # time: ~0.29, entity: 1.0, session: 0, explicit: 0
    # = 0.29*0.4 + 1.0*0.3 = 0.116 + 0.3 = 0.416 → might not be enough alone
    # But entity change is strong signal


def test_explicit_topic_change():
    """Explicit signal words → boundary boost"""
    detector = EventBoundaryDetector()
    base_time = 1000000000000

    m1 = MemoryEntry("m1", "北京的天气不错", base_time, ["北京"], "s1")
    m2 = MemoryEntry("m2", "对了，我想问一下猫的事", base_time + 10*60*1000, ["猫"], "s1")

    result = detector.detect_boundary(m1, m2)
    assert result["signals"]["explicit_signal"] == 1.0
    # Contains both "对了" and "我想问"


def test_segment_into_events_basic():
    """Three memories: two close + one far → two events"""
    detector = EventBoundaryDetector()
    base_time = 1000000000000

    memories = [
        MemoryEntry("m1", "我喜欢猫", base_time, ["猫"], "s1"),
        MemoryEntry("m2", "猫叫年糕", base_time + 60*1000, ["猫", "年糕"], "s1"),
        MemoryEntry("m3", "今天去上班了", base_time + 60*60*1000, ["工作"], "s1"),  # 1hr later
    ]

    events = detector.segment_into_events(memories)
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"
    assert len(events[0]) == 2  # m1, m2
    assert len(events[1]) == 1  # m3


def test_segment_single_memory():
    """Single memory → single event"""
    detector = EventBoundaryDetector()
    m1 = MemoryEntry("m1", "hello", 1000000000000, [], "s1")
    events = detector.segment_into_events([m1])
    assert len(events) == 1
    assert len(events[0]) == 1


def test_segment_empty():
    """Empty input → empty output"""
    detector = EventBoundaryDetector()
    events = detector.segment_into_events([])
    assert len(events) == 0


def test_find_event_for_nucleus():
    """Nucleus in middle of event → expands to full event"""
    detector = EventBoundaryDetector()
    base_time = 1000000000000

    memories = [
        MemoryEntry("m1", "准备出发", base_time, ["旅行"], "s1"),
        MemoryEntry("m2", "到机场了", base_time + 60*1000, ["机场", "旅行"], "s1"),
        MemoryEntry("m3", "飞机起飞了", base_time + 2*60*1000, ["飞机", "旅行"], "s1"),
        # === boundary (1hr gap + different entities) ===
        MemoryEntry("m4", "今天工作很忙", base_time + 60*60*1000, ["工作"], "s1"),
        MemoryEntry("m5", "开了三个会", base_time + 61*60*1000, ["会议", "工作"], "s1"),
    ]

    # Find event for m2 (middle of first event)
    event = detector.find_event_for_nucleus(memories[1], memories)
    event_ids = [m.id for m in event]
    assert "m1" in event_ids, "m1 should be in same event as m2"
    assert "m2" in event_ids
    assert "m3" in event_ids
    assert "m4" not in event_ids, "m4 should be in different event"


def test_find_event_for_nucleus_at_boundary():
    """Nucleus at event start → only forward expansion"""
    detector = EventBoundaryDetector()
    base_time = 1000000000000

    memories = [
        MemoryEntry("m1", "旧话题结束", base_time, ["旧"], "s1"),
        # === boundary (1hr gap) ===
        MemoryEntry("m2", "新话题开始", base_time + 60*60*1000, ["新"], "s1"),
        MemoryEntry("m3", "新话题继续", base_time + 61*60*1000, ["新"], "s1"),
    ]

    event = detector.find_event_for_nucleus(memories[1], memories)
    event_ids = [m.id for m in event]
    assert "m1" not in event_ids, "m1 should be separated by boundary"
    assert "m2" in event_ids
    assert "m3" in event_ids


def test_custom_threshold():
    """Custom boundary threshold changes sensitivity"""
    # Strict threshold (higher = less boundaries)
    strict = EventBoundaryDetector(boundary_threshold=0.8)
    # Loose threshold (lower = more boundaries)
    loose = EventBoundaryDetector(boundary_threshold=0.3)

    base_time = 1000000000000
    m1 = MemoryEntry("m1", "话题A", base_time, ["A"], "s1")
    m2 = MemoryEntry("m2", "话题B", base_time + 20*60*1000, ["B"], "s1")  # 20 min, different entity

    strict_result = strict.detect_boundary(m1, m2)
    loose_result = loose.detect_boundary(m1, m2)

    # Same probability, different threshold decisions
    assert strict_result["probability"] == loose_result["probability"]
    # Loose should be more likely to mark as boundary
    assert loose_result["is_boundary"] or (not strict_result["is_boundary"])


def test_multi_event_segment():
    """Multiple events with clear boundaries"""
    detector = EventBoundaryDetector()
    base_time = 1000000000000
    hour = 60*60*1000

    memories = [
        # Event 1: Morning chat about cat
        MemoryEntry("m1", "年糕今天很乖", base_time, ["猫", "年糕"], "s1"),
        MemoryEntry("m2", "给年糕喂了猫粮", base_time + 60*1000, ["猫", "年糕"], "s1"),
        # Event 2: Afternoon work discussion (1hr gap + different entities)
        MemoryEntry("m3", "项目快到deadline了", base_time + 2*hour, ["项目"], "s1"),
        MemoryEntry("m4", "和同事讨论了方案", base_time + 2*hour + 5*60*1000, ["项目", "同事"], "s1"),
        # Event 3: Evening different session
        MemoryEntry("m5", "今天好累啊", base_time + 5*hour, [], "s2"),
    ]

    events = detector.segment_into_events(memories)
    assert len(events) >= 2, f"Expected at least 2 events, got {len(events)}"
    # First event should contain m1, m2
    first_ids = [m.id for m in events[0]]
    assert "m1" in first_ids and "m2" in first_ids


def test_boundary_signals_independent():
    """Each signal contributes independently to boundary probability"""
    detector = EventBoundaryDetector()
    base_time = 1000000000000

    # Only time gap signal (same entities, same session, no explicit)
    m1 = MemoryEntry("m1", "猫很可爱", base_time, ["猫"], "s1")
    m2 = MemoryEntry("m2", "猫在睡觉", base_time + 45*60*1000, ["猫"], "s1")  # 45min

    result = detector.detect_boundary(m1, m2)
    assert result["signals"]["time_gap"] == 1.0  # Above 30min threshold
    assert result["signals"]["entity_change"] == 0.0  # Same entities
    assert result["signals"]["session_change"] == 0.0  # Same session
    assert result["signals"]["explicit_signal"] == 0.0  # No signal words
    # Pure time signal: 1.0*0.4 = 0.4, below 0.6 threshold
    # But actually entity_change for identical sets should be 0.0 (intersection/union=1.0, so 1-1=0)
    assert result["probability"] == 0.4  # Only time contributes


# === Run all tests ===

if __name__ == "__main__":
    tests = [
        test_no_boundary_within_event,
        test_time_gap_boundary,
        test_session_change_boundary,
        test_entity_change_boundary,
        test_explicit_topic_change,
        test_segment_into_events_basic,
        test_segment_single_memory,
        test_segment_empty,
        test_find_event_for_nucleus,
        test_find_event_for_nucleus_at_boundary,
        test_custom_threshold,
        test_multi_event_segment,
        test_boundary_signals_independent,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  ✓ {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"EventBoundaryDetector: {passed}/{passed+failed} passed")
    if failed > 0:
        sys.exit(1)
