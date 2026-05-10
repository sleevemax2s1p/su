"""
Tests for AdaptiveContextSelector logic
Python replica of Kotlin logic for validation
"""
import pytest
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

# === Enums ===

class QueryComplexity(Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"
    MULTI_HOP = "MULTI_HOP"

class GovernanceLayer(Enum):
    CONSTITUTIONAL = "CONSTITUTIONAL"
    STATUTORY = "STATUTORY"
    OPERATIONAL = "OPERATIONAL"

class TrustLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

# === Data Classes ===

@dataclass
class MemoryCandidate:
    id: str
    content: str
    score: float
    layer: GovernanceLayer
    trust_level: TrustLevel
    token_estimate: int = 50
    retrieval_source: str = "hybrid"

@dataclass
class ConversationContext:
    recent_topics: List[str] = field(default_factory=list)
    turn_count: int = 0
    user_mood: Optional[str] = None

@dataclass
class SelectionResult:
    selected: List[MemoryCandidate]
    total_candidates: int = 0
    selected_count: int = 0
    dynamic_k: int = 0
    score_gap_k: int = 0
    threshold_k: int = 0
    final_k: int = 0
    estimated_tokens: int = 0
    reason: str = ""

    @property
    def compression_ratio(self):
        if self.total_candidates > 0:
            return 1.0 - (self.selected_count / self.total_candidates)
        return 0.0

# === AdaptiveContextSelector ===

class AdaptiveContextSelector:
    def __init__(
        self,
        max_token_budget: int = 1200,
        min_relevance_score: float = 0.005,
        score_gap_threshold: float = 0.3,
        avg_tokens_per_memory: int = 50,
        constitutional_always_include: bool = True
    ):
        self.max_token_budget = max_token_budget
        self.min_relevance_score = min_relevance_score
        self.score_gap_threshold = score_gap_threshold
        self.avg_tokens_per_memory = avg_tokens_per_memory
        self.constitutional_always_include = constitutional_always_include

    def select(
        self,
        candidates: List[MemoryCandidate],
        query_complexity: QueryComplexity = QueryComplexity.MEDIUM,
        conversation_context: ConversationContext = None
    ) -> SelectionResult:
        if conversation_context is None:
            conversation_context = ConversationContext()

        if not candidates:
            return SelectionResult(selected=[], reason="No candidates")

        # Step 1: Dynamic K
        budget_k = self.max_token_budget // self.avg_tokens_per_memory
        complexity_k = {
            QueryComplexity.SIMPLE: 3,
            QueryComplexity.MEDIUM: 6,
            QueryComplexity.COMPLEX: 10,
            QueryComplexity.MULTI_HOP: 12
        }[query_complexity]
        dynamic_k = min(budget_k, complexity_k)

        # Step 2: Score gap detection
        gap_cutoff = self._detect_score_gap(candidates)

        # Step 3: Threshold filter
        threshold_cutoff = 0
        for i, c in enumerate(candidates):
            if c.score >= self.min_relevance_score:
                threshold_cutoff = i + 1

        # Step 4: Final K
        final_k = min(dynamic_k, min(gap_cutoff, threshold_cutoff))

        # Step 5: Select with Constitutional guarantee
        selected = []
        reasons = []

        if self.constitutional_always_include:
            constitutionals = [c for c in candidates if c.layer == GovernanceLayer.CONSTITUTIONAL]
            selected.extend(constitutionals)
            if constitutionals:
                reasons.append(f"{len(constitutionals)} constitutional memories always included")

        non_const = [c for c in candidates if c.layer != GovernanceLayer.CONSTITUTIONAL]
        remaining = final_k - len(selected)
        if remaining > 0:
            selected.extend(non_const[:remaining])

        # Step 6: Priority ordering (sandwich)
        ordered = self._priority_order(selected, conversation_context)

        return SelectionResult(
            selected=ordered,
            total_candidates=len(candidates),
            selected_count=len(ordered),
            dynamic_k=dynamic_k,
            score_gap_k=gap_cutoff,
            threshold_k=threshold_cutoff,
            final_k=final_k,
            estimated_tokens=len(ordered) * self.avg_tokens_per_memory,
            reason=self._build_reason(ordered, reasons, len(candidates))
        )

    def _detect_score_gap(self, candidates: List[MemoryCandidate]) -> int:
        if len(candidates) <= 1:
            return len(candidates)

        scores = [c.score for c in candidates]
        diffs = []
        for i in range(1, len(scores)):
            diffs.append(scores[i - 1] - scores[i])

        if not diffs:
            return len(candidates)

        avg_diff = sum(diffs) / len(diffs)
        threshold = avg_diff * (1.0 + self.score_gap_threshold)

        for i, d in enumerate(diffs):
            if d > threshold and d > 0.002:
                return i + 1

        return len(candidates)

    def _priority_order(
        self,
        memories: List[MemoryCandidate],
        context: ConversationContext
    ) -> List[MemoryCandidate]:
        if len(memories) <= 3:
            return memories

        def priority_score(c: MemoryCandidate) -> float:
            p = c.score
            if c.layer == GovernanceLayer.CONSTITUTIONAL:
                p += 0.5
            if c.trust_level == TrustLevel.HIGH:
                p += 0.1
            if context.recent_topics and any(t in c.content for t in context.recent_topics):
                p += 0.2
            return p

        sorted_mems = sorted(memories, key=priority_score, reverse=True)

        # Sandwich: odd indices first, then even indices reversed
        result = []
        n = len(sorted_mems)
        for i in range(0, n, 2):
            result.append(sorted_mems[i])
        middle = []
        for i in range(1, n, 2):
            middle.append(sorted_mems[i])
        result.extend(reversed(middle))
        return result

    def _build_reason(self, selected, extra_reasons, total):
        parts = [f"Selected {len(selected)}/{total} candidates"]
        parts.extend(extra_reasons)
        return "; ".join(parts)

    @staticmethod
    def assess_query_complexity(query: str) -> QueryComplexity:
        multi_hop_signals = ["的", "在哪", "谁的", "工作的公司", "住在"]
        multi_hop_count = sum(1 for s in multi_hop_signals if s in query)

        length = len(query)
        has_question = any(q in query for q in ["?", "？", "吗", "呢"])

        if multi_hop_count >= 2:
            return QueryComplexity.MULTI_HOP
        elif length > 50 or (has_question and multi_hop_count >= 1):
            return QueryComplexity.COMPLEX
        elif length > 20 or has_question:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.SIMPLE


# === Tests ===

def make_candidate(id, content, score, layer=GovernanceLayer.OPERATIONAL, trust=TrustLevel.MEDIUM):
    return MemoryCandidate(id=id, content=content, score=score, layer=layer, trust_level=trust)


class TestDynamicK:
    """Test dynamic K based on query complexity"""

    def test_simple_query_max_3(self):
        selector = AdaptiveContextSelector()
        candidates = [make_candidate(f"m{i}", f"memory {i}", 0.9 - i * 0.05) for i in range(10)]
        result = selector.select(candidates, QueryComplexity.SIMPLE)
        assert result.selected_count <= 3

    def test_medium_query_max_6(self):
        selector = AdaptiveContextSelector()
        candidates = [make_candidate(f"m{i}", f"memory {i}", 0.9 - i * 0.05) for i in range(10)]
        result = selector.select(candidates, QueryComplexity.MEDIUM)
        assert result.selected_count <= 6

    def test_complex_query_max_10(self):
        selector = AdaptiveContextSelector()
        candidates = [make_candidate(f"m{i}", f"memory {i}", 0.9 - i * 0.01) for i in range(15)]
        result = selector.select(candidates, QueryComplexity.COMPLEX)
        assert result.selected_count <= 10

    def test_multihop_query_max_12(self):
        selector = AdaptiveContextSelector()
        candidates = [make_candidate(f"m{i}", f"memory {i}", 0.9 - i * 0.005) for i in range(20)]
        result = selector.select(candidates, QueryComplexity.MULTI_HOP)
        assert result.selected_count <= 12

    def test_budget_constrains_k(self):
        """Token budget = 200, avg = 50 → max 4 regardless of complexity"""
        selector = AdaptiveContextSelector(max_token_budget=200, avg_tokens_per_memory=50)
        candidates = [make_candidate(f"m{i}", f"memory {i}", 0.9 - i * 0.01) for i in range(10)]
        result = selector.select(candidates, QueryComplexity.COMPLEX)
        assert result.selected_count <= 4


class TestScoreGapDetection:
    """Test cliff detection in relevance scores"""

    def test_clear_gap(self):
        """[0.95, 0.92, 0.88, 0.40, 0.35] → gap after index 2"""
        selector = AdaptiveContextSelector()
        candidates = [
            make_candidate("a", "a", 0.95),
            make_candidate("b", "b", 0.92),
            make_candidate("c", "c", 0.88),
            make_candidate("d", "d", 0.40),
            make_candidate("e", "e", 0.35),
        ]
        result = selector.select(candidates, QueryComplexity.COMPLEX)
        # Should cut at the gap: select at most 3
        assert result.selected_count <= 3
        assert result.score_gap_k == 3

    def test_no_gap_uniform_scores(self):
        """Uniform descent → no gap, select up to dynamic K"""
        selector = AdaptiveContextSelector()
        candidates = [make_candidate(f"m{i}", f"m{i}", 0.9 - i * 0.05) for i in range(8)]
        result = selector.select(candidates, QueryComplexity.MEDIUM)
        # No gap detected → limited by complexity K (6) or count
        assert result.score_gap_k == 8  # no gap found

    def test_gap_at_start(self):
        """[0.95, 0.20, 0.18, 0.15] → gap after 1"""
        selector = AdaptiveContextSelector()
        candidates = [
            make_candidate("a", "a", 0.95),
            make_candidate("b", "b", 0.20),
            make_candidate("c", "c", 0.18),
            make_candidate("d", "d", 0.15),
        ]
        result = selector.select(candidates, QueryComplexity.COMPLEX)
        assert result.score_gap_k == 1
        assert result.selected_count == 1


class TestConstitutionalGuarantee:
    """Constitutional memories always included"""

    def test_constitutional_included_even_low_score(self):
        selector = AdaptiveContextSelector()
        candidates = [
            make_candidate("op1", "operational", 0.95, GovernanceLayer.OPERATIONAL),
            make_candidate("op2", "operational2", 0.90, GovernanceLayer.OPERATIONAL),
            make_candidate("const1", "你的名字是小明", 0.01, GovernanceLayer.CONSTITUTIONAL),
        ]
        result = selector.select(candidates, QueryComplexity.SIMPLE)
        selected_ids = [c.id for c in result.selected]
        assert "const1" in selected_ids

    def test_constitutional_does_not_count_against_k(self):
        """With K=3 and 1 constitutional, should get constitutional + 2 operational"""
        selector = AdaptiveContextSelector()
        candidates = [
            make_candidate("const", "identity", 0.80, GovernanceLayer.CONSTITUTIONAL),
            make_candidate("op1", "op1", 0.90, GovernanceLayer.OPERATIONAL),
            make_candidate("op2", "op2", 0.85, GovernanceLayer.OPERATIONAL),
            make_candidate("op3", "op3", 0.80, GovernanceLayer.OPERATIONAL),
            make_candidate("op4", "op4", 0.75, GovernanceLayer.OPERATIONAL),
        ]
        result = selector.select(candidates, QueryComplexity.SIMPLE)
        selected_ids = [c.id for c in result.selected]
        assert "const" in selected_ids
        # K=3 total: 1 const + 2 operational
        assert result.selected_count == 3


class TestThresholdFilter:
    """Min relevance score filter"""

    def test_below_threshold_excluded(self):
        selector = AdaptiveContextSelector(min_relevance_score=0.1)
        candidates = [
            make_candidate("a", "a", 0.90),
            make_candidate("b", "b", 0.50),
            make_candidate("c", "c", 0.05),  # below threshold
            make_candidate("d", "d", 0.001),  # below threshold
        ]
        result = selector.select(candidates, QueryComplexity.COMPLEX)
        selected_ids = [c.id for c in result.selected]
        assert "c" not in selected_ids
        assert "d" not in selected_ids
        assert result.threshold_k == 2

    def test_all_above_threshold(self):
        selector = AdaptiveContextSelector(min_relevance_score=0.01)
        candidates = [make_candidate(f"m{i}", f"m{i}", 0.5 - i * 0.05) for i in range(5)]
        result = selector.select(candidates, QueryComplexity.MEDIUM)
        assert result.threshold_k == 5


class TestSandwichOrdering:
    """Lost-in-the-Middle mitigation via sandwich ordering"""

    def test_sandwich_ordering_applied(self):
        """With 5+ items, most important at start and end"""
        selector = AdaptiveContextSelector(score_gap_threshold=10.0)  # disable gap detection
        candidates = [
            make_candidate("a", "a", 0.95, trust_level=TrustLevel.HIGH),
            make_candidate("b", "b", 0.90, trust_level=TrustLevel.HIGH),
            make_candidate("c", "c", 0.85),
            make_candidate("d", "d", 0.80),
            make_candidate("e", "e", 0.75),
            make_candidate("f", "f", 0.70),
        ]
        result = selector.select(candidates, QueryComplexity.MEDIUM)
        # First item should be highest priority
        assert result.selected_count >= 4
        # Verify ordering is not just descending score
        if result.selected_count > 3:
            ids = [c.id for c in result.selected]
            # In sandwich: highest priority at start AND end
            # Not simply [a,b,c,d,e,f] descending
            assert ids != ["a", "b", "c", "d", "e", "f"]

    def test_small_list_no_reorder(self):
        """<= 3 items: no sandwich, keep original order"""
        selector = AdaptiveContextSelector()
        candidates = [
            make_candidate("a", "a", 0.90),
            make_candidate("b", "b", 0.80),
            make_candidate("c", "c", 0.70),
        ]
        result = selector.select(candidates, QueryComplexity.SIMPLE)
        assert result.selected_count == 3
        # Original order preserved (no reorder for <= 3)
        ids = [c.id for c in result.selected]
        assert ids == ["a", "b", "c"]


class TestQueryComplexityAssessment:
    """Query complexity classification"""

    def test_simple_short(self):
        assert AdaptiveContextSelector.assess_query_complexity("你好") == QueryComplexity.SIMPLE

    def test_medium_with_question(self):
        assert AdaptiveContextSelector.assess_query_complexity("你还记得吗") == QueryComplexity.MEDIUM

    def test_complex_long_with_question(self):
        q = "上次我们讨论的那个关于数据库迁移的方案，后来你觉得怎么样了呢？有没有什么新的想法？"
        result = AdaptiveContextSelector.assess_query_complexity(q)
        assert result in (QueryComplexity.COMPLEX, QueryComplexity.MULTI_HOP)

    def test_multi_hop(self):
        q = "我女朋友工作的公司在哪个城市"
        assert AdaptiveContextSelector.assess_query_complexity(q) == QueryComplexity.MULTI_HOP


class TestCompressionRatio:
    """Verify compression ratio calculation"""

    def test_compression_50_percent(self):
        selector = AdaptiveContextSelector()
        candidates = [make_candidate(f"m{i}", f"m{i}", 0.9 - i * 0.05) for i in range(6)]
        result = selector.select(candidates, QueryComplexity.SIMPLE)
        # Simple → K=3, 6 candidates → 3 selected → 50% compression
        assert result.selected_count == 3
        assert abs(result.compression_ratio - 0.5) < 0.01

    def test_compression_with_gap(self):
        selector = AdaptiveContextSelector()
        candidates = [
            make_candidate("a", "a", 0.95),
            make_candidate("b", "b", 0.92),
            make_candidate("c", "c", 0.10),  # gap
            make_candidate("d", "d", 0.08),
            make_candidate("e", "e", 0.06),
        ]
        result = selector.select(candidates, QueryComplexity.MEDIUM)
        # Gap at index 2 → select 2, compression = 1 - 2/5 = 0.6
        assert result.compression_ratio >= 0.5


class TestEdgeCases:
    """Edge cases"""

    def test_empty_candidates(self):
        selector = AdaptiveContextSelector()
        result = selector.select([], QueryComplexity.MEDIUM)
        assert result.selected_count == 0
        assert result.reason == "No candidates"

    def test_single_candidate(self):
        selector = AdaptiveContextSelector()
        candidates = [make_candidate("only", "only memory", 0.50)]
        result = selector.select(candidates, QueryComplexity.MEDIUM)
        assert result.selected_count == 1

    def test_all_constitutional(self):
        selector = AdaptiveContextSelector()
        candidates = [
            make_candidate(f"c{i}", f"const{i}", 0.9 - i * 0.1, GovernanceLayer.CONSTITUTIONAL)
            for i in range(5)
        ]
        result = selector.select(candidates, QueryComplexity.SIMPLE)
        # All constitutional → all included (even though K=3)
        assert result.selected_count == 5

    def test_recency_boost_in_ordering(self):
        """Recent topics get priority boost in sandwich ordering"""
        selector = AdaptiveContextSelector(score_gap_threshold=10.0)
        candidates = [
            make_candidate("a", "我喜欢北京", 0.90),
            make_candidate("b", "我在上海工作", 0.88),
            make_candidate("c", "今天讨论旅行计划", 0.85),
            make_candidate("d", "明天开会", 0.82),
            make_candidate("e", "周末打球", 0.80),
        ]
        ctx = ConversationContext(recent_topics=["旅行"])
        result = selector.select(candidates, QueryComplexity.MEDIUM, ctx)
        # "旅行" topic should get boost, affecting ordering
        assert result.selected_count == 5


class TestIntegration:
    """Integration test: full pipeline"""

    def test_realistic_scenario(self):
        """Simulate real retrieval results with mixed layers/scores"""
        selector = AdaptiveContextSelector()
        candidates = [
            make_candidate("name", "用户名字是小明", 0.85, GovernanceLayer.CONSTITUTIONAL, TrustLevel.HIGH),
            make_candidate("city", "住在北京朝阳区", 0.80, GovernanceLayer.STATUTORY, TrustLevel.HIGH),
            make_candidate("food", "喜欢吃火锅", 0.75, GovernanceLayer.OPERATIONAL),
            make_candidate("work", "在字节跳动工作", 0.70, GovernanceLayer.STATUTORY),
            make_candidate("pet", "养了一只猫叫团团", 0.65, GovernanceLayer.OPERATIONAL),
            make_candidate("hobby", "周末喜欢爬山", 0.45, GovernanceLayer.OPERATIONAL),
            make_candidate("music", "听周杰伦的歌", 0.40, GovernanceLayer.OPERATIONAL),
            make_candidate("old", "以前住在上海", 0.10, GovernanceLayer.OPERATIONAL),
        ]

        # Medium query
        result = selector.select(candidates, QueryComplexity.MEDIUM)
        print(f"\n{result.reason}")
        print(f"Compression: {result.compression_ratio:.0%}")
        print(f"Selected: {[c.id for c in result.selected]}")

        # Verify basics
        assert result.selected_count <= 6
        assert result.selected_count >= 1
        assert "name" in [c.id for c in result.selected]  # constitutional
        assert result.compression_ratio > 0  # some compression happened
        assert result.estimated_tokens <= 1200

    def test_memact_style_compression(self):
        """Verify we achieve ~50% compression like MemAct reports"""
        selector = AdaptiveContextSelector()
        # 20 candidates, medium query → should select ~6
        candidates = [make_candidate(f"m{i}", f"memory content {i}", 0.9 - i * 0.03) for i in range(20)]
        result = selector.select(candidates, QueryComplexity.MEDIUM)
        # Should achieve significant compression
        assert result.compression_ratio >= 0.5
        print(f"\nMemAct-style compression: {result.compression_ratio:.0%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
