"""
Tests for AdmissionController — A-MAC inspired memory quality gate
"""
import time
import math
import uuid
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Set
from collections import defaultdict

# === Models ===

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
    user_id: Optional[str]

@dataclass
class AdmissionScores:
    relevance: float
    novelty: float
    importance: float
    actionability: float
    specificity: float

@dataclass
class AdmissionDecision:
    admitted: bool
    score: float
    reason: str
    scores: AdmissionScores


class AdmissionController:
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.weights = {
            'relevance': 0.2, 'novelty': 0.25, 'importance': 0.3,
            'actionability': 0.1, 'specificity': 0.15
        }
        self.high_value_pats = [
            re.compile(r'过敏|忌口|禁忌'),
            re.compile(r'生日.*(\d+月|\d+号)'),
            re.compile(r'(密码|账号|手机号)'),
            re.compile(r'(紧急联系人|ICE)'),
            re.compile(r'(怀孕|手术|住院)')
        ]
        self.low_signal_pats = [
            re.compile(r'^(嗯|哦|好的?|行|ok|OK|是的?|对|没有?|不是|哈+|嘿|呵|啊|噢|呃)$'),
            re.compile(r'^(谢谢|感谢|好吧|随便|都行|无所谓)$'),
            re.compile(r'^[?？!！.。]+$')
        ]
    
    def evaluate(self, content, source=MemorySource.USER, existing=None, context=''):
        existing = existing or []
        
        if self._is_high_value(content):
            return AdmissionDecision(True, 1.0, 'high_value_pattern',
                                    AdmissionScores(1.0, 1.0, 1.0, 1.0, 1.0))
        
        if self._is_low_signal(content):
            return AdmissionDecision(False, 0.0, 'low_signal',
                                    AdmissionScores(0.0, 0.0, 0.0, 0.0, 0.0))
        
        scores = AdmissionScores(
            relevance=self._relevance(content, context),
            novelty=self._novelty(content, existing),
            importance=self._importance(content),
            actionability=self._actionability(content),
            specificity=self._specificity(content, source)
        )
        
        final = (scores.relevance * self.weights['relevance'] +
                scores.novelty * self.weights['novelty'] +
                scores.importance * self.weights['importance'] +
                scores.actionability * self.weights['actionability'] +
                scores.specificity * self.weights['specificity'])
        
        return AdmissionDecision(
            admitted=final >= self.threshold,
            score=final,
            reason='admitted' if final >= self.threshold else 'below_threshold',
            scores=scores
        )
    
    def _is_high_value(self, content):
        return any(p.search(content) for p in self.high_value_pats)
    
    def _is_low_signal(self, content):
        if len(content.strip()) <= 2:
            return True
        return any(p.match(content.strip()) for p in self.low_signal_pats)
    
    def _relevance(self, content, context):
        if not context: return 0.5
        sc, cc = set(content), set(context)
        union = sc | cc
        if not union: return 0.0
        return min(1.0, len(sc & cc) / len(union))
    
    def _novelty(self, content, existing):
        if not existing: return 1.0
        max_sim = max(self._jaccard(content, m.content) for m in existing)
        return max(0.0, 1.0 - max_sim)
    
    def _importance(self, content):
        score = 0.3
        if len(content) > 20: score += 0.1
        if re.search(r'(叫|名字|姓|年龄|岁|生日|住|家|老家)', content): score += 0.3
        if re.search(r'(喜欢|讨厌|爱好|偏好|最爱|不喜欢|过敏|忌口)', content): score += 0.2
        if re.search(r'(面试|会议|约会|旅行|搬家|毕业|入职|结婚)', content): score += 0.15
        if re.search(r'(朋友|同事|老板|女友|男友|爸|妈|哥|姐)', content): score += 0.2
        return min(1.0, score)
    
    def _actionability(self, content):
        score = 0.2
        if re.search(r'(明天|下周|下个月|之后|打算|计划|准备|要去|想去|约了)', content): score += 0.5
        if re.search(r'(需要|想要|希望|能不能|帮我|提醒我)', content): score += 0.3
        if re.search(r'(目标|希望能|想成为|努力|梦想)', content): score += 0.3
        return min(1.0, score)
    
    def _specificity(self, content, source):
        score = {MemorySource.USER: 0.5, MemorySource.AGENT: 0.3, MemorySource.SYSTEM: 0.2}[source]
        if '我' in content: score += 0.2
        if re.search(r'[\u4e00-\u9fa5]{2,4}(?:市|区|街|路|公司|大学|医院)', content): score += 0.15
        if re.search(r'\d{4}|\d+月|\d+号|\d+日|\d+岁', content): score += 0.15
        return min(1.0, score)
    
    def _jaccard(self, a, b):
        sa, sb = set(a), set(b)
        union = sa | sb
        if not union: return 0.0
        return len(sa & sb) / len(union)


# ============================
# Tests
# ============================

def test_low_signal_rejected():
    """Pure fillers/confirmations get rejected"""
    ctrl = AdmissionController()
    
    low_signal = ['嗯', '好', '好的', '哈哈', 'ok', '是', '对', '谢谢', '好吧', '都行', '???']
    
    for content in low_signal:
        d = ctrl.evaluate(content)
        assert not d.admitted, f"'{content}' should be rejected, got admitted with score={d.score}"
    
    print(f"  Rejected {len(low_signal)} low-signal inputs")
    print("✓ test_low_signal_rejected")

def test_high_value_always_admitted():
    """Critical health/safety info always gets in"""
    ctrl = AdmissionController()
    
    high_value = [
        '我对花生过敏',
        '我的生日是3月15号',
        '下周二要做手术',
        '紧急联系人是我妈',
    ]
    
    for content in high_value:
        d = ctrl.evaluate(content)
        assert d.admitted, f"'{content}' should be admitted (high value), got score={d.score}"
        assert d.reason == 'high_value_pattern'
    
    print(f"  Admitted {len(high_value)} high-value inputs")
    print("✓ test_high_value_always_admitted")

def test_personal_info_admitted():
    """Personal information scores highly"""
    ctrl = AdmissionController()
    
    personal = [
        '我叫小明，今年25岁',
        '我住在北京市朝阳区',
        '我在字节跳动公司工作',
        '我最喜欢吃火锅',
    ]
    
    for content in personal:
        d = ctrl.evaluate(content)
        assert d.admitted, f"'{content}' should be admitted, got score={d.score:.3f}"
        assert d.score >= 0.3
    
    print("✓ test_personal_info_admitted")

def test_novelty_decreases_with_repetition():
    """Repeated info gets lower novelty score"""
    ctrl = AdmissionController()
    
    existing = [
        MemoryEntry(id='1', content='我住在北京', source=MemorySource.USER,
                   entities=[], timestamp=0, user_id='u1')
    ]
    
    # New info → high novelty
    d_new = ctrl.evaluate('我喜欢打篮球', existing=existing)
    # Similar info → low novelty
    d_repeat = ctrl.evaluate('我住在北京市', existing=existing)
    
    assert d_new.scores.novelty > d_repeat.scores.novelty, \
        f"New info novelty ({d_new.scores.novelty:.3f}) should > repeat ({d_repeat.scores.novelty:.3f})"
    
    print(f"  New info novelty={d_new.scores.novelty:.3f}, repeat={d_repeat.scores.novelty:.3f}")
    print("✓ test_novelty_decreases_with_repetition")

def test_actionability_future_events():
    """Future events score high on actionability"""
    ctrl = AdmissionController()
    
    # High actionability
    d_future = ctrl.evaluate('明天下午2点要面试')
    # Low actionability
    d_past = ctrl.evaluate('昨天看了一部电影')
    
    assert d_future.scores.actionability > d_past.scores.actionability
    print(f"  Future event actionability={d_future.scores.actionability:.3f}")
    print(f"  Past event actionability={d_past.scores.actionability:.3f}")
    print("✓ test_actionability_future_events")

def test_source_affects_specificity():
    """USER source gets higher specificity than SYSTEM"""
    ctrl = AdmissionController()
    
    content = '系统更新完成'
    d_user = ctrl.evaluate(content, source=MemorySource.USER)
    d_system = ctrl.evaluate(content, source=MemorySource.SYSTEM)
    
    assert d_user.scores.specificity > d_system.scores.specificity
    print("✓ test_source_affects_specificity")

def test_context_relevance():
    """Content relevant to conversation context scores higher"""
    ctrl = AdmissionController()
    
    context = '我们在讨论工作和项目进度'
    
    d_relevant = ctrl.evaluate('这个项目下周要交付', context=context)
    d_irrelevant = ctrl.evaluate('今天天气真好', context=context)
    
    assert d_relevant.scores.relevance > d_irrelevant.scores.relevance
    print(f"  Relevant: {d_relevant.scores.relevance:.3f}, Irrelevant: {d_irrelevant.scores.relevance:.3f}")
    print("✓ test_context_relevance")

def test_threshold_boundary():
    """Scores around threshold behave correctly"""
    ctrl = AdmissionController(threshold=0.5)
    
    # Generic short content → likely below 0.5
    d_low = ctrl.evaluate('路上堵车')
    # Rich personal content → likely above 0.5
    d_high = ctrl.evaluate('我叫张三，住在上海市浦东新区，在腾讯公司工作')
    
    # d_high should be admitted with higher threshold
    assert d_high.admitted
    print(f"  Low: score={d_low.score:.3f} admitted={d_low.admitted}")
    print(f"  High: score={d_high.score:.3f} admitted={d_high.admitted}")
    print("✓ test_threshold_boundary")

def test_batch_evaluation():
    """Batch evaluate multiple candidates"""
    ctrl = AdmissionController()
    
    candidates = [
        ('嗯', MemorySource.USER),
        ('我对花生过敏', MemorySource.USER),
        ('明天要面试', MemorySource.USER),
        ('好的', MemorySource.AGENT),
        ('用户提到了过敏史', MemorySource.AGENT),
    ]
    
    decisions = [ctrl.evaluate(c, s) for c, s in candidates]
    
    admitted_count = sum(1 for d in decisions if d.admitted)
    rejected_count = sum(1 for d in decisions if not d.admitted)
    
    # '嗯' and '好的' should be rejected, rest admitted
    assert rejected_count >= 2
    assert admitted_count >= 2
    
    print(f"  Batch: {admitted_count} admitted, {rejected_count} rejected out of {len(candidates)}")
    print("✓ test_batch_evaluation")

def test_edge_cases():
    """Edge cases: empty string, single char, very long"""
    ctrl = AdmissionController()
    
    # Empty/single char → low signal
    d_empty = ctrl.evaluate('')
    assert not d_empty.admitted
    
    d_single = ctrl.evaluate('嗯')
    assert not d_single.admitted
    
    # Very long informative text → should be admitted
    long_text = '我叫李明，今年28岁，住在深圳市南山区，在腾讯公司做产品经理，喜欢跑步和摄影'
    d_long = ctrl.evaluate(long_text)
    assert d_long.admitted
    assert d_long.score > 0.5
    
    print("✓ test_edge_cases")

def test_integration_with_store_concept():
    """Simulates integration: extract → evaluate → ADD only admitted"""
    ctrl = AdmissionController()
    
    messages = [
        '嗯嗯，好的',
        '我叫小红，在阿里巴巴工作',
        '哈哈',
        '明天下午3点有个重要会议',
        '好',
        '我对海鲜过敏，特别是虾',
    ]
    
    admitted = []
    rejected = []
    
    for msg in messages:
        d = ctrl.evaluate(msg)
        if d.admitted:
            admitted.append(msg)
        else:
            rejected.append(msg)
    
    print(f"  Admitted ({len(admitted)}): {admitted}")
    print(f"  Rejected ({len(rejected)}): {rejected}")
    
    # Should admit the informative ones
    assert any('阿里' in m for m in admitted)
    assert any('海鲜' in m or '过敏' in m for m in admitted)
    # Should reject fillers
    assert any('嗯' in m or '好' == m for m in rejected)
    
    # Key metric: we reduced store writes significantly
    reduction = len(rejected) / len(messages) * 100
    print(f"  Storage reduction: {reduction:.0f}% of messages filtered out")
    assert reduction >= 30, "Should filter at least 30% of low-value messages"
    print("✓ test_integration_with_store_concept")


if __name__ == '__main__':
    tests = [
        test_low_signal_rejected,
        test_high_value_always_admitted,
        test_personal_info_admitted,
        test_novelty_decreases_with_repetition,
        test_actionability_future_events,
        test_source_affects_specificity,
        test_context_relevance,
        test_threshold_boundary,
        test_batch_evaluation,
        test_edge_cases,
        test_integration_with_store_concept,
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
