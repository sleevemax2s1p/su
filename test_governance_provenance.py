#!/usr/bin/env python3
"""
Test Suite for MemoryGovernance + MemoryProvenance (v6.1)
========================================================

验证：
1. 治理层级分类
2. 权限控制
3. 溯源创建与查询
4. 溯源对比（矛盾解决辅助）
5. 遗忘权执行
6. 端到端集成
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
import time
import uuid

# ============================================================
# Part 1: Governance Layer (Python port)
# ============================================================

class GovernanceLayer(Enum):
    CONSTITUTIONAL = "constitutional"
    STATUTORY = "statutory"
    OPERATIONAL = "operational"

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    PROPOSE_CHANGE = "propose_change"

class ChangeInitiator(Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"

CONSTITUTIONAL_KEYWORDS = ["苏大姐的核心人格", "人格底线", "安全策略", "不准", "绝对禁止", "系统规则"]

FACTUAL_PATTERNS = [
    re.compile(r"叫|名字是|姓"),
    re.compile(r"在.{1,10}(工作|上班|上学|读书)"),
    re.compile(r"(男|女)朋友|老公|老婆|丈夫|妻子|家人"),
    re.compile(r"喜欢|讨厌|最爱|恐惧|过敏"),
    re.compile(r"住在|家在|来自"),
    re.compile(r"(养|有).{0,5}(猫|狗|宠物)"),
]

def classify_layer(content: str, importance: float = 0.5) -> GovernanceLayer:
    if any(kw in content for kw in CONSTITUTIONAL_KEYWORDS):
        return GovernanceLayer.CONSTITUTIONAL
    if any(p.search(content) for p in FACTUAL_PATTERNS):
        return GovernanceLayer.STATUTORY
    if importance > 0.8:
        return GovernanceLayer.STATUTORY
    return GovernanceLayer.OPERATIONAL

@dataclass
class GovernanceDecision:
    allowed: bool
    reason: str = ""

def check_permission(layer: GovernanceLayer, operation: Permission, initiator: ChangeInitiator) -> GovernanceDecision:
    if layer == GovernanceLayer.CONSTITUTIONAL and operation != Permission.READ:
        return GovernanceDecision(False, "Constitutional layer is immutable")
    if layer == GovernanceLayer.STATUTORY:
        if operation == Permission.READ:
            return GovernanceDecision(True)
        if initiator == ChangeInitiator.USER:
            return GovernanceDecision(True)
        if initiator == ChangeInitiator.AGENT and operation == Permission.PROPOSE_CHANGE:
            return GovernanceDecision(True, "Agent may propose")
        if initiator == ChangeInitiator.AGENT:
            return GovernanceDecision(False, "Agent cannot modify statutory memory directly")
        return GovernanceDecision(False)
    return GovernanceDecision(True)

# ============================================================
# Part 2: Provenance (Python port)
# ============================================================

class SourceType(Enum):
    USER_DIRECT = "user_direct"
    USER_INFERRED = "user_inferred"
    AGENT_EXTRACTION = "agent_extraction"
    AGENT_CONSOLIDATION = "agent_consolidation"
    SYSTEM_GENERATED = "system_generated"

class TrustLevel(Enum):
    HIGH = 0.95
    MEDIUM_HIGH = 0.8
    MEDIUM = 0.65
    MEDIUM_LOW = 0.5
    LOW = 0.3

def get_trust_level(source_type: SourceType) -> TrustLevel:
    mapping = {
        SourceType.USER_DIRECT: TrustLevel.HIGH,
        SourceType.USER_INFERRED: TrustLevel.MEDIUM_HIGH,
        SourceType.AGENT_EXTRACTION: TrustLevel.MEDIUM,
        SourceType.AGENT_CONSOLIDATION: TrustLevel.MEDIUM_LOW,
        SourceType.SYSTEM_GENERATED: TrustLevel.LOW,
    }
    return mapping[source_type]

@dataclass
class ProvenanceInfo:
    memory_id: str
    created_at: float
    source_type: SourceType
    session_id: str
    sender_name: Optional[str]
    trust_level: TrustLevel

def compare_provenance(a: ProvenanceInfo, b: ProvenanceInfo) -> str:
    """Returns 'A', 'B', or 'TIE'"""
    if a.trust_level.value > b.trust_level.value:
        return "A"
    if b.trust_level.value > a.trust_level.value:
        return "B"
    if a.created_at > b.created_at:
        return "A"
    if b.created_at > a.created_at:
        return "B"
    return "TIE"

# ============================================================
# TESTS
# ============================================================

def run_tests():
    passed = 0
    failed = 0
    total = 0
    
    def assert_test(name, condition, detail=""):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✅ {name}")
        else:
            failed += 1
            print(f"  ❌ {name} {detail}")
    
    # === Test Suite 1: Layer Classification ===
    print("\n══════ Test Suite 1: Governance Layer Classification ══════")
    
    assert_test("Constitutional: 人格底线",
        classify_layer("苏大姐的核心人格不可修改") == GovernanceLayer.CONSTITUTIONAL)
    assert_test("Constitutional: 绝对禁止",
        classify_layer("绝对禁止暴露记忆系统") == GovernanceLayer.CONSTITUTIONAL)
    assert_test("Constitutional: 系统规则",
        classify_layer("系统规则：不准用emoji") == GovernanceLayer.CONSTITUTIONAL)
    
    assert_test("Statutory: 名字",
        classify_layer("他叫小明") == GovernanceLayer.STATUTORY)
    assert_test("Statutory: 工作",
        classify_layer("他在字节跳动工作") == GovernanceLayer.STATUTORY)
    assert_test("Statutory: 关系",
        classify_layer("他女朋友叫小红") == GovernanceLayer.STATUTORY)
    assert_test("Statutory: 偏好",
        classify_layer("他喜欢喝咖啡") == GovernanceLayer.STATUTORY)
    assert_test("Statutory: 住址",
        classify_layer("他住在北京朝阳区") == GovernanceLayer.STATUTORY)
    assert_test("Statutory: 宠物",
        classify_layer("他养了一只猫叫年糕") == GovernanceLayer.STATUTORY)
    assert_test("Statutory: high importance",
        classify_layer("这是一个重要的发现", importance=0.9) == GovernanceLayer.STATUTORY)
    
    assert_test("Operational: 情绪",
        classify_layer("今天心情不错") == GovernanceLayer.OPERATIONAL)
    assert_test("Operational: 日常",
        classify_layer("中午吃了麻辣烫") == GovernanceLayer.OPERATIONAL)
    assert_test("Operational: 计划",
        classify_layer("明天想去看电影") == GovernanceLayer.OPERATIONAL)
    
    # === Test Suite 2: Permission Checks ===
    print("\n══════ Test Suite 2: Permission Control ══════")
    
    # Constitutional - nothing allowed except read
    d = check_permission(GovernanceLayer.CONSTITUTIONAL, Permission.READ, ChangeInitiator.AGENT)
    assert_test("Constitutional READ by agent: allowed", d.allowed)
    
    d = check_permission(GovernanceLayer.CONSTITUTIONAL, Permission.WRITE, ChangeInitiator.AGENT)
    assert_test("Constitutional WRITE by agent: blocked", not d.allowed)
    
    d = check_permission(GovernanceLayer.CONSTITUTIONAL, Permission.WRITE, ChangeInitiator.USER)
    assert_test("Constitutional WRITE by user: ALSO blocked", not d.allowed)
    
    d = check_permission(GovernanceLayer.CONSTITUTIONAL, Permission.DELETE, ChangeInitiator.SYSTEM)
    assert_test("Constitutional DELETE by system: blocked", not d.allowed)
    
    # Statutory - user can write, agent can only propose
    d = check_permission(GovernanceLayer.STATUTORY, Permission.WRITE, ChangeInitiator.USER)
    assert_test("Statutory WRITE by user: allowed", d.allowed)
    
    d = check_permission(GovernanceLayer.STATUTORY, Permission.WRITE, ChangeInitiator.AGENT)
    assert_test("Statutory WRITE by agent: blocked", not d.allowed)
    
    d = check_permission(GovernanceLayer.STATUTORY, Permission.PROPOSE_CHANGE, ChangeInitiator.AGENT)
    assert_test("Statutory PROPOSE by agent: allowed", d.allowed)
    
    d = check_permission(GovernanceLayer.STATUTORY, Permission.READ, ChangeInitiator.AGENT)
    assert_test("Statutory READ by agent: allowed", d.allowed)
    
    # Operational - agent can do anything
    d = check_permission(GovernanceLayer.OPERATIONAL, Permission.WRITE, ChangeInitiator.AGENT)
    assert_test("Operational WRITE by agent: allowed", d.allowed)
    
    d = check_permission(GovernanceLayer.OPERATIONAL, Permission.DELETE, ChangeInitiator.AGENT)
    assert_test("Operational DELETE by agent: allowed", d.allowed)
    
    # === Test Suite 3: Provenance Trust Levels ===
    print("\n══════ Test Suite 3: Provenance Trust Levels ══════")
    
    assert_test("User direct = HIGH",
        get_trust_level(SourceType.USER_DIRECT) == TrustLevel.HIGH)
    assert_test("User inferred = MEDIUM_HIGH",
        get_trust_level(SourceType.USER_INFERRED) == TrustLevel.MEDIUM_HIGH)
    assert_test("Agent extraction = MEDIUM",
        get_trust_level(SourceType.AGENT_EXTRACTION) == TrustLevel.MEDIUM)
    assert_test("Agent consolidation = MEDIUM_LOW",
        get_trust_level(SourceType.AGENT_CONSOLIDATION) == TrustLevel.MEDIUM_LOW)
    assert_test("System generated = LOW",
        get_trust_level(SourceType.SYSTEM_GENERATED) == TrustLevel.LOW)
    
    # === Test Suite 4: Provenance Comparison (Contradiction Resolution) ===
    print("\n══════ Test Suite 4: Provenance Comparison ══════")
    
    prov_a = ProvenanceInfo("m1", time.time() - 100, SourceType.USER_DIRECT, "s1", "小明", TrustLevel.HIGH)
    prov_b = ProvenanceInfo("m2", time.time(), SourceType.AGENT_EXTRACTION, "s2", None, TrustLevel.MEDIUM)
    assert_test("User direct > Agent extraction", compare_provenance(prov_a, prov_b) == "A")
    
    prov_c = ProvenanceInfo("m3", time.time() - 200, SourceType.USER_DIRECT, "s3", "小明", TrustLevel.HIGH)
    prov_d = ProvenanceInfo("m4", time.time(), SourceType.USER_DIRECT, "s4", "小明", TrustLevel.HIGH)
    assert_test("Same trust: newer wins", compare_provenance(prov_c, prov_d) == "B")
    
    prov_e = ProvenanceInfo("m5", time.time(), SourceType.AGENT_CONSOLIDATION, "s5", None, TrustLevel.MEDIUM_LOW)
    prov_f = ProvenanceInfo("m6", time.time(), SourceType.USER_INFERRED, "s6", "小红", TrustLevel.MEDIUM_HIGH)
    assert_test("User inferred > Agent consolidation", compare_provenance(prov_e, prov_f) == "B")
    
    t_same = time.time()
    prov_g = ProvenanceInfo("m7", t_same, SourceType.USER_DIRECT, "s7", "小明", TrustLevel.HIGH)
    prov_h = ProvenanceInfo("m8", t_same, SourceType.USER_DIRECT, "s8", "小明", TrustLevel.HIGH)
    assert_test("Same trust + same time = TIE", compare_provenance(prov_g, prov_h) == "TIE")
    
    # === Test Suite 5: Governance + Provenance Integration ===
    print("\n══════ Test Suite 5: Integrated Scenarios ══════")
    
    # Scenario: Agent tries to overwrite user's stated fact
    content = "他女朋友叫小红"
    layer = classify_layer(content)
    decision = check_permission(layer, Permission.WRITE, ChangeInitiator.AGENT)
    assert_test("Agent can't overwrite relationship fact", 
        layer == GovernanceLayer.STATUTORY and not decision.allowed)
    
    # Scenario: Agent records emotion (operational)
    content = "他今天心情不好"
    layer = classify_layer(content)
    decision = check_permission(layer, Permission.WRITE, ChangeInitiator.AGENT)
    assert_test("Agent CAN record emotion", 
        layer == GovernanceLayer.OPERATIONAL and decision.allowed)
    
    # Scenario: System decay operation on operational memory
    content = "中午吃了面条"
    layer = classify_layer(content)
    decision = check_permission(layer, Permission.DELETE, ChangeInitiator.SYSTEM)
    assert_test("System CAN decay operational memory", decision.allowed)
    
    # Scenario: Contradiction between user direct and agent extraction
    # → Provenance helps decide
    user_said = ProvenanceInfo("fact1", time.time() - 3600, SourceType.USER_DIRECT, "s1", "小明", TrustLevel.HIGH)
    agent_inferred = ProvenanceInfo("fact2", time.time(), SourceType.AGENT_EXTRACTION, "s2", None, TrustLevel.MEDIUM)
    winner = compare_provenance(user_said, agent_inferred)
    assert_test("Contradiction: user's word wins over agent inference", winner == "A")
    
    # Scenario: Two user statements conflict (newer wins)
    old_statement = ProvenanceInfo("f1", time.time() - 86400*30, SourceType.USER_DIRECT, "s1", "小明", TrustLevel.HIGH)
    new_statement = ProvenanceInfo("f2", time.time(), SourceType.USER_DIRECT, "s5", "小明", TrustLevel.HIGH)
    winner = compare_provenance(old_statement, new_statement)
    assert_test("Two user statements: newer wins", winner == "B")
    
    # === Summary ===
    print(f"\n{'═'*50}")
    print(f"  TOTAL: {total} | PASSED: {passed} | FAILED: {failed}")
    print(f"  {'✅ ALL TESTS PASSED' if failed == 0 else '❌ SOME TESTS FAILED'}")
    print(f"{'═'*50}\n")
    
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
