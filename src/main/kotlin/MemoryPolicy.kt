package com.memory.policy

import com.memory.context.MemoryCandidate
import com.memory.context.QueryComplexity
import com.memory.governance.GovernanceLayer
import com.memory.provenance.TrustLevel

/**
 * RL-Ready Memory Policy Interface
 * 
 * 核心思想 (from AgeMem arXiv:2601.01885):
 * "将记忆管理抽象为一组 tool-based actions，通过强化学习训练决策策略"
 * 
 * 设计哲学：
 * - 所有决策点统一为 Policy.decide() → Action
 * - 当前实现为 rule-based (RuleBasedPolicy)
 * - 接口兼容 RL 训练：state → action → reward
 * - 未来可插入 GRPO/PPO 训练的神经网络策略
 * 
 * Action Space (from AtomMem arXiv:2601.08323 CRUD + AgeMem):
 * - STORE: 存储新记忆
 * - RETRIEVE: 检索相关记忆
 * - UPDATE: 更新已有记忆
 * - DELETE: 删除/遗忘记忆
 * - SKIP: 不做任何操作
 * - SUMMARIZE: 将多条记忆压缩为摘要
 * - FILTER: 过滤注入的上下文
 * 
 * 与正交维度设计的关系：
 * - Policy 是查询时的决策函数，不改变底层存储
 * - Action 是对信息本体的操作指令
 * - Reward 信号可以来自下游任务 (对话质量/记忆命中率)
 */

// === Action Space ===

enum class MemoryAction {
    STORE,      // 存储新记忆到长期存储
    RETRIEVE,   // 从存储中检索
    UPDATE,     // 更新已有记忆 (覆盖/追加)
    DELETE,     // 标记为遗忘
    SKIP,       // 不操作 (对话轮次不含值得记忆的信息)
    SUMMARIZE,  // 压缩多条为摘要
    FILTER      // 过滤上下文注入 (决定注入哪些)
}

// === State Representation ===

/**
 * 决策状态 — Policy 的输入
 * 
 * 包含当前轮次的所有决策所需信息。
 * 设计为 immutable，方便 RL 训练时的 state replay。
 */
data class PolicyState(
    // 当前对话信息
    val currentMessage: String,
    val turnIndex: Int,
    val conversationLength: Int,
    
    // 检索结果
    val retrievedMemories: List<RetrievedMemoryState> = emptyList(),
    
    // 提取结果 (LLM extraction output)
    val extractedFacts: List<ExtractedFactState> = emptyList(),
    
    // 元信息
    val queryComplexity: QueryComplexity = QueryComplexity.MEDIUM,
    val tokenBudgetRemaining: Int = 1200,
    val existingMemoryCount: Int = 0,
    
    // 历史 action (用于序列决策)
    val previousActions: List<MemoryAction> = emptyList()
)

data class RetrievedMemoryState(
    val id: String,
    val content: String,
    val score: Double,
    val layer: GovernanceLayer,
    val trustLevel: TrustLevel,
    val accessCount: Int = 0,
    val daysSinceCreation: Int = 0,
    val daysSinceLastAccess: Int = 0
)

data class ExtractedFactState(
    val content: String,
    val confidence: Double,
    val matchingMemoryId: String? = null,  // 如果与已有记忆重复
    val matchScore: Double = 0.0           // 与已有记忆的匹配度
)

// === Action Output ===

/**
 * Policy 决策输出
 * 
 * 每个决策点产生一个 ActionDecision，包含：
 * - action: 要执行的动作
 * - targets: 动作的目标 (记忆ID / 新内容)
 * - confidence: 决策置信度 (RL 训练时用作 exploration signal)
 * - reason: 可解释性 (rule-based 时有具体原因)
 */
data class ActionDecision(
    val action: MemoryAction,
    val targets: List<String> = emptyList(),
    val confidence: Double = 1.0,
    val reason: String = "",
    val metadata: Map<String, Any> = emptyMap()
)

// === Policy Interface ===

/**
 * 核心策略接口
 * 
 * 所有记忆管理决策都通过此接口。
 * 当前默认实现为 RuleBasedPolicy，
 * 未来可替换为 NeuralPolicy (RL-trained)。
 */
interface MemoryPolicy {
    /**
     * 对当前轮次进行决策
     * 
     * @param state 当前状态
     * @param phase 决策阶段 (extraction/injection/maintenance)
     * @return 一组决策动作
     */
    fun decide(state: PolicyState, phase: DecisionPhase): List<ActionDecision>
    
    /**
     * 接收 reward 信号 (RL 训练用)
     * 
     * @param state 产生动作时的状态
     * @param actions 执行的动作
     * @param reward 奖励信号
     */
    fun receiveReward(state: PolicyState, actions: List<ActionDecision>, reward: Double) {
        // 默认无操作 (rule-based 不需要)
    }
    
    /**
     * 获取策略的统计信息
     */
    fun getStats(): PolicyStats = PolicyStats()
}

enum class DecisionPhase {
    EXTRACTION,    // 从对话中提取信息时的决策 (STORE/UPDATE/SKIP/DELETE)
    INJECTION,     // 向 prompt 注入记忆时的决策 (RETRIEVE/FILTER)
    MAINTENANCE    // 定期维护时的决策 (SUMMARIZE/DELETE)
}

data class PolicyStats(
    val totalDecisions: Int = 0,
    val actionDistribution: Map<MemoryAction, Int> = emptyMap(),
    val avgConfidence: Double = 0.0,
    val rewardHistory: List<Double> = emptyList()
)

// === Rule-Based Policy (Default) ===

/**
 * 规则策略 — 基于当前 v8 引擎逻辑的显式规则
 * 
 * 这是当前系统的决策逻辑，显式化为 Policy 接口。
 * 好处：
 * 1. 可追踪、可解释
 * 2. 可以和 RL 策略做 A/B 对比
 * 3. 作为 RL 训练的初始化策略 (behavior cloning warm-start)
 */
class RuleBasedPolicy(
    private val skipThreshold: Double = 0.85,    // 匹配度超过此值则 SKIP
    private val updateThreshold: Double = 0.60,  // 匹配度在此区间则 UPDATE
    private val minConfidenceForStore: Double = 0.3,
    private val maxContextMemories: Int = 8,
    private val summarizeThreshold: Int = 50     // 超过此数量触发 SUMMARIZE
) : MemoryPolicy {
    
    private var stats = MutablePolicyStats()
    
    override fun decide(state: PolicyState, phase: DecisionPhase): List<ActionDecision> {
        return when (phase) {
            DecisionPhase.EXTRACTION -> decideExtraction(state)
            DecisionPhase.INJECTION -> decideInjection(state)
            DecisionPhase.MAINTENANCE -> decideMaintenance(state)
        }
    }
    
    private fun decideExtraction(state: PolicyState): List<ActionDecision> {
        val decisions = mutableListOf<ActionDecision>()
        
        for (fact in state.extractedFacts) {
            val decision = when {
                // 高匹配 → SKIP (重复信息)
                fact.matchScore >= skipThreshold -> {
                    ActionDecision(
                        action = MemoryAction.SKIP,
                        targets = listOf(fact.matchingMemoryId ?: ""),
                        confidence = fact.matchScore,
                        reason = "High similarity (${fact.matchScore}) to existing memory"
                    )
                }
                // 中匹配 → UPDATE (补充/修正)
                fact.matchScore >= updateThreshold && fact.matchingMemoryId != null -> {
                    ActionDecision(
                        action = MemoryAction.UPDATE,
                        targets = listOf(fact.matchingMemoryId, fact.content),
                        confidence = 0.7 + (fact.matchScore - updateThreshold) * 0.5,
                        reason = "Partial match (${fact.matchScore}) — update existing"
                    )
                }
                // 低匹配 + 高置信 → STORE (新信息)
                fact.confidence >= minConfidenceForStore -> {
                    ActionDecision(
                        action = MemoryAction.STORE,
                        targets = listOf(fact.content),
                        confidence = fact.confidence,
                        reason = "New information with confidence ${fact.confidence}"
                    )
                }
                // 低匹配 + 低置信 → SKIP (不确定信息)
                else -> {
                    ActionDecision(
                        action = MemoryAction.SKIP,
                        confidence = 1.0 - fact.confidence,
                        reason = "Low confidence (${fact.confidence}) — skip"
                    )
                }
            }
            decisions.add(decision)
            stats.record(decision.action)
        }
        
        // 如果没有提取到任何信息，也是 SKIP
        if (state.extractedFacts.isEmpty()) {
            decisions.add(ActionDecision(
                action = MemoryAction.SKIP,
                confidence = 0.9,
                reason = "No extractable facts in this turn"
            ))
            stats.record(MemoryAction.SKIP)
        }
        
        return decisions
    }
    
    private fun decideInjection(state: PolicyState): List<ActionDecision> {
        val decisions = mutableListOf<ActionDecision>()
        
        if (state.retrievedMemories.isEmpty()) {
            decisions.add(ActionDecision(
                action = MemoryAction.SKIP,
                confidence = 1.0,
                reason = "No memories retrieved"
            ))
            return decisions
        }
        
        // FILTER: 决定注入哪些 (委托给 AdaptiveContextSelector 的逻辑)
        val toInject = state.retrievedMemories.filter { it.score >= 0.005 }
        
        if (toInject.isNotEmpty()) {
            decisions.add(ActionDecision(
                action = MemoryAction.FILTER,
                targets = toInject.map { it.id },
                confidence = toInject.first().score,
                reason = "Injecting ${toInject.size}/${state.retrievedMemories.size} memories"
            ))
        }
        
        stats.record(MemoryAction.FILTER)
        return decisions
    }
    
    private fun decideMaintenance(state: PolicyState): List<ActionDecision> {
        val decisions = mutableListOf<ActionDecision>()
        
        // SUMMARIZE: 当记忆过多时
        if (state.existingMemoryCount > summarizeThreshold) {
            decisions.add(ActionDecision(
                action = MemoryAction.SUMMARIZE,
                confidence = 0.8,
                reason = "Memory count (${state.existingMemoryCount}) exceeds threshold ($summarizeThreshold)"
            ))
            stats.record(MemoryAction.SUMMARIZE)
        }
        
        // DELETE: 检查是否有低保留率的记忆 (委托给 SelectiveForgetting)
        val staleMemories = state.retrievedMemories.filter { 
            it.daysSinceLastAccess > 30 && it.layer == GovernanceLayer.OPERATIONAL
        }
        if (staleMemories.isNotEmpty()) {
            decisions.add(ActionDecision(
                action = MemoryAction.DELETE,
                targets = staleMemories.map { it.id },
                confidence = 0.6,
                reason = "${staleMemories.size} stale operational memories"
            ))
            stats.record(MemoryAction.DELETE)
        }
        
        return decisions
    }
    
    override fun getStats(): PolicyStats = PolicyStats(
        totalDecisions = stats.total,
        actionDistribution = stats.distribution.toMap(),
        avgConfidence = stats.avgConfidence
    )
    
    private class MutablePolicyStats {
        var total = 0
        val distribution = mutableMapOf<MemoryAction, Int>()
        private var confidenceSum = 0.0
        val avgConfidence get() = if (total > 0) confidenceSum / total else 0.0
        
        fun record(action: MemoryAction, confidence: Double = 1.0) {
            total++
            distribution[action] = (distribution[action] ?: 0) + 1
            confidenceSum += confidence
        }
    }
}

// === RL Policy Stub (Future) ===

/**
 * 神经网络策略占位 — 未来 RL 训练后替换
 * 
 * Training approach (from AgeMem):
 * 1. Behavior cloning warm-start from RuleBasedPolicy traces
 * 2. Step-wise GRPO with memory quality reward
 * 3. Progressive RL: single-action → multi-action → full episode
 * 
 * Reward design:
 * - Retrieval reward: 检索命中率 × 相关性
 * - Extraction reward: 去重率 + 信息增益
 * - Context reward: 下游任务质量 (BLEU/ROUGE vs ground truth)
 * - Efficiency reward: 1 - (tokens_used / token_budget)
 */
class NeuralPolicy(
    private val fallback: MemoryPolicy = RuleBasedPolicy()
) : MemoryPolicy {
    
    // Placeholder: 在实际实现中，这里会是一个轻量级 transformer
    // 输入: state embedding → 输出: action distribution
    private var modelLoaded = false
    
    override fun decide(state: PolicyState, phase: DecisionPhase): List<ActionDecision> {
        // TODO: 当模型训练完成后，替换为神经网络推理
        // 目前 fallback 到 rule-based
        return if (modelLoaded) {
            neuralDecide(state, phase)
        } else {
            fallback.decide(state, phase)
        }
    }
    
    override fun receiveReward(state: PolicyState, actions: List<ActionDecision>, reward: Double) {
        // TODO: 收集 (state, action, reward) tuples 用于训练
        // 当前先存入 replay buffer
        replayBuffer.add(Experience(state, actions, reward))
    }
    
    private fun neuralDecide(state: PolicyState, phase: DecisionPhase): List<ActionDecision> {
        // Placeholder for neural inference
        return fallback.decide(state, phase)
    }
    
    // Experience replay buffer for future RL training
    private val replayBuffer = mutableListOf<Experience>()
    
    private data class Experience(
        val state: PolicyState,
        val actions: List<ActionDecision>,
        val reward: Double
    )
}

// === Policy Factory ===

object PolicyFactory {
    
    private var currentPolicy: MemoryPolicy = RuleBasedPolicy()
    
    fun getPolicy(): MemoryPolicy = currentPolicy
    
    fun setPolicy(policy: MemoryPolicy) {
        currentPolicy = policy
    }
    
    /**
     * 创建 A/B 测试策略
     * 以概率 p 使用实验策略，1-p 使用基线策略
     */
    fun createABPolicy(
        baseline: MemoryPolicy = RuleBasedPolicy(),
        experiment: MemoryPolicy,
        experimentRatio: Double = 0.1
    ): MemoryPolicy = ABTestPolicy(baseline, experiment, experimentRatio)
}

private class ABTestPolicy(
    private val baseline: MemoryPolicy,
    private val experiment: MemoryPolicy,
    private val experimentRatio: Double
) : MemoryPolicy {
    
    override fun decide(state: PolicyState, phase: DecisionPhase): List<ActionDecision> {
        val useExperiment = Math.random() < experimentRatio
        return if (useExperiment) {
            experiment.decide(state, phase)
        } else {
            baseline.decide(state, phase)
        }
    }
    
    override fun receiveReward(state: PolicyState, actions: List<ActionDecision>, reward: Double) {
        // 两个策略都收集 reward 数据用于对比
        baseline.receiveReward(state, actions, reward)
        experiment.receiveReward(state, actions, reward)
    }
}
