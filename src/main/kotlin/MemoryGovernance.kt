package com.memory.governance

import com.memory.store.KnowledgeStore
import java.text.SimpleDateFormat
import java.util.Date

/**
 * 记忆治理层 (Memory Governance)
 * 
 * 灵感来源：
 * - Animesis CMA 四层治理架构 (§14.22.1)
 * - Context Engineering 五大质量标准中的 Isolation (§14.22.3)
 * - Letta Context Constitution: 治理上升到「宪法」级别 (§14.21.1)
 * 
 * 核心理念：
 * 并非所有记忆规则具有相同约束力。某些规则对其他规则具有约束力。
 * Agent 自身不可完全信任——需要有些东西它改不了。
 * 
 * 三层治理模型（简化自 Animesis 四层）：
 * 1. Constitutional Layer（宪法层）—— 不可被 Agent 修改
 *    - 人格核心设定
 *    - 安全底线
 *    - 用户权利（如遗忘权）
 * 
 * 2. Statutory Layer（法规层）—— 仅可通过显式用户授权修改
 *    - 长期事实记忆
 *    - 人际关系定义
 *    - 核心偏好
 * 
 * 3. Operational Layer（运营层）—— Agent 可自主修改
 *    - 情绪状态
 *    - 短期上下文
 *    - 交互模式学习
 *    - 会话级信息
 */
class MemoryGovernance(
    private val store: KnowledgeStore
) {
    // 宪法层规则（硬编码，不可运行时修改）
    private val constitutionalRules = ConstitutionalRules()
    
    // 各层级的访问控制表
    private val layerPermissions = mapOf(
        GovernanceLayer.CONSTITUTIONAL to setOf(Permission.READ),  // Agent 只能读
        GovernanceLayer.STATUTORY to setOf(Permission.READ, Permission.PROPOSE_CHANGE),  // Agent 可提议，用户确认后才生效
        GovernanceLayer.OPERATIONAL to setOf(Permission.READ, Permission.WRITE, Permission.DELETE)  // Agent 完全控制
    )
    
    private fun log(msg: String) {
        val ts = SimpleDateFormat("HH:mm:ss.SSS").format(Date())
        println("[$ts][Governance] $msg")
    }

    /**
     * 判断一条记忆属于哪个治理层级
     * 
     * 分层逻辑：
     * - 包含人格定义关键词 → Constitutional
     * - 包含长期事实/关系 → Statutory
     * - 其他 → Operational
     */
    fun classifyLayer(content: String, metadata: MemoryMetadata? = null): GovernanceLayer {
        // Constitutional: 涉及 Agent 身份或安全底线
        if (isConstitutionalContent(content)) {
            return GovernanceLayer.CONSTITUTIONAL
        }
        
        // Statutory: 长期事实性记忆、关系定义、核心偏好
        if (isStatutoryContent(content, metadata)) {
            return GovernanceLayer.STATUTORY
        }
        
        // Operational: 其他一切
        return GovernanceLayer.OPERATIONAL
    }

    /**
     * 检查一个操作是否被当前治理规则允许
     */
    fun checkPermission(
        layer: GovernanceLayer,
        operation: Permission,
        initiator: ChangeInitiator
    ): GovernanceDecision {
        // 宪法层：任何运行时修改都被拒绝
        if (layer == GovernanceLayer.CONSTITUTIONAL && operation != Permission.READ) {
            return GovernanceDecision(
                allowed = false,
                reason = "Constitutional layer is immutable at runtime",
                suggestion = "This memory is protected by constitutional rules. It cannot be modified."
            )
        }
        
        // 法规层：只有用户发起才能写入；Agent 只能提议
        if (layer == GovernanceLayer.STATUTORY) {
            return when {
                operation == Permission.READ -> GovernanceDecision(allowed = true)
                initiator == ChangeInitiator.USER -> GovernanceDecision(allowed = true)
                initiator == ChangeInitiator.AGENT && operation == Permission.PROPOSE_CHANGE -> 
                    GovernanceDecision(
                        allowed = true,
                        reason = "Agent may propose changes to statutory memory, pending user confirmation"
                    )
                initiator == ChangeInitiator.AGENT && operation in setOf(Permission.WRITE, Permission.DELETE) ->
                    GovernanceDecision(
                        allowed = false,
                        reason = "Agent cannot directly modify statutory memory",
                        suggestion = "Use contradiction resolution flow to propose change to user"
                    )
                else -> GovernanceDecision(allowed = false, reason = "Unknown operation")
            }
        }
        
        // 运营层：Agent 完全自主
        return GovernanceDecision(allowed = true)
    }

    /**
     * 在记忆写入前执行治理检查
     * 
     * 返回：是否允许写入 + 建议的治理层级
     */
    fun preWriteCheck(
        content: String,
        initiator: ChangeInitiator,
        metadata: MemoryMetadata? = null
    ): WriteDecision {
        val layer = classifyLayer(content, metadata)
        val permission = checkPermission(layer, Permission.WRITE, initiator)
        
        return WriteDecision(
            allowed = permission.allowed,
            assignedLayer = layer,
            reason = permission.reason,
            requiresUserConfirmation = (layer == GovernanceLayer.STATUTORY && initiator == ChangeInitiator.AGENT)
        )
    }

    /**
     * 用户行使遗忘权 (Right to be Forgotten)
     * Constitutional rule: 用户可要求删除任何关于自己的记忆
     */
    fun exerciseRightToForget(userId: String, scope: ForgetScope): ForgetResult {
        log("用户 $userId 行使遗忘权 | scope=$scope")
        
        return when (scope) {
            ForgetScope.ALL -> {
                // 删除所有用户相关记忆（保留 Agent 核心人格）
                val count = store.deleteByUser(userId)
                ForgetResult(success = true, deletedCount = count, message = "已删除所有与您相关的记忆")
            }
            ForgetScope.RECENT_SESSION -> {
                // 只删最近一次会话的记忆
                val count = store.deleteRecentSession(userId)
                ForgetResult(success = true, deletedCount = count, message = "已删除最近一次对话产生的记忆")
            }
            ForgetScope.SPECIFIC_TOPIC -> {
                // 需要具体指定（这里简化处理）
                ForgetResult(success = false, deletedCount = 0, message = "请指定要遗忘的具体内容")
            }
        }
    }

    // === Private Classification Logic ===
    
    private fun isConstitutionalContent(content: String): Boolean {
        val constitutionalKeywords = listOf(
            "苏大姐的核心人格", "人格底线", "安全策略",
            "不准", "绝对禁止", "系统规则"
        )
        return constitutionalKeywords.any { content.contains(it) }
    }
    
    private fun isStatutoryContent(content: String, metadata: MemoryMetadata?): Boolean {
        // 长期事实：名字、职业、家人、关系等
        val factualPatterns = listOf(
            Regex("叫|名字是|姓"),
            Regex("在.{1,10}(工作|上班|上学|读书)"),
            Regex("(男|女)朋友|老公|老婆|丈夫|妻子|家人"),
            Regex("喜欢|讨厌|最爱|恐惧|过敏"),
            Regex("住在|家在|来自"),
            Regex("(养|有).{0,5}(猫|狗|宠物)"),
        )
        
        if (factualPatterns.any { it.containsMatchIn(content) }) {
            return true
        }
        
        // 如果 metadata 标记为高重要度，也算 Statutory
        if (metadata != null && metadata.importance > 0.8) {
            return true
        }
        
        return false
    }
    
    fun getStats(): String {
        return "治理层 | Constitutional: protected | Statutory: user-gated | Operational: agent-autonomous"
    }
}

/**
 * 宪法层规则（不可运行时修改）
 */
class ConstitutionalRules {
    // Rule 1: Agent 不能自我否定人格
    val personaImmutability = true
    
    // Rule 2: 用户拥有遗忘权
    val rightToForget = true
    
    // Rule 3: 安全过滤不可被绕过
    val safetyFilterMandatory = true
    
    // Rule 4: 记忆溯源不可被删除（审计需要）
    val provenanceImmutability = true
}

// === Data Models ===

enum class GovernanceLayer {
    CONSTITUTIONAL,  // 宪法层：不可修改
    STATUTORY,       // 法规层：用户授权才可修改
    OPERATIONAL      // 运营层：Agent 自主
}

enum class Permission {
    READ,
    WRITE,
    DELETE,
    PROPOSE_CHANGE
}

enum class ChangeInitiator {
    USER,       // 用户发起
    AGENT,      // Agent 自主发起
    SYSTEM      // 系统级操作（如衰减）
}

enum class ForgetScope {
    ALL,             // 删除所有
    RECENT_SESSION,  // 最近一次会话
    SPECIFIC_TOPIC   // 特定话题
}

data class GovernanceDecision(
    val allowed: Boolean,
    val reason: String = "",
    val suggestion: String = ""
)

data class WriteDecision(
    val allowed: Boolean,
    val assignedLayer: GovernanceLayer,
    val reason: String = "",
    val requiresUserConfirmation: Boolean = false
)

data class ForgetResult(
    val success: Boolean,
    val deletedCount: Int,
    val message: String
)

data class MemoryMetadata(
    val importance: Double = 0.5,
    val source: String = "",
    val timestamp: Long = System.currentTimeMillis()
)
