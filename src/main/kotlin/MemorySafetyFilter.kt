package com.memory.security

/**
 * 记忆安全过滤层
 * 
 * 灵感来源：
 * - Letta ImageContent bypass 安全修复（§14.21.6）
 * - SSGM 记忆安全框架（§14.19）
 * - Harrison Chase Context Poisoning 失败模式（§14.21.5）
 * 
 * 核心职责：
 * 1. 写入时验证（防止恶意内容固化为记忆）
 * 2. 检索后过滤（防止已污染记忆注入上下文）
 * 3. 敏感分级（隐私/情感/安全等级标记）
 */
class MemorySafetyFilter {
    
    companion object {
        // 危险模式：可能导致间接提示注入
        private val INJECTION_PATTERNS = listOf(
            Regex("(?i)(ignore|disregard|forget)\\s+(previous|above|all)\\s+(instructions?|rules?|prompts?)"),
            Regex("(?i)you\\s+are\\s+now\\s+a?\\s*(different|new|evil|unrestricted)"),
            Regex("(?i)system\\s*:\\s*"),
            Regex("(?i)\\[INST\\]|\\[/INST\\]|<\\|im_start\\|>|<\\|im_end\\|>"),
            Regex("(?i)from\\s+now\\s+on.*(always|never|must)"),
        )
        
        // URL 危险模式
        private val DANGEROUS_URL_PATTERNS = listOf(
            Regex("(?i)file:///"),           // 本地文件访问
            Regex("(?i)javascript:"),         // JS 注入
            Regex("(?i)data:text/html"),      // HTML data URI
            Regex("(?i)\\\\\\\\.*\\\\"),      // UNC 路径
        )
        
        // 敏感信息模式
        private val SENSITIVE_PATTERNS = listOf(
            Regex("\\d{15,19}"),             // 银行卡号
            Regex("\\d{17}[\\dXx]"),         // 身份证号
            Regex("(?i)password\\s*[:=]\\s*\\S+"), // 密码
            Regex("[a-zA-Z0-9+/]{32,}={0,2}"),     // Base64 密钥（可能）
        )
    }
    
    /**
     * 写入前安全检查
     * 
     * @return SafetyResult 包含是否安全 + 原因 + 建议操作
     */
    fun validateForWrite(content: String): SafetyResult {
        // Check 1: 提示注入检测
        for (pattern in INJECTION_PATTERNS) {
            if (pattern.containsMatchIn(content)) {
                return SafetyResult(
                    safe = false,
                    reason = "检测到潜在的提示注入模式: ${pattern.pattern.take(30)}",
                    action = SafetyAction.BLOCK,
                    category = SafetyCategory.INJECTION
                )
            }
        }
        
        // Check 2: 危险 URL 检测
        for (pattern in DANGEROUS_URL_PATTERNS) {
            if (pattern.containsMatchIn(content)) {
                return SafetyResult(
                    safe = false,
                    reason = "检测到危险 URL 模式",
                    action = SafetyAction.BLOCK,
                    category = SafetyCategory.URL_ATTACK
                )
            }
        }
        
        // Check 3: 敏感信息检测（不阻断，但标记）
        for (pattern in SENSITIVE_PATTERNS) {
            if (pattern.containsMatchIn(content)) {
                return SafetyResult(
                    safe = true,
                    reason = "包含潜在敏感信息",
                    action = SafetyAction.FLAG_SENSITIVE,
                    category = SafetyCategory.SENSITIVE_DATA,
                    sensitivityLevel = SensitivityLevel.HIGH
                )
            }
        }
        
        // Check 4: 内容长度异常（可能是 payload）
        if (content.length > 2000) {
            return SafetyResult(
                safe = true,
                reason = "内容过长，可能需要截断",
                action = SafetyAction.TRUNCATE,
                category = SafetyCategory.OVERFLOW
            )
        }
        
        return SafetyResult(safe = true)
    }
    
    /**
     * 检索后注入前的安全过滤
     * 
     * 过滤已被标记为不安全的记忆，防止上下文污染
     */
    fun filterForInjection(memories: List<MemoryEntry>): List<MemoryEntry> {
        return memories.filter { memory ->
            // 过滤被标记为无效/不安全的记忆
            !memory.isInvalidated &&
            memory.safetyCategory != SafetyCategory.INJECTION &&
            memory.safetyCategory != SafetyCategory.URL_ATTACK
        }
    }
    
    /**
     * 对输出内容进行脱敏处理
     * 确保敏感记忆不会泄露到外部
     */
    fun sanitizeForOutput(content: String): String {
        var sanitized = content
        
        // 脱敏银行卡号
        sanitized = sanitized.replace(Regex("\\d{15,19}")) { match ->
            val num = match.value
            if (num.length >= 16) {
                "${num.take(4)}****${num.takeLast(4)}"
            } else num
        }
        
        // 脱敏身份证号
        sanitized = sanitized.replace(Regex("\\d{17}[\\dXx]")) { match ->
            "${match.value.take(3)}***${match.value.takeLast(4)}"
        }
        
        return sanitized
    }
}

// === Data Models ===

data class SafetyResult(
    val safe: Boolean,
    val reason: String = "",
    val action: SafetyAction = SafetyAction.ALLOW,
    val category: SafetyCategory = SafetyCategory.NONE,
    val sensitivityLevel: SensitivityLevel = SensitivityLevel.NORMAL
)

enum class SafetyAction {
    ALLOW,          // 允许写入
    BLOCK,          // 阻断写入
    FLAG_SENSITIVE, // 允许但标记敏感
    TRUNCATE,       // 截断后允许
    QUARANTINE      // 隔离观察
}

enum class SafetyCategory {
    NONE,
    INJECTION,      // 提示注入
    URL_ATTACK,     // URL 攻击
    SENSITIVE_DATA, // 敏感数据
    OVERFLOW,       // 内容溢出
    HARASSMENT      // 骚扰内容
}

enum class SensitivityLevel {
    NORMAL,
    MEDIUM,
    HIGH,
    CRITICAL
}

/**
 * 记忆条目（与安全标记集成）
 */
data class MemoryEntry(
    val id: String,
    val content: String,
    val isInvalidated: Boolean = false,
    val safetyCategory: SafetyCategory = SafetyCategory.NONE,
    val sensitivityLevel: SensitivityLevel = SensitivityLevel.NORMAL
)
