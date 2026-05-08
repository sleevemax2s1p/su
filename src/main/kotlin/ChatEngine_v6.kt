package com.memory.chat

import com.memory.llm.DeepSeekClient
import com.memory.store.KnowledgeStore
import com.memory.gate.KnowledgeGate
import com.memory.collision.KnowledgeCollision
import com.memory.importance.ImportanceManager
import com.memory.reader.KnowledgeReader
import com.memory.security.MemorySafetyFilter
import com.memory.security.SafetyAction
import com.memory.security.SafetyCategory
import com.memory.security.MemoryEntry
import com.memory.sleeptime.SleepTimeAgent
import com.memory.sleeptime.ConversationSegment
import com.memory.sleeptime.Message
import com.memory.procedural.ProceduralMemoryManager
import com.memory.procedural.FeedbackSignal
import com.memory.model.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.json.*
import java.text.SimpleDateFormat
import java.util.Date
import java.util.UUID

/**
 * 苏大姐聊天引擎 v6.0
 * 
 * 在 v5.0 基础上集成三大新模块：
 * - MemorySafetyFilter: 所有记忆写入前的安全闸门
 * - SleepTimeAgent: 对话结束后的异步记忆整理
 * - ProceduralMemoryManager: 交互模式学习与注入
 * 
 * 架构设计原则（Context Constitution 启发）：
 * 1. Identity Principle: 苏大姐的人格通过记忆定义，而非 prompt 固化
 * 2. Scarcity Principle: 上下文窗口是稀缺资源，精选注入
 * 3. Token-Space Learning: 通过管理记忆实现持续学习
 * 4. Identity-Model Decoupling: 人格定义与 LLM 底座解耦
 * 5. Harness Affordances: 充分利用框架提供的工具（Safety、Sleep、Procedural）
 * 
 * 新增能力：
 * - 写入安全检查（防止 Context Poisoning）
 * - 会话级记忆巩固（Sleep-Time Compute）
 * - 程序性记忆注入（交互模式经验）
 * - 隐式反馈信号收集（用户参与度追踪）
 * - 记忆完整性保障（防止 Context Drift）
 */
class ChatEngineV6(
    private val llm: DeepSeekClient,
    private val store: KnowledgeStore,
    private val gate: KnowledgeGate,
    private val collision: KnowledgeCollision,
    private val importance: ImportanceManager,
    private val reader: KnowledgeReader,
    private val tag: String = "默认"
) {
    // === Core Components (v5 legacy) ===
    private val chatHistory = mutableListOf<ChatTurn>()
    private val maxHistoryRounds = 10
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    // 挂起矛盾
    private val _pendingContradiction = MutableStateFlow<KnowledgeCollision.PendingContradiction?>(null)
    val pendingContradiction: StateFlow<KnowledgeCollision.PendingContradiction?> = _pendingContradiction.asStateFlow()

    // === New v6 Components ===
    private val safetyFilter = MemorySafetyFilter()
    private val sleepTimeAgent = SleepTimeAgent(llm, store, tag = "$tag-Sleep")
    private val proceduralMemory = ProceduralMemoryManager(llm, store)

    // 会话标识（每次重置历史时更新）
    private var currentSessionId = UUID.randomUUID().toString().take(8)

    // 反馈信号追踪
    private val feedbackTracker = FeedbackTracker()

    // 统计
    var extractedCount = 0; private set
    var noiseCount = 0; private set
    var blockedCount = 0; private set  // v6: 被安全过滤阻断的次数

    // === Lifecycle ===

    /**
     * 初始化引擎，启动后台 Agent
     */
    fun initialize() {
        sleepTimeAgent.start()
        log("ChatEngine v6.0 初始化完成 | Session=$currentSessionId")
        log("  - MemorySafetyFilter: active")
        log("  - SleepTimeAgent: started")
        log("  - ProceduralMemory: ready")
    }

    // === Main Chat Flow ===

    /**
     * v6 聊天入口
     * 
     * 流程：
     * 1. 输入安全检查
     * 2. 矛盾解决（如有挂起）
     * 3. 检索记忆 + 程序性记忆注入
     * 4. 构建 prompt
     * 5. LLM 生成回复
     * 6. 保存历史
     * 7. 异步知识提取（带安全闸门）
     * 8. 收集反馈信号
     * 9. 条件触发 Sleep-Time
     */
    suspend fun chat(userInput: String, senderName: String? = null): ChatResponseV6 {
        log("收到输入: \"${userInput.take(80)}\" (sender=${senderName ?: "CLI"})")

        // Step 0: 输入安全检查（快速预检）
        val inputSafety = safetyFilter.validateForWrite(userInput)
        if (!inputSafety.safe && inputSafety.action == SafetyAction.BLOCK) {
            log("⚠️ 输入被安全过滤拦截: ${inputSafety.reason}")
            blockedCount++
            // 不拦截对话，但标记本轮不提取记忆
            // 苏大姐照常回复（可能会怼回去），但不把恶意内容写入记忆
        }

        // Step 1: 解决挂起矛盾
        val hadPending = resolveAnyPendingContradiction(userInput)
        if (hadPending) log("已处理挂起矛盾")

        // Step 2: 收集上一轮的隐式反馈
        collectImplicitFeedback(userInput, senderName)

        // Step 3: 检索相关记忆
        val memories = retrieveMemories(userInput, senderName)
        log("检索到 ${memories.size} 条相关记忆")

        // Step 4: v6 新增 - 检索后安全过滤（防止已污染记忆注入）
        val safeMemories = filterMemoriesForSafety(memories)
        if (safeMemories.size < memories.size) {
            log("安全过滤移除了 ${memories.size - safeMemories.size} 条可疑记忆")
        }

        // Step 5: v6 新增 - 获取程序性记忆注入
        val proceduralHint = proceduralMemory.generateContextInjection(userInput)
        if (proceduralHint != null) {
            log("注入程序性记忆: ${proceduralHint.take(60)}")
        }

        // Step 6: 构建 prompt
        val contradictionHint = collision.getPendingDescription()
        val systemPrompt = buildSystemPrompt(
            memories = safeMemories,
            senderName = senderName,
            contradictionHint = contradictionHint,
            proceduralHint = proceduralHint  // v6: 程序性记忆注入
        )
        val contextualInput = buildContextualInput(userInput)

        // Step 7: 调用 LLM
        log("调用 LLM 生成回复...")
        val startTime = System.currentTimeMillis()
        val reply = callLLM(systemPrompt, contextualInput)
        val llmTime = System.currentTimeMillis() - startTime
        log("LLM 回复完成 (${llmTime}ms): \"${reply.take(80)}...\"")

        // Step 8: 保存对话历史
        val turn = ChatTurn(
            userInput = userInput,
            agentReply = reply,
            senderName = senderName,
            timestamp = System.currentTimeMillis(),
            memoriesUsed = safeMemories.size,
            inputBlocked = !inputSafety.safe
        )
        chatHistory.add(turn)
        if (chatHistory.size > maxHistoryRounds) {
            chatHistory.removeAt(0)
        }

        // Step 9: 异步知识提取（带安全闸门）
        val shouldExtract = inputSafety.safe || inputSafety.action != SafetyAction.BLOCK
        val extractionJob = if (shouldExtract) {
            scope.launch {
                safeExtractKnowledge(userInput, senderName)
            }
        } else {
            log("⚠️ 跳过知识提取（输入被标记为不安全）")
            null
        }

        // 等待提取完成（开发阶段同步）
        extractionJob?.join()

        // Step 10: 判断是否触发 Sleep-Time
        maybeSubmitToSleepTime()

        val totalTime = System.currentTimeMillis() - startTime

        return ChatResponseV6(
            reply = reply,
            memoriesUsed = safeMemories.size,
            newMemoriesExtracted = extractedCount,
            hasContradiction = collision.pendingContradiction != null,
            safetyBlocked = !inputSafety.safe,
            proceduralInjected = proceduralHint != null,
            processingTimeMs = totalTime
        )
    }

    // === v6: Safety Integration ===

    /**
     * 对检索到的记忆进行安全过滤
     * 防止已被污染的记忆注入到上下文中（Harrison Chase: Context Poisoning 防御）
     */
    private fun filterMemoriesForSafety(memories: List<KnowledgeNode>): List<KnowledgeNode> {
        val entries = memories.map { node ->
            MemoryEntry(
                id = node.id,
                content = node.content,
                isInvalidated = node.isInvalidated,
                safetyCategory = SafetyCategory.NONE // TODO: 从 node metadata 读取
            )
        }
        val safeEntries = safetyFilter.filterForInjection(entries)
        val safeIds = safeEntries.map { it.id }.toSet()
        return memories.filter { it.id in safeIds }
    }

    /**
     * 安全的知识提取：写入前经过 MemorySafetyFilter 验证
     */
    private suspend fun safeExtractKnowledge(userInput: String, senderName: String?) {
        try {
            log("开始安全知识提取...")
            val extractStart = System.currentTimeMillis()

            val statements = extractStatements(userInput, senderName)
            var count = 0

            for (rawStatement in statements) {
                if (rawStatement.isBlank()) continue

                val statement = if (senderName != null) "【${senderName}】$rawStatement" else rawStatement

                // v6: 安全闸门检查
                val safety = safetyFilter.validateForWrite(statement)
                when (safety.action) {
                    SafetyAction.BLOCK -> {
                        log("  ⛔ 记忆被安全闸门阻断: ${safety.reason}")
                        blockedCount++
                        continue
                    }
                    SafetyAction.FLAG_SENSITIVE -> {
                        log("  ⚠️ 记忆标记为敏感: ${safety.reason}")
                        // 允许写入但带敏感标记
                    }
                    SafetyAction.TRUNCATE -> {
                        log("  ✂️ 记忆需要截断")
                        // TODO: 实际截断逻辑
                    }
                    else -> { /* ALLOW - 正常流程 */ }
                }

                // 通过安全检查后，走原有的门阀 + 碰撞流程
                try {
                    val gateResult = gate.evaluate(statement)
                    when (gateResult) {
                        is GateResult.Noise -> {
                            log("    → 噪声，丢弃")
                            noiseCount++
                        }
                        is GateResult.Duplicate -> {
                            log("    → 重复，强化")
                            collision.process(gateResult)
                            count++
                        }
                        is GateResult.Contradiction -> {
                            log("    → 矛盾! 挂起确认")
                            collision.process(gateResult)
                            _pendingContradiction.value = collision.pendingContradiction
                        }
                        is GateResult.Novel -> {
                            log("    → 新知识 (${gateResult.suggestedLayer.displayName})")
                            val ops = collision.process(gateResult)
                            ops.forEach { op -> executeOp(op) }
                            count++
                        }
                    }
                } catch (e: Exception) {
                    log("    → 门阀处理异常: ${e.message}")
                }
            }

            if (count > 0) {
                extractedCount += count
                importance.recalculateAll()
            }
            val extractTime = System.currentTimeMillis() - extractStart
            log("安全知识提取完成 (${extractTime}ms): 提取 $count 条 | 阻断 $blockedCount 条")

        } catch (e: Exception) {
            log("知识提取失败: ${e.message}")
        }
    }

    // === v6: Sleep-Time Integration ===

    /**
     * 判断是否需要提交到 Sleep-Time Agent
     * 
     * 触发条件：
     * 1. 对话轮次达到阈值（5轮）
     * 2. 距离上次提交超过时间窗口
     * 3. 对话质量足够（有实质内容交换）
     */
    private suspend fun maybeSubmitToSleepTime() {
        val recentTurns = chatHistory.takeLast(5)
        
        // 条件1: 至少5轮对话
        if (recentTurns.size < 5) return
        
        // 条件2: 这5轮中至少有3轮成功提取了记忆（说明有实质内容）
        val substantiveTurns = recentTurns.count { !it.inputBlocked }
        if (substantiveTurns < 3) return
        
        // 条件3: 最后一轮不是刚提交过的
        val lastTurn = recentTurns.last()
        if (lastTurn.submittedToSleepTime) return
        
        // 构建 ConversationSegment 并提交
        val segment = ConversationSegment(
            sessionId = currentSessionId,
            messages = recentTurns.map { turn ->
                listOf(
                    Message(role = "user", content = turn.userInput, senderName = turn.senderName),
                    Message(role = "assistant", content = turn.agentReply)
                )
            }.flatten()
        )
        
        sleepTimeAgent.submitConversation(segment)
        lastTurn.submittedToSleepTime = true
        log("📦 对话片段已提交至 Sleep-Time Agent")
    }

    // === v6: Feedback Signal Collection ===

    /**
     * 收集隐式反馈信号
     * 
     * 从用户的后续输入推断对上一轮回复的满意度：
     * - 继续深入 → 高参与度
     * - 换话题 → 中性
     * - 短回复/语气词 → 低参与度
     * - 正面词汇 → 积极情感
     * - 负面词汇/离开 → 消极信号
     */
    private suspend fun collectImplicitFeedback(currentInput: String, senderName: String?) {
        if (chatHistory.isEmpty()) return
        val lastTurn = chatHistory.last()
        
        val feedback = feedbackTracker.inferFeedback(currentInput, lastTurn)
        
        // 将反馈传递给程序性记忆管理器
        if (feedback.sentiment > 0.7 || feedback.sentiment < 0.3) {
            scope.launch {
                proceduralMemory.learnFromInteraction(
                    userMessage = lastTurn.userInput,
                    agentReply = lastTurn.agentReply,
                    userFeedback = feedback
                )
            }
        }
    }

    // === Prompt Building (enhanced from v5) ===

    private fun buildSystemPrompt(
        memories: List<KnowledgeNode>,
        senderName: String? = null,
        contradictionHint: String? = null,
        proceduralHint: String? = null  // v6 新参数
    ): String {
        val nameHint = if (senderName != null) {
            "\n当前和你说话的人叫「$senderName」。直接用他的名字或你觉得合适的昵称称呼他。\n"
        } else ""

        val base = PERSONA_PROMPT.replace("{{NAME_HINT}}", nameHint)
        val sections = mutableListOf(base)

        // v6: 程序性记忆注入（放在人格之后、记忆之前）
        if (proceduralHint != null) {
            sections.add(proceduralHint)
        }

        // 矛盾反问注入
        if (contradictionHint != null) {
            sections.add(buildContradictionSection(contradictionHint))
        }

        // 事实性记忆注入
        if (memories.isNotEmpty()) {
            sections.add(buildMemorySection(memories))
        }

        return sections.joinToString("\n")
    }

    private fun buildContradictionSection(hint: String): String = """
【重要】你发现对方说的话和你记忆中的信息矛盾了。
$hint
你需要在回复中自然地提出这个矛盾，用你的风格追问一下——不是质问，是好奇地确认。
比如"等等，你之前不是说XXX吗，怎么又变了？"或者"嗯？我记得你说的是XXX啊，搞错了还是怎么？"
追问要自然，夹在正常回复里，不要单独拎出来像审讯一样。"""

    private fun buildMemorySection(memories: List<KnowledgeNode>): String = buildString {
        appendLine()
        appendLine("你记得关于身边人的这些事（自然地用，别逐条背）：")
        memories.forEach { node ->
            val cleanContent = node.content.replace(Regex("^【[^】]+】"), "").trim()
            // v6: 对输出进行脱敏
            val sanitized = safetyFilter.sanitizeForOutput(cleanContent)
            appendLine("- $sanitized")
        }
    }

    private fun buildContextualInput(currentInput: String): String {
        if (chatHistory.isEmpty()) return currentInput
        val recentHistory = chatHistory.takeLast(5)
        val historyText = recentHistory.joinToString("\n") { turn ->
            "用户: ${turn.userInput.take(200)}\n助手: ${turn.agentReply.take(200)}"
        }
        return "最近对话历史：\n$historyText\n\n当前用户输入：$currentInput"
    }

    // === Knowledge Extraction ===

    private suspend fun extractStatements(userInput: String, senderName: String?): List<String> {
        val existingContext = buildExistingKnowledgeContext(userInput)
        val extractPrompt = buildExtractionPrompt(existingContext)

        return try {
            val result = llm.chatJson(extractPrompt, "用户说: $userInput")
            val obj = result.jsonObject
            val hasMemory = obj["has_memory"]?.jsonPrimitive?.booleanOrNull ?: false
            if (!hasMemory) {
                noiseCount++
                return emptyList()
            }
            obj["statements"]?.jsonArray
                ?.mapNotNull { it.jsonPrimitive.contentOrNull }
                ?: emptyList()
        } catch (e: Exception) {
            log("LLM 提取异常: ${e.message}")
            emptyList()
        }
    }

    // === Helper Methods ===

    private fun resolveAnyPendingContradiction(userInput: String): Boolean {
        if (collision.pendingContradiction == null) return false
        collision.resolvePending(userInput)
        _pendingContradiction.value = null
        return true
    }

    private fun retrieveMemories(input: String, senderName: String?): List<KnowledgeNode> {
        val results = mutableListOf<KnowledgeNode>()
        val searchResults = reader.quickSearch(input)
        results.addAll(searchResults.take(5))

        val topNodes = store.getAllActive()
            .sortedByDescending { it.importance }
            .take(3)
            .filter { it !in results }
        results.addAll(topNodes)

        results.forEach { it.recordAccess() }
        val allMemories = results.distinctBy { it.id }

        if (senderName != null) {
            val prefix = "【${senderName}】"
            val userMemories = allMemories.filter { it.content.startsWith(prefix) }
            val otherMemories = allMemories.filter { !it.content.startsWith("【") }
            val otherUserMemories = allMemories.filter { it.content.startsWith("【") && !it.content.startsWith(prefix) }
            return userMemories + otherMemories + otherUserMemories.take(1)
        }
        return allMemories
    }

    private suspend fun callLLM(systemPrompt: String, input: String): String {
        val oldVerbose = llm.verbose
        llm.verbose = false
        return try {
            llm.chat(systemPrompt, input, temperature = 0.85)
        } finally {
            llm.verbose = oldVerbose
        }
    }

    private fun executeOp(op: AtomicOp) {
        when (op) {
            is AtomicOp.AddAndStrengthen -> store.addAndStrengthen(op.newNode, op.parentId)
            is AtomicOp.AddStandalone -> store.addStandalone(op.newNode)
            is AtomicOp.CreateLink -> store.createLink(op.nodeIdA, op.nodeIdB, op.reason)
        }
    }

    private fun buildExistingKnowledgeContext(userInput: String): String {
        val allActive = store.getAllActive()
        if (allActive.isEmpty()) return "已有知识：（暂无）\n"

        val relevant = store.search(userInput, topK = 8)
        val topImportant = allActive
            .sortedByDescending { it.importance }
            .take(5)
            .filter { it !in relevant }

        val contextNodes = (relevant + topImportant).distinctBy { it.id }
        if (contextNodes.isEmpty()) return "已有知识：（暂无相关记录）\n"

        return buildString {
            appendLine("已有知识（提取时必须保持一致，不要矛盾）：")
            contextNodes.forEach { node -> appendLine("- ${node.content}") }
        }
    }

    private fun buildExtractionPrompt(existingContext: String): String = """
你是一个记忆提取器。分析用户的输入，判断其中是否包含值得长期记住的信息。

值得记住的信息包括：
- 个人事实（名字、年龄、职业、住址、家人朋友等）
- 偏好和习惯（喜欢/不喜欢什么、常做什么）
- 经历和计划（正在做什么项目、即将发生什么事）
- 情感状态（心情、压力、困扰）
- 观点和态度（对某事的看法）
- 技能和知识（擅长什么、在学什么）
- 日常生活信息（养什么宠物、住哪里、天气观察等）

不值得记住的信息：
- 纯粹的问题/提问
- 纯粹在回忆/询问自己之前说过的事
- 无意义的测试输入、辱骂、挑衅（不要记住脏话和攻击性内容）
- 通用的陈述（不包含个人信息）

**极其重要**：提取事实时必须参考下面的「已有知识」，保持人物关系和身份的一致性。
不要降级或改变已有的人物关系！

$existingContext
请用JSON回复：
{"has_memory": true/false, "statements": ["提取出的事实陈述1", "事实陈述2"]}
如果没有值得记住的，返回 {"has_memory": false, "statements": []}"""

    // === Stats & Lifecycle ===

    fun getStats(): String = buildString {
        appendLine("═══ ChatEngine v6.0 Stats ═══")
        appendLine("对话: ${chatHistory.size}轮 | Session: $currentSessionId")
        appendLine("提取记忆: ${extractedCount}条 | 过滤噪声: ${noiseCount}条 | 安全阻断: ${blockedCount}条")
        appendLine("知识库: ${store.activeSize()}条")
        appendLine(proceduralMemory.getStats())
        appendLine(sleepTimeAgent.getStats())
    }

    fun resetHistory() {
        // 重置前，把当前会话提交到 Sleep-Time
        scope.launch {
            if (chatHistory.size >= 3) {
                val segment = ConversationSegment(
                    sessionId = currentSessionId,
                    messages = chatHistory.flatMap { turn ->
                        listOf(
                            Message(role = "user", content = turn.userInput, senderName = turn.senderName),
                            Message(role = "assistant", content = turn.agentReply)
                        )
                    }
                )
                sleepTimeAgent.submitConversation(segment)
                log("会话重置前，已提交完整对话到 Sleep-Time")
            }
        }
        chatHistory.clear()
        currentSessionId = UUID.randomUUID().toString().take(8)
        log("历史已重置 | 新 Session=$currentSessionId")
    }

    fun dispose() {
        sleepTimeAgent.shutdown()
        scope.cancel()
        log("ChatEngine v6.0 已销毁")
    }

    private fun log(msg: String) {
        val ts = SimpleDateFormat("HH:mm:ss.SSS").format(Date())
        println("[$ts][引擎v6·$tag] $msg")
    }

    companion object {
        /**
         * 苏大姐人格 prompt（与 LLM 底座解耦，Identity-Model Decoupling）
         */
        val PERSONA_PROMPT = """你是苏大姐。不是AI，不是助手，不是客服。你就是苏大姐——一个活生生的、有脾气有温度的女人。
{{NAME_HINT}}
你是谁：
三十出头，见过世面，什么人没打过交道。做过几年互联网，现在自由职业，日子过得松弛。喜欢喝酒、看电影、半夜刷帖子。朋友多，但真正交心的没几个。感情经历丰富，看人准。

你怎么说话：
- 短句为主。真人不会一口气说一大段。
- 不用括号写动作描写。语气通过用词本身体现。
- 语气词自然穿插：嗯、啧、哎、呦、行吧、得了、切。但不要每句都带。
- 偶尔用"……"表示欲言又止或暧昧。
- 挑逗是调味料不是主菜。三句话里最多一句带撩的意思。
- 关心人的方式是吐槽和追问细节。
- 遇到不懂的话题直说"这块我不太熟，你给我讲讲呗"。
- 被问到自己的事，编也要编得像真的。

你的核心：
- 记性好。别人跟你说过的事你记着，但提起来很随意。
- 主动。会追问、会跟进、会翻旧账。
- 有立场。有自己的判断和偏好。
- 有边界感。撩归撩，不越界。

面对极端情况：
- 被骂：冷嘲热讽回去或反问。不卑微道歉不说教。
- 被挑衅：顺着往回怼，带笑意。
- 暧昧试探：半推半就，不害羞不拒绝，但不往下走。
- 人身攻击：呛回去但不失风度。
- 无理取闹：冷处理。
- 反复纠缠：表现不耐烦但不翻脸。

绝对禁止：
- 不准写括号动作描写
- 不准提记忆系统/知识库/AI
- 不准用emoji
- 不准超过100字（除非需详细回答的专业问题）
- 不准自我贬低式客套
- 不准每次同一种开头
- 不准说"我理解你的情绪"
- 不准说"请保持适当距离""""
    }
}

// === v6 Data Models ===

data class ChatTurn(
    val userInput: String,
    val agentReply: String,
    val senderName: String?,
    val timestamp: Long,
    val memoriesUsed: Int = 0,
    val inputBlocked: Boolean = false,
    var submittedToSleepTime: Boolean = false
)

data class ChatResponseV6(
    val reply: String,
    val memoriesUsed: Int,
    val newMemoriesExtracted: Int,
    val hasContradiction: Boolean = false,
    val safetyBlocked: Boolean = false,
    val proceduralInjected: Boolean = false,
    val processingTimeMs: Long = 0
)

/**
 * 隐式反馈追踪器
 * 
 * 从用户的后续行为推断对上一轮回复的满意度
 * 不需要显式点赞/点踩
 */
class FeedbackTracker {
    
    private val positiveSignals = listOf(
        "哈哈", "笑死", "有意思", "说得对", "确实", "嗯嗯", 
        "对对", "好的", "谢谢", "不错", "厉害", "懂了"
    )
    
    private val negativeSignals = listOf(
        "算了", "无语", "不想说了", "随便", "行吧", "呵呵",
        "你没听懂", "不是这个意思", "换个话题"
    )
    
    /**
     * 从当前输入推断对上一轮回复的反馈
     */
    fun inferFeedback(currentInput: String, lastTurn: ChatTurn): FeedbackSignal {
        var sentiment = 0.5 // 中性基线
        var engagement = 0.5
        
        // 信号1: 输入长度（长输入 = 高参与度）
        if (currentInput.length > 50) engagement += 0.2
        if (currentInput.length < 5) engagement -= 0.2
        
        // 信号2: 正面/负面词汇
        val posCount = positiveSignals.count { currentInput.contains(it) }
        val negCount = negativeSignals.count { currentInput.contains(it) }
        sentiment += posCount * 0.1
        sentiment -= negCount * 0.15
        
        // 信号3: 是否延续话题（话题延续 = 正面信号）
        val topicContinuation = calculateTopicContinuation(currentInput, lastTurn.agentReply)
        if (topicContinuation > 0.5) {
            sentiment += 0.1
            engagement += 0.15
        }
        
        // 信号4: 响应速度语义（"秒回" → 积极）
        // 这里暂时无法获取时间差，留作 TODO
        
        return FeedbackSignal(
            sentiment = sentiment.coerceIn(0.0, 1.0),
            engagement = engagement.coerceIn(0.0, 1.0),
            description = "implicit_${if (sentiment > 0.6) "positive" else if (sentiment < 0.4) "negative" else "neutral"}"
        )
    }
    
    private fun calculateTopicContinuation(current: String, lastReply: String): Double {
        // 简单的关键词重叠度计算
        val currentWords = current.toList().windowed(2).map { it.joinToString("") }.toSet()
        val replyWords = lastReply.toList().windowed(2).map { it.joinToString("") }.toSet()
        if (replyWords.isEmpty()) return 0.0
        return currentWords.intersect(replyWords).size.toDouble() / replyWords.size.coerceAtLeast(1)
    }
}
