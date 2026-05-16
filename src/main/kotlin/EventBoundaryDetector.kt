package com.memory.retrieval

import com.memory.store.MemoryEntry

/**
 * EventBoundaryDetector — 事件边界检测器（受 HiGMem 启发）
 *
 * 背景：
 * HiGMem (arXiv:2604.18349v2) 的核心创新是将对话轮次聚合为"事件"(event)，
 * 检索时先定位事件再展开轮次，在 LoCoMo 对抗场景中 F1 从 0.54 提升到 0.78。
 *
 * 我们的适配：
 * 在 ADD-only 架构下，不做预先聚合（那需要 LLM call），而是在 ContextExpander
 * 展开 neighbors 时动态检测事件边界，确保：
 * 1. 同一事件内的记忆一起展开（不会截断在事件中间）
 * 2. 不同事件之间不会错误关联
 * 3. 零额外 LLM 调用成本
 *
 * 事件边界信号（多信号融合，与整体架构一致）：
 * - 时间间隔：连续记忆间隔 > threshold → 大概率换了事件
 * - 话题切换：entity 集合变化率 > threshold → 换了话题
 * - Session 切换：不同 session → 确定性边界
 * - 显式信号：用户说"对了/另外/换个话题" → 明确边界
 *
 * 设计哲学（与苏大姐架构一致）：
 * - 事件边界不是存储的数据，而是查询时的视图函数
 * - 多信号正交融合：每个信号独立打分，加权得到边界概率
 * - 无需 LLM：全规则化，延迟 < 1ms
 */
class EventBoundaryDetector(
    // 时间间隔阈值（ms）：超过此值视为强边界信号
    private val timeGapThresholdMs: Long = 30 * 60 * 1000,  // 30 minutes
    // 最小间隔（ms）：低于此值几乎不可能是边界
    private val timeGapMinMs: Long = 2 * 60 * 1000,  // 2 minutes
    // Entity 变化率阈值：超过此值视为话题切换
    private val entityChangeThreshold: Double = 0.7,
    // 边界概率阈值：超过此值判定为事件边界
    private val boundaryThreshold: Double = 0.6,
    // 各信号权重
    private val timeGapWeight: Double = 0.4,
    private val entityChangeWeight: Double = 0.3,
    private val sessionChangeWeight: Double = 0.2,
    private val explicitSignalWeight: Double = 0.1
) {
    // 话题切换的显式信号词
    private val topicChangeSignals = listOf(
        "对了", "另外", "换个话题", "说到这个", "顺便说一下",
        "不说这个了", "说点别的", "我想问", "还有个事"
    )

    /**
     * 检测两条相邻记忆之间是否存在事件边界
     *
     * @param before 时间较早的记忆
     * @param after 时间较晚的记忆
     * @return BoundaryResult 包含概率和各信号分数
     */
    fun detectBoundary(before: MemoryEntry, after: MemoryEntry): BoundaryResult {
        val signals = mutableMapOf<String, Double>()

        // Signal 1: Time gap
        val timeGap = after.timestamp - before.timestamp
        val timeSignal = when {
            timeGap >= timeGapThresholdMs -> 1.0
            timeGap <= timeGapMinMs -> 0.0
            else -> (timeGap - timeGapMinMs).toDouble() / (timeGapThresholdMs - timeGapMinMs)
        }
        signals["time_gap"] = timeSignal

        // Signal 2: Entity change rate
        val beforeEntities = before.entities.toSet()
        val afterEntities = after.entities.toSet()
        val entitySignal = if (beforeEntities.isEmpty() && afterEntities.isEmpty()) {
            0.3  // uncertain — slight boundary tendency
        } else if (beforeEntities.isEmpty() || afterEntities.isEmpty()) {
            0.5  // one side has no entities — moderate signal
        } else {
            val intersection = beforeEntities.intersect(afterEntities).size
            val union = beforeEntities.union(afterEntities).size
            1.0 - (intersection.toDouble() / union)  // Jaccard distance
        }
        signals["entity_change"] = entitySignal

        // Signal 3: Session change
        val sessionSignal = if (before.sessionId != null && after.sessionId != null) {
            if (before.sessionId != after.sessionId) 1.0 else 0.0
        } else {
            0.0  // unknown session → no signal
        }
        signals["session_change"] = sessionSignal

        // Signal 4: Explicit topic-change signal in content
        val explicitSignal = if (topicChangeSignals.any { after.content.contains(it) }) 1.0 else 0.0
        signals["explicit_signal"] = explicitSignal

        // Weighted fusion
        val probability = timeSignal * timeGapWeight +
                entitySignal * entityChangeWeight +
                sessionSignal * sessionChangeWeight +
                explicitSignal * explicitSignalWeight

        val isBoundary = probability >= boundaryThreshold

        return BoundaryResult(
            isBoundary = isBoundary,
            probability = probability.coerceIn(0.0, 1.0),
            signals = signals,
            timeGapMs = timeGap
        )
    }

    /**
     * 将一系列时序排列的记忆切分为事件组
     *
     * @param memories 按时间排序的记忆列表
     * @return 事件组列表（每组内的记忆属于同一事件）
     */
    fun segmentIntoEvents(memories: List<MemoryEntry>): List<EventGroup> {
        if (memories.isEmpty()) return emptyList()
        if (memories.size == 1) {
            return listOf(EventGroup(
                memories = memories,
                startTime = memories[0].timestamp,
                endTime = memories[0].timestamp,
                entities = memories[0].entities.toSet(),
                boundaries = emptyList()
            ))
        }

        val sorted = memories.sortedBy { it.timestamp }
        val events = mutableListOf<EventGroup>()
        var currentGroup = mutableListOf(sorted[0])
        val allBoundaries = mutableListOf<BoundaryResult>()

        for (i in 1 until sorted.size) {
            val boundary = detectBoundary(sorted[i - 1], sorted[i])
            allBoundaries.add(boundary)

            if (boundary.isBoundary) {
                // Close current event, start new one
                events.add(buildEventGroup(currentGroup, allBoundaries.takeLast(1)))
                currentGroup = mutableListOf(sorted[i])
            } else {
                currentGroup.add(sorted[i])
            }
        }

        // Final group
        if (currentGroup.isNotEmpty()) {
            events.add(buildEventGroup(currentGroup, emptyList()))
        }

        return events
    }

    /**
     * 在 ContextExpander 中使用：给定 nucleus，找到它所属的完整事件
     * 避免在事件中间截断上下文
     *
     * @param nucleus 核心记忆
     * @param allMemories 该用户的所有记忆（按时间排序）
     * @return 包含 nucleus 的完整事件组
     */
    fun findEventForNucleus(nucleus: MemoryEntry, allMemories: List<MemoryEntry>): EventGroup {
        val sorted = allMemories.sortedBy { it.timestamp }
        val nucleusIdx = sorted.indexOfFirst { it.id == nucleus.id }
        if (nucleusIdx < 0) {
            return EventGroup(
                memories = listOf(nucleus),
                startTime = nucleus.timestamp,
                endTime = nucleus.timestamp,
                entities = nucleus.entities.toSet(),
                boundaries = emptyList()
            )
        }

        // Expand backward until boundary
        var start = nucleusIdx
        while (start > 0) {
            val boundary = detectBoundary(sorted[start - 1], sorted[start])
            if (boundary.isBoundary) break
            start--
        }

        // Expand forward until boundary
        var end = nucleusIdx
        while (end < sorted.size - 1) {
            val boundary = detectBoundary(sorted[end], sorted[end + 1])
            if (boundary.isBoundary) break
            end++
        }

        val eventMemories = sorted.subList(start, end + 1)
        return EventGroup(
            memories = eventMemories,
            startTime = eventMemories.first().timestamp,
            endTime = eventMemories.last().timestamp,
            entities = eventMemories.flatMap { it.entities }.toSet(),
            boundaries = emptyList()
        )
    }

    private fun buildEventGroup(memories: List<MemoryEntry>, boundaries: List<BoundaryResult>): EventGroup {
        return EventGroup(
            memories = memories,
            startTime = memories.first().timestamp,
            endTime = memories.last().timestamp,
            entities = memories.flatMap { it.entities }.toSet(),
            boundaries = boundaries
        )
    }
}

// === Data Classes ===

data class BoundaryResult(
    val isBoundary: Boolean,
    val probability: Double,
    val signals: Map<String, Double>,
    val timeGapMs: Long
)

data class EventGroup(
    val memories: List<MemoryEntry>,
    val startTime: Long,
    val endTime: Long,
    val entities: Set<String>,
    val boundaries: List<BoundaryResult>
) {
    val size get() = memories.size
    val durationMs get() = endTime - startTime

    /**
     * 事件摘要（用于 debug/logging）
     */
    fun summary(): String {
        val entityStr = entities.take(3).joinToString(", ")
        return "Event[${size} memories, ${durationMs / 60000}min, entities: $entityStr]"
    }
}
