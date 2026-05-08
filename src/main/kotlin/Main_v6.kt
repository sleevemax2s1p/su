package com.memory

import com.memory.chat.ChatEngineV6
import com.memory.llm.DeepSeekClient
import com.memory.store.KnowledgeStore
import com.memory.gate.KnowledgeGate
import com.memory.collision.KnowledgeCollision
import com.memory.importance.ImportanceManager
import com.memory.reader.KnowledgeReader
import kotlinx.coroutines.runBlocking

/**
 * 苏大姐 v6.0 入口
 * 
 * 架构概览：
 * ┌─────────────────────────────────────────────────────────────┐
 * │                     ChatEngine v6.0                         │
 * │                                                             │
 * │  ┌──────────┐   ┌──────────────┐   ┌───────────────────┐  │
 * │  │ Safety   │   │ Procedural   │   │    SleepTime      │  │
 * │  │ Filter   │   │ Memory Mgr   │   │    Agent          │  │
 * │  │          │   │              │   │                   │  │
 * │  │ Write ✓  │   │ Pattern →    │   │ LinkEvolution     │  │
 * │  │ Read  ✓  │   │ Context Inj  │   │ Consolidation     │  │
 * │  │ Output✓  │   │ Strategy     │   │ DecayCheck        │  │
 * │  └──────────┘   └──────────────┘   └───────────────────┘  │
 * │         │               │                    ↑              │
 * │         ▼               ▼                    │              │
 * │  ┌──────────────────────────────────────┐    │              │
 * │  │         Core Chat Loop               │    │              │
 * │  │  Input → Retrieve → Prompt → LLM     │────┘              │
 * │  │  → Reply → Extract → Feedback        │                   │
 * │  └──────────────────────────────────────┘                   │
 * │         │                                                    │
 * │         ▼                                                    │
 * │  ┌──────────────────────────────────────┐                   │
 * │  │    KnowledgeStore (with Gate/Collision/Importance)  │    │
 * │  └──────────────────────────────────────┘                   │
 * └─────────────────────────────────────────────────────────────┘
 * 
 * 数据流：
 * 1. 用户输入 → SafetyFilter.validateForWrite() → 预检
 * 2. 记忆检索 → SafetyFilter.filterForInjection() → 安全记忆
 * 3. ProceduralMemory.generateContextInjection() → 经验注入
 * 4. LLM 生成回复
 * 5. 异步提取 → SafetyFilter.validateForWrite() → 安全写入
 * 6. FeedbackTracker → ProceduralMemory.learnFromInteraction()
 * 7. 条件触发 → SleepTimeAgent.submitConversation()
 */
fun main() = runBlocking {
    println("""
    ╔═══════════════════════════════════════╗
    ║       苏大姐 Memory System v6.0      ║
    ║                                       ║
    ║  New in v6:                           ║
    ║  • MemorySafetyFilter (防投毒)        ║
    ║  • SleepTimeAgent (异步巩固)          ║
    ║  • ProceduralMemory (经验学习)        ║
    ║  • FeedbackTracker (隐式反馈)         ║
    ╚═══════════════════════════════════════╝
    """.trimIndent())
    
    // 初始化依赖
    val llm = DeepSeekClient(apiKey = System.getenv("DEEPSEEK_API_KEY") ?: "")
    val store = KnowledgeStore()
    val gate = KnowledgeGate(llm, store)
    val collision = KnowledgeCollision(store)
    val importance = ImportanceManager(store)
    val reader = KnowledgeReader(store)
    
    // 创建 v6 引擎
    val engine = ChatEngineV6(
        llm = llm,
        store = store,
        gate = gate,
        collision = collision,
        importance = importance,
        reader = reader,
        tag = "苏大姐"
    )
    
    // 启动（初始化 Sleep-Time Agent 等后台组件）
    engine.initialize()
    
    println("\n苏大姐已上线。输入 /stats 查看状态，/quit 退出。\n")
    
    // CLI 对话循环
    while (true) {
        print("你: ")
        val input = readLine()?.trim() ?: break
        
        when {
            input == "/quit" || input == "/exit" -> {
                println("苏大姐: 行，走了啊。")
                break
            }
            input == "/stats" -> {
                println(engine.getStats())
                continue
            }
            input == "/reset" -> {
                engine.resetHistory()
                println("[系统] 对话历史已重置，Sleep-Time 已收到上一段对话。")
                continue
            }
            input.isBlank() -> continue
        }
        
        val response = engine.chat(input)
        println("苏大姐: ${response.reply}")
        
        // 调试信息
        if (response.safetyBlocked) {
            println("  [debug] ⚠️ 安全过滤已激活")
        }
        if (response.proceduralInjected) {
            println("  [debug] 💡 程序性记忆已注入")
        }
        if (response.hasContradiction) {
            println("  [debug] ⚡ 发现矛盾，下轮将反问")
        }
    }
    
    engine.dispose()
    println("Engine disposed. Bye.")
}
