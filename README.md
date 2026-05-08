# 苏大姐 v6.1 — LoCoMo Eval (Kotlin)

## 快速运行

```bash
# 1. 确保有 JDK 11+
java -version

# 2. 直接运行（Gradle 会自动下载）
./gradlew run

# 或者手动编译运行
./gradlew build
java -jar build/libs/sudajie-eval-6.1.0.jar
```

## 不想用 Gradle？直接 kotlinc 编译

```bash
# 需要先安装 kotlin compiler
kotlinc src/main/kotlin/EvalV17Full.kt -include-runtime -d eval.jar
java -jar eval.jar
```

## 项目结构

```
sudajie-eval/
├── src/main/kotlin/
│   ├── EvalV17Full.kt          # ← 主评测程序（可独立编译运行）
│   ├── MemorySafetyFilter.kt   # 安全过滤模块
│   ├── MemoryGovernance.kt     # 治理层级模块
│   ├── MemoryProvenance.kt     # 溯源模块
│   ├── ProceduralMemoryManager.kt  # 程序性记忆
│   ├── SleepTimeAgent.kt       # 异步巩固
│   └── ChatEngine_v6.kt        # 统一引擎
├── locomo10.json               # LoCoMo 评测数据集
├── build.gradle.kts
└── README.md
```

## 评测内容

- 全量 10 conversations / 1986 questions
- 5 个 category: single-hop, temporal, multi-hop, open-domain, adversarial
- 集成 Governance Layer 检索增强 + Provenance Trust 标注 + Safety Filter

## API Keys

评测程序内置了 DeepSeek + 智谱 API Key，如需替换请修改 EvalV17Full.kt 顶部常量。

## 注意

- `EvalV17Full.kt` 是**独立可编译的**，不依赖其他 .kt 文件
- 其他 .kt 文件是苏大姐完整架构代码，需要补充接口实现才能编译
- 评测程序已将 Governance/Provenance/Safety 核心逻辑内联
