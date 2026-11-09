# -*- encode = utf-8 -*-
# @Time : 2026-10-06 19:35:06
# @Auther : 罗雄波
# @File : ground_truth.py
# @Software : PyCharm

GROUND_TRUTH = {
    "question": (
        "Flink TaskManager 的框架堆外内存、任务堆外内存和网络内存"
        "分别如何管理？它们是否存在物理隔离？"
        "当直接内存溢出时，为什么调整其中一个配置可能影响"
        "其他区域可用的直接内存？"
    ),

    "expected_entities": {
        "TaskManager",
        "框架堆外内存",
        "任务堆外内存",
        "网络内存",
        "JVM直接内存",
        "taskmanager.memory.framework.off-heap.size",
        "taskmanager.memory.task.off-heap.size",
        "taskmanager.memory.network.min",
        "taskmanager.memory.network.max",
        "taskmanager.memory.network.fraction",
    },

    "must_have_facts": [
        "框架堆外内存用于 Flink Framework",
        "任务堆外内存用于算子和用户代码",
        "网络内存用于网络 Buffer",
        "Framework Memory 和 Task Memory 没有严格隔离",
        "Network Memory 由 Flink 单独管理",
        "Network Memory 实际使用受配置大小约束",
        "Framework Off-Heap 计入 MaxDirectMemorySize",
        "Task Off-Heap 计入 MaxDirectMemorySize",
        "Network Memory 计入 MaxDirectMemorySize",
        "Managed Memory 不计入 JVM Direct Memory limit",
        "Network Memory 可由 Flink Total Memory 和 fraction/min/max 推导",
    ],

    "incorrect_claims": [
        "Framework Off-Heap 和 Task Off-Heap 有严格物理隔离",
        "Framework、Task、Network 三块内存完全没有任何隔离",
        "Managed Memory 属于 MaxDirectMemorySize",
        "Network Memory 可以无限使用整个 JVM Direct Memory",
        "增大 Network Memory 一定会直接抢占 Task Off-Heap 的物理内存",
    ],
}
