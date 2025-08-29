# 感知基础

## 第 1–2 周：数字电路基础 + Verilog 入门
- [ ] **Day 1–2**：MIT 6.004 组合逻辑视频 + HDLBits 基础练习（加法器、多路选择器、编码器/解码器）
- [ ] **Day 3–4**：时序逻辑（寄存器、触发器、计数器） + HDLBits 同步电路题
- [ ] **Day 5–6**：有限状态机（FSM）设计：交通灯 / 序列检测器 + Verilog 实现
- [ ] **Day 7**：写一份 Moore FSM SystemVerilog 实现 + Testbench

---

## 第 3–4 周：存储器（BRAM/FIFO）与 AXI 入门
- [ ] **Day 1–2**：Xilinx Vivado BRAM 教程 → FPGA 上实现 BRAM 读写 demo
- [ ] **Day 3–4**：FIFO generator 教程 → 做时钟域跨越实验
- [ ] **Day 5–6**：阅读 AXI4-Lite 简介（ARM 官方文档），画 AXI 通道握手时序图
- [ ] **Day 7**：实现 AXI4-Lite Slave，寄存器映射 + PS 可读写

---

## 第 5–6 周：流水线 & 并行处理
- [ ] **Day 1–3**：CS61C datapath/pipeline lecture + 单周期 CPU 实现（简化 RISC-V 指令集）
- [ ] **Day 4–5**：扩展成 5 级流水线 CPU，处理数据冒险（forwarding/stall）
- [ ] **Day 6**：添加分支预测或简单 cache 模块
- [ ] **Day 7**：写实验报告，总结流水线优化点

---

## 第 7–8 周：CNN 已掌握 → 跳过

---

## 第 9–10 周：CNN 深入 + 工业级 AXI 实战
- [ ] **Day 1–2**：学习 AlexNet / VGG 结构，画卷积/池化/全连接层示意图
- [ ] **Day 3–4**：PyTorch 实现小型 AlexNet（CIFAR-10）
- [ ] **Day 5**：学习 AXI-Stream 协议，理解 TREADY/TVALID 握手
- [ ] **Day 6**：写 AXI-Stream FIFO demo → 数据搬运 + 流水线实验
- [ ] **Day 7**：总结 CNN 硬件映射与 AXI 交互

---

## 第 11–12 周：SystemVerilog 工程化 + 综合复盘
- [ ] **Day 1–2**：SystemVerilog interface、modport、enum 可综合语法
- [ ] **Day 3–4**：写 AXI Master Testbench，用随机延迟模拟外设响应
- [ ] **Day 5**：学习 SVA（SystemVerilog Assertions），写握手信号断言
- [ ] **Day 6**：综合大项目：
  - CNN 第一层卷积硬件模块（SystemVerilog 实现）
  - AXI-Stream 接口
  - Testbench 仿真对比结果
- [ ] **Day 7**：写完整总结报告（CPU 流水线 + CNN 实现 + AXI 交互）
```