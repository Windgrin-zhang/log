### 1.5 天 YOLO 调参学习-实战计划（可直接执行）

- **适用对象**：已能跑通自己数据集的 YOLO（v5/v8 均可）
- **目标**：掌握高性价比的调参顺序与范围，能独立完成一次从 baseline 到收敛的调优闭环

### 准备（30 分钟）
- **数据与环境**
  - 固定划分与随机种子：`seed=0`，train/val 不变
  - 检查标签：类别是否错标/漏标；样本数是否严重失衡
  - 先选小模型跑通：`yolov5s` 或 `yolov8s`，imgsz 640
- **指标与记录**
  - 对齐评估：看 `mAP50-95` 为主，同时关注 `P/R`、混淆矩阵
  - 每次实验记录：配置、最佳权重路径、关键指标、曲线截图

### 快速命令模板（二选一）
- **YOLOv5（Ultralytics v5 仓库）**
```bash
# 基线
python train.py --data data.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 100 --project exp_baseline --name v5s_baseline --hyp data/hyps/hyp.scratch-low.yaml

# 演化(自动 HPO，给 30-100 代看时间)
python train.py --data data.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 80 --evolve 60 --project exp_evolve --name v5s_evolve
```
- **YOLOv8（Ultralytics CLI）**
```bash
# 基线
yolo detect train data=data.yaml model=yolov8s.pt imgsz=640 batch=16 epochs=100 project=exp_baseline name=v8s_baseline

# 快速超参搜索（官方 tune；如不可用则手动多跑几组）
yolo detect tune data=data.yaml model=yolov8s.pt imgsz=640 batch=16 epochs=60 time=2h project=exp_tune name=v8s_tune
```

### Day 1（约 8-10 小时）

- **0) 建立基线（1 小时）**
  - 训练一次标准配置，确认可稳定复现
  - 产出：baseline 指标、曲线（loss/mAP）、推理样例

- **1) 数据与图像尺寸优先（2 小时）**
  - imgsz 快速对比：512/640/800（各 20-30 epoch 小跑）
  - 小目标多：偏大 imgsz；目标大：640 足够；若显存紧张，用梯度累积替代大 batch
  - 产出：选定 imgsz

- **2) 增强策略微调（2-3 小时，A/B 测试，每组 20-30 epoch）**
  - 优先顺序与范围：
    - mosaic: 0.3-1.0（小数据集易过拟合→开大一点；细粒度定位差→适度减小）
    - mixup: 0.0-0.2（类别混淆多时减小）
    - hsv_h/s/v: 0.01/0.5/0.3-0.5（光照/风格多样时适度增大）
    - degrees: 0-10，translate: 0-0.2，scale: 0.5，shear: 0-2，fliplr: 0.5，flipud: 0-0.1
  - 产出：1 套最稳的增强组合

- **3) 优化器与学习率（2 小时）**
  - batch 设为显存可承受的最大值（或开启梯度累积）
  - 学习率与调度（优先影响大）：
    - SGD：`lr0 0.005-0.02`，`lrf 0.05-0.2`
    - AdamW（小数据/迁移微调）：`lr0 0.0005-0.003`，`weight_decay 0.01-0.05`
    - 预热：`warmup_epochs 2-5`
  - 产出：选定 optimizer+lr 组合与调度

- **4) 模型规模与训练时长（1-2 小时）**
  - 明显欠拟合（低 train/val 指标、loss 不下降）：从 `n/s -> m`，或延长 `epochs`
  - 明显过拟合（train 好 val 差）：增强加大或 `s/n` 模型、增大正则（weight_decay）
  - 产出：最终模型大小与 epoch 上限、EarlyStopping `patience ~ 20`

### Day 2 上午（约 4-6 小时）

- **5) 自动化超参搜索（2-3 小时）**
  - v5 用 `--evolve`；v8 用 `yolo ... tune`（若环境不支持则手动网格/随机搜索）
  - 搜索空间精简到“有效的那几个”：`lr0, lrf, momentum/weight_decay, mosaic, mixup, scale`
  - 时间到即止，挑前 3 名配置复训 80-150 epoch 做最终对比

- **6) 误差分析与收尾（1-2 小时）**
  - 看 PR 曲线、混淆矩阵、badcases（漏检/误检）
  - 针对问题微调：
    - 小目标漏检多：加大 imgsz、提高训练分辨率、适度减小 mosaic、检查标注框是否过紧
    - 类别混淆：减小 mixup、适度增大 `cls` 损失（v5），筛查脏标注
    - 召回低：检查增强是否过强、学习率是否过小
  - 用最佳配置全量复训并导出最终模型

### 推荐的调参优先级与安全范围
- **1) imgsz 与 batch**：imgsz 512-800；batch 以显存为限
- **2) 学习率与调度**：`lr0 0.005-0.02(SGD)` / `0.0005-0.003(AdamW)`；`lrf 0.05-0.2`
- **3) 增强强度**：mosaic 0.3-1.0，mixup 0.0-0.2，degrees 0-10，translate 0-0.2，scale ~0.5，fliplr 0.5
- **4) weight_decay**：0.0005(v5 常见)；AdamW 时 0.01-0.05
- **5) 特定版本项**
  - v5：`--evolve` 很高效；anchors 默认会自动校准（一般不用手调）
  - v8：损失权重通常不动；用官方 `tune`/`val`/`predict` 完成闭环

### 小抄：问题-对应动作
- **欠拟合（loss 高、mAP 低）**：更大模型/更久训练/稍增 lr、减弱增强
- **过拟合（train 好 val 差）**：增强加大、增 weight_decay、用更小模型
- **召回低（R 低）**：适度提高 imgsz、减弱过强增强、检查标签漏标
- **精度低（P 低）**：减 mixup、调低推理 conf，适当调 NMS iou（0.5-0.7）
- **小目标差**：imgsz↑、合理裁剪/缩放、检查标注是否过度紧贴

### 两个高性价比打法
- **快速 A/B**：每次只改 1 类参数，20-30 epoch 小跑对比，赢者再长训
- **自动演化**：让工具探索，再精训前三名

### 资料（精读不超过 1 小时）
- Ultralytics 文档（训练/调参/增强）：[Ultralytics Docs](https://docs.ultralytics.com)
- YOLOv5 超参说明与 evolve 用法：[YOLOv5 Docs](https://docs.ultralytics.com/yolov5)
- 数据增强与可视化建议：[Albumentations Guide](https://albumentations.ai/docs)

如果告诉我你用的是 v5 还是 v8、GPU 显存大小、数据集规模与类别分布，我可以给你一版“带数值的”最小搜索空间和可直接复制的完整命令清单。

- **产出清单**
  - 1 个稳定基线（含配置与日志）
  - 1 组最优增强与 lr/scheduler
  - 1 次自动或手动 HPO 的前三名与复训结果
  - 最终模型与推理阈值建议（conf/iou）

- **结果达标判据**
  - 相比基线 mAP50-95 提升 ≥ 2-5 个点（或相同精度下速度↑/模型更小）
  - 训练/验证曲线稳定，无明显过拟合，误检漏检显著减少

- **快捷参数范围**
  - imgsz: 512-800
  - lr0: 0.005-0.02(SGD)/0.0005-0.003(AdamW)
  - lrf: 0.05-0.2
  - momentum: 0.90-0.98
  - weight_decay: 0.0005-0.001（SGD）/0.01-0.05（AdamW）
  - mosaic: 0.3-1.0，mixup: 0.0-0.2，degrees: 0-10，translate: 0-0.2，scale: ~0.5，fliplr: 0.5