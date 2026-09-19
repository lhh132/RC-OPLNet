# RC-OPLNet

RC-OPLNet 是用于跨域糖尿病视网膜病变（DR）五级分类的研究项目。
项目在 DGDR/GDRNet 的训练框架上，结合残差置信度校准的联合域—类别重加权
（RC-JDCR）与有序原型学习（PROTO），缓解多中心数据的域偏移及类别不平衡。

## 方法与默认设置

- **RC-JDCR**：根据源域训练集的域—类别计数计算有界权重，结合置信度、残差裁剪和渐进预热。
- **PROTO**：维护跨批次 EMA 类别原型；原型分类约束在类别获得至少两个源域支持后启用，
  同时对当前批次的类别中心施加随 DR 等级距离变化的间隔约束。
- 保留 FundusAug 和原有混合监督损失，默认骨干为 ImageNet 预训练 ResNet-50。
- `--algorithm RC-OPLNet` 为默认方法，自动启用 RC-JDCR 与 PROTO，默认 `w_max=6`。
  可用 `--rc-jdcr-max-weight` 做敏感性实验。独立 CDF/ORD 消融损失不属于完整方法默认设置。
- 支持 DG/ESDG 数据协议；完整 RC-OPLNet 的跨域原型要求至少两个源域，因此使用多源 DG。
  消融实验使用 `--algorithm RC-OPLNet-Ablation`，按需启用组件。

为兼容已有组件和检查点配置，内部 `GDRNET` 配置字段、组件文件名及 `--gdrnet-*`
消融参数仍保留。完整方法的算法标识、运行记录、结果目录和安装包名均为 RC-OPLNet。

## 安装

使用 Python 3.10+ 和 NVIDIA CUDA GPU。先按本机 CUDA 环境安装匹配的 PyTorch/torchvision，
再在项目目录执行：

```bash
python -m pip install -e ".[dev]"
python main.py --help
```

PyTorch、torchvision 不由本项目安装依赖自动选择 CUDA 版本。
预训练 ResNet 权重会在首次运行时由 PyTorch 下载到缓存；离线运行需提前准备相同权重。

## 数据准备

代码包不包含数据集、预训练权重、训练结果或检查点。`dataset/` 是数据加载程序。
已保留官方划分文本 `GDRBench/splits/` 及预处理程序；实际数据独立存放：

```text
/path/to/GDRBench/
  images/APTOS/nodr/xxx.png
  masks/APTOS/nodr/xxx.png
  images/DEEPDR/...
  masks/DEEPDR/...
  splits/APTOS_train.txt
  splits/APTOS_crossval.txt
  splits_strict/APTOS_train.txt
  splits_strict/APTOS_crossval.txt
  ...
```

类别为 `nodr`、`mild_npdr`、`moderate_npdr`、`severe_npdr`、`pdr`，标签为 0–4。
划分文本每行是相对于 images/masks 的路径和整数标签，格式以随附官方文本为准。
使用 `tools/prepare_gdrbench.py --help` 查看数据整理参数，
使用 `tools/import_fgadr.py --help` 查看获得授权后的 FGADR 导入方式。
构建严格划分：

```bash
python tools/build_strict_splits.py --root /path/to/GDRBench --seed 42 --val-fraction 0.2
```

`--split-profile strict` 读取 `splits_strict/`；`official` 读取 `splits/`，不可混用结果。
严格划分的患者信息可得性限制记录在生成的 audit.json 中。
APTOS、DEEPDR、FGADR、IDRID、RLDR 用于五域留一域外实验，DDR/EYEPACS 仅作为额外测试域。

## 训练

在项目根目录执行，以下以 RLDR 为未见目标域：

```bash
python main.py --algorithm RC-OPLNet --root /path/to/GDRBench --dg_mode DG --split-profile strict --source-domains APTOS DEEPDR FGADR IDRID --target-domains RLDR --epochs 100 --batch-size 32 --num-workers 4 --val_ep 1 --seed 42 --output rc_oplnet_rldr_seed42
```

Linux 也可设置 `DATA_ROOT` 后运行 `bash scripts/train_rc_oplnet.sh`。
留一域外实验依次更换目标域，源域设为其他四域，并为每次运行设置不同的 `--output`。
使用源域 crossval AUC 选择最佳检查点，训练结束后重载最佳权重，再评估目标域。
目标域数据不用于训练、权重计数或模型选择。

结果目录为 `result/fundusaug/strict/RC-OPLNet/<output>/`，包含最佳骨干/分类器权重、
最后训练状态、验证/测试指标、TensorBoard 和运行配置。
`--output` 也接受绝对路径，方便把所有训练结果放到代码目录之外。
继续中断训练时使用相同配置和 `--output`，附加 `--resume`；勿改动总 epoch 数。

## 独立验证与测试

```bash
python main.py --algorithm RC-OPLNet --root /path/to/GDRBench --dg_mode DG --split-profile strict --source-domains APTOS DEEPDR FGADR IDRID --target-domains RLDR --eval-only --checkpoint-dir /path/to/training/run --output rc_oplnet_rldr_eval --num-workers 4
```

检查点目录应包含 `best_model.pth` 和 `best_classifier.pth`，骨干及类别数需与训练一致。
该模式不执行训练，分别输出源域验证与目标域测试指标，输出目录必须与检查点目录不同。
评估报告包含 pooled、逐域、域宏平均和最差域指标（AUC、ACC、Macro-F1）。

## 消融实验

`RC-OPLNet-Ablation` 复用同一训练实现，组件默认关闭，RC-JDCR 的默认权重上限为 4。
保留 JDCR、HCS-DCR、RC-JDCR、独立 ORD 及 PROTO 的消融功能：

```bash
python main.py --algorithm RC-OPLNet-Ablation --root /path/to/GDRBench --dg_mode DG --split-profile strict --source-domains APTOS DEEPDR FGADR IDRID --target-domains RLDR --gdrnet-joint-dcr --output ablation_jdcr
```

可分别换为 `--gdrnet-hcs-dcr`、`--gdrnet-rc-jdcr`、`--gdrnet-ordinal-loss` 或
`--gdrnet-proto-contrast`；三种重加权方案互斥，PROTO 要求至少两个源域。
完整 RC-OPLNet 继续默认使用 RC-JDCR + PROTO、`w_max=6`，无需这些消融开关。
`scripts/lodo/` 中保留的组件实验脚本已接入新消融入口；结果写入独立的
`RC-OPLNet-Ablation-*` 目录。原有文件名中的 `gdrnet` 表示历史组件来源。

## 运行测试

不需要数据的回归测试：

```bash
python -m pytest -q
```

默认跳过需要外部数据的集成测试；设置 `GDRBENCH_ROOT` 后可运行数据集审计测试。
使用真实数据进行短流程训练测试：

```bash
python main.py --root /path/to/GDRBench --algorithm RC-OPLNet --dg_mode DG --split-profile strict --source-domains APTOS DEEPDR FGADR IDRID --target-domains RLDR --epochs 2 --batch-size 16 --num-workers 2 --val_ep 1 --max-train-batches 2 --max-eval-batches 2 --output smoke
```

短测试只验证执行链路。截断评估可能缺少类别，导致 AUC 为 null；论文实验必须移除两个
`--max-*-batches` 参数并完成全部训练和评估。

## 致谢

**本项目 RC-OPLNet 基于 [DGDR](https://github.com/chehx/DGDR) 修改完成。**
感谢 Haoxuan Che、Yuhan Cheng、Haibo Jin、Hao Chen 等作者提供 GDRNet、GDRBench
及相关代码，为本项目的研究和实现提供基础。

```bibtex
@inproceedings{che2023DGDR,
  title={Towards generalizable diabetic retinopathy grading in unseen domains},
  author={Che, Haoxuan and Cheng, Yuhan and Jin, Haibo and Chen, Hao},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={430--440},
  year={2023},
  organization={Springer}
}
```
