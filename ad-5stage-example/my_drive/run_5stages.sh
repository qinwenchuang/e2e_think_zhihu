#!/usr/bin/env bash
# =============================================================================
# my_drive 5 阶段训练一键脚本
#   stage 0:   pretext 未来特征预测（md_pretext，自监督，需 pretext 缓存）
#   stage 1:   冻结 DINO 主干，LoRA + Planner 可训（md_train --stage 1）
#   stage 1.5: GRPO 强化微调 scorer（md_grpo，需验证集划分）
#   stage 2:   仅 BEV 头可训（md_train --stage 2）
#   stage 3:   LoRA + Planner + BEV 联合可训（md_train --stage 3）
# 阶段间自动衔接：stage0 best_val → stage1 → grpo best → stage2 → stage3。
#
# 用法：
#   bash my_drive/run_5stages.sh                 # 默认全量配置
#   DRY_RUN=1 bash my_drive/run_5stages.sh       # 只打印将执行的命令
#   SKIP_PRETEXT=1 SKIP_GRPO=1 bash my_drive/run_5stages.sh   # 只跑 3 个 SFT 阶段
#   FORESIGHT_TARGET=proj bash my_drive/run_5stages.sh        # stage 0 用
#       可学习 projector + SIGReg（AD-E2E-JEPA 化；自动跳过 prepare-pca）
#   FORESIGHT_TARGET=proj ROLLOUT=1 bash my_drive/run_5stages.sh  # 再加 rollout 一致性
#
# 可用环境变量：
#   VENV          虚拟环境路径（默认 /home/qwc/navsim_venv）
#   CACHE         有标签缓存（默认 my_drive/workspace/cache）
#   PRETEXT_CACHE pretext 缓存（默认 my_drive/workspace/cache_pretext）
#   RUNS          输出目录（默认 my_drive/workspace/runs）
#   EPOCHS0/1/2/3、GRPO_EPOCHS   各阶段 epoch 数（默认 10/50/8/50/50）
#   BATCH / ACCUM / WORKERS      批大小 / 梯度累积 / 数据加载进程
#   LR0/1/2/3     各阶段学习率（默认 2.5e-4 / 2.5e-4 / 2.5e-4 / 1e-4）
#   PATIENCE      val 早停耐心（默认 6 次评估）
#   STAGE0_CKPT   外部 pretext 权重（设置后跳过 stage 0 训练，直接用它初始化 stage 1）
#   SKIP_PRETEXT=1 跳过 stage 0；SKIP_GRPO=1 跳过 stage 1.5（stage 2 改从 stage 1 初始化）
#   FORESIGHT_TARGET  stage 0 预测目标：pca（默认）/ last / proj
#                 proj = 可学习 PatchProjector + SIGReg（arXiv:2609.34085），
#                 patch 16×↓、维度 4×↓，无需 prepare-pca
#   SIGREG_WEIGHT   SIGReg 权重 λ（默认读 Config=0.09；有效 batch 大时建议下调，
#                 论文 batch=512 时用 0.025）
#   SIGREG_DIRS     SIGReg 随机方向数 M（默认读 Config=256；论文为 1024）
#   ROLLOUT=1       proj 模式开自回归 rollout 一致性损失（TBPTT，论文 Eq.12）
#   NO_PLAN_EVAL=1  关 val 的 hit@1 规划诊断（proj 模式默认开）
#   PREP_DATA=1   缓存缺失时自动构建有标签缓存（TRAIN/VAL/TEST_COUNT 控制数量）
#   PREP_PRETEXT=1 缓存缺失时自动构建 pretext 缓存（NUM_WINDOWS 控制窗口数）
#                 注意：新版为 8 帧多步长窗口，与旧 5 帧缓存不兼容
#                 pca 模式下 train 前会自动补跑 prepare-pca（若 pca.pt 缺失）
#   MAX_LOGS      数据准备时最多扫描的日志数（默认 2000）
#   EXTRA         追加给 md_train 的额外参数（如 --limit-batches 4 --no-tb）
#   DRY_RUN=1     只打印命令不执行
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-/home/qwc/navsim_venv}"
source "$VENV/bin/activate"

CACHE="${CACHE:-my_drive/workspace/cache}"
PRETEXT_CACHE="${PRETEXT_CACHE:-my_drive/workspace/cache_pretext}"
RUNS="${RUNS:-my_drive/workspace/runs}"

EPOCHS0="${EPOCHS0:-10}"
EPOCHS1="${EPOCHS1:-50}"
GRPO_EPOCHS="${GRPO_EPOCHS:-8}"
EPOCHS2="${EPOCHS2:-50}"
EPOCHS3="${EPOCHS3:-50}"
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-2}"
WORKERS="${WORKERS:-4}"
LR0="${LR0:-2.5e-4}"
LR1="${LR1:-2.5e-4}"
LR2="${LR2:-2.5e-4}"
LR3="${LR3:-1e-4}"
PATIENCE="${PATIENCE:-6}"
STAGE0_CKPT="${STAGE0_CKPT:-}"
SKIP_PRETEXT="${SKIP_PRETEXT:-0}"
SKIP_GRPO="${SKIP_GRPO:-0}"
FORESIGHT_TARGET="${FORESIGHT_TARGET:-pca}"
SIGREG_WEIGHT="${SIGREG_WEIGHT:-}"
SIGREG_DIRS="${SIGREG_DIRS:-}"
ROLLOUT="${ROLLOUT:-0}"
NO_PLAN_EVAL="${NO_PLAN_EVAL:-0}"
PREP_DATA="${PREP_DATA:-0}"
PREP_PRETEXT="${PREP_PRETEXT:-0}"
TRAIN_COUNT="${TRAIN_COUNT:-10000}"
VAL_COUNT="${VAL_COUNT:-2000}"
TEST_COUNT="${TEST_COUNT:-1500}"
NUM_WINDOWS="${NUM_WINDOWS:-50000}"
MAX_LOGS="${MAX_LOGS:-2000}"
EXTRA="${EXTRA:-}"

# 命令执行器：DRY_RUN=1 时只打印
run() { if [ "${DRY_RUN:-0}" = "1" ]; then printf '%s\n' "$*"; else "$@"; fi; }
# 各阶段产物选择：优先 best_val.pt，缺省回退 last.pt；GRPO 产物是 best.pt
best_of() { if [ -f "$1/best_val.pt" ]; then echo "$1/best_val.pt"; else echo "$1/last.pt"; fi; }
grpo_best() { if [ -f "$1/best.pt" ]; then echo "$1/best.pt"; else echo "$1/last.pt"; fi; }
# 缓存是否有 val 划分（GRPO / pretext 的验证集依赖）
has_val() { python -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if any(e.get("split")=="val" for e in d) else 1)' "$1"; }

run mkdir -p "$RUNS"

if [ "${PREP_DATA}" = "1" ] && [ ! -f "$CACHE/index.json" ]; then
    echo "==================== 数据准备：有标签缓存 ===================="
    run python -m my_drive.data.md_prepare --train "$TRAIN_COUNT" --val "$VAL_COUNT" \
        --test "$TEST_COUNT" --max-logs "$MAX_LOGS" --cache-root "$CACHE"
fi
if [ "${PREP_PRETEXT}" = "1" ] && [ ! -f "$PRETEXT_CACHE/index.json" ]; then
    echo "==================== 数据准备：pretext 缓存 ===================="
    run python -m my_drive.train.md_pretext prepare --num-windows "$NUM_WINDOWS" \
        --max-logs "$MAX_LOGS" --cache-root "$PRETEXT_CACHE"
fi

# stage 0 附加参数（proj 模式的 projector/SIGReg/rollout 开关）
S0_EXTRA=(--target "$FORESIGHT_TARGET")
if [ -n "$SIGREG_WEIGHT" ]; then S0_EXTRA+=(--sigreg-weight "$SIGREG_WEIGHT"); fi
if [ -n "$SIGREG_DIRS" ]; then S0_EXTRA+=(--sigreg-dirs "$SIGREG_DIRS"); fi
if [ "$ROLLOUT" = "1" ]; then S0_EXTRA+=(--rollout); fi
if [ "$NO_PLAN_EVAL" = "1" ]; then S0_EXTRA+=(--no-plan-eval); fi

INIT1=()
if [ -n "$STAGE0_CKPT" ]; then
    # 外部 pretext 权重：跳过 stage 0 训练
    INIT1=(--init-from "$STAGE0_CKPT")
elif [ "${SKIP_PRETEXT}" = "1" ]; then
    echo "stage 0 跳过（SKIP_PRETEXT=1）→ stage 1 从随机初始化开始"
elif [ -f "$PRETEXT_CACHE/index.json" ] && has_val "$PRETEXT_CACHE/index.json"; then
    echo "==================== stage 0: pretext（未来特征预测，target=${FORESIGHT_TARGET}）===================="
    # PCA 目标矩阵仅 pca 模式需要（proj 模式用可学习 projector，无需离线统计）
    if [ "$FORESIGHT_TARGET" = "pca" ] && [ ! -f "$PRETEXT_CACHE/pca.pt" ]; then
        echo "---- prepare-pca：预计算 PCA 目标矩阵 ----"
        run python -m my_drive.train.md_pretext prepare-pca --cache-root "$PRETEXT_CACHE"
    fi
    run python -m my_drive.train.md_pretext train --epochs "$EPOCHS0" --lr "$LR0" \
        "${S0_EXTRA[@]}" \
        --cache-root "$PRETEXT_CACHE" --out-dir "$RUNS/stage0" \
        --batch-size $BATCH --grad-accum $ACCUM --num-workers $WORKERS --patience $PATIENCE
    INIT1=(--init-from "$(best_of "$RUNS/stage0")")
else
    echo "stage 0 跳过（无 pretext 缓存 $PRETEXT_CACHE/index.json 或无 val 划分）"
fi

if [ ! -f "$CACHE/index.json" ]; then
    echo "缺少有标签缓存 $CACHE/index.json：先运行 md_prepare 或设置 PREP_DATA=1"
    exit 1
fi

echo "==================== stage 1: LoRA + Planner（冻结 DINO）===================="
run python -m my_drive.train.md_train --stage 1 --epochs "$EPOCHS1" --lr "$LR1" \
    "${INIT1[@]}" \
    --cache-root "$CACHE" --out-dir "$RUNS/stage1" \
    --batch-size $BATCH --grad-accum $ACCUM --num-workers $WORKERS --patience $PATIENCE $EXTRA

INIT2=()
if [ "${SKIP_GRPO}" = "1" ]; then
    echo "stage 1.5 跳过（SKIP_GRPO=1）→ stage 2 从 stage 1 初始化"
    INIT2=(--init-from "$(best_of "$RUNS/stage1")")
elif has_val "$CACHE/index.json"; then
    echo "==================== stage 1.5: GRPO（scorer 强化微调）===================="
    run python -m my_drive.train.md_grpo --init-from "$(best_of "$RUNS/stage1")" \
        --cache-root "$CACHE" --out-dir "$RUNS/grpo" --epochs "$GRPO_EPOCHS" \
        --batch-size $BATCH --grad-accum $ACCUM --num-workers $WORKERS
    INIT2=(--init-from "$(grpo_best "$RUNS/grpo")")
else
    echo "stage 1.5 跳过（缓存无 val 划分，GRPO 需要验证集）→ stage 2 从 stage 1 初始化"
    INIT2=(--init-from "$(best_of "$RUNS/stage1")")
fi

echo "==================== stage 2: 仅 BEV（冻结 DINO+LoRA+Planner）===================="
run python -m my_drive.train.md_train --stage 2 --epochs "$EPOCHS2" --lr "$LR2" \
    "${INIT2[@]}" \
    --cache-root "$CACHE" --out-dir "$RUNS/stage2" \
    --batch-size $BATCH --grad-accum $ACCUM --num-workers $WORKERS --patience $PATIENCE $EXTRA

echo "==================== stage 3: LoRA + Planner + BEV 联合 ===================="
run python -m my_drive.train.md_train --stage 3 --epochs "$EPOCHS3" --lr "$LR3" \
    --init-from "$(best_of "$RUNS/stage2")" \
    --cache-root "$CACHE" --out-dir "$RUNS/stage3" \
    --batch-size $BATCH --grad-accum $ACCUM --num-workers $WORKERS --patience $PATIENCE $EXTRA

echo "全部完成 → $RUNS"