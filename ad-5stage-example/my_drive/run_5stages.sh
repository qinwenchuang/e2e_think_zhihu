#!/usr/bin/env bash
# =============================================================================
# my_drive 5 阶段训练一键脚本
#   stage 0:   pretext 未来特征预测（md_pretext，自监督，需 pretext 缓存）
#   stage 1:   冻结 DINO 主干，LoRA + Planner 可训（md_train --stage 1）
#   stage 2:   仅 BEV 头可训（md_train --stage 2）
#   stage 3:   LoRA + Planner + BEV 联合可训（md_train --stage 3）
#   stage 3.5: GRPO 强化微调 scorer（md_grpo，需验证集划分）
# 阶段间自动衔接：stage0 best_val → stage1 → stage2 → stage3 → grpo best。
#
# 用法：
#   bash my_drive/run_5stages.sh                 # 默认全量配置
#   DRY_RUN=1 bash my_drive/run_5stages.sh       # 只打印将执行的命令
#   SKIP_PRETEXT=1 SKIP_GRPO=1 bash my_drive/run_5stages.sh   # 只跑 3 个 SFT 阶段
#   FORESIGHT_TARGET=proj bash my_drive/run_5stages.sh        # stage 0 用
#       可学习 projector + SIGReg（AD-E2E-JEPA 化）
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
#   FORESIGHT_TARGET  stage 0 预测目标：proj（默认）/ last
#                 proj = 可学习 PatchProjector + SIGReg（arXiv:2609.34085），
#                 patch 16×↓、维度 4×↓
#   SIGREG_WEIGHT   SIGReg 权重 λ（默认读 Config=0.09；有效 batch 大时建议下调，
#                 论文 batch=512 时用 0.025）
#   SIGREG_DIRS     SIGReg 随机方向数 M（默认读 Config=256；论文为 1024）
#   ROLLOUT=1       proj 模式开自回归 rollout 一致性损失（TBPTT，论文 Eq.12）
#   NO_PLAN_EVAL=1  关 val 的 hit@1 规划诊断（proj 模式默认开）
#   PREP_DATA=1   缓存缺失时自动构建有标签缓存（TRAIN/VAL/TEST_COUNT 控制数量）
#   PREP_PRETEXT=1 缓存缺失时自动构建 pretext 缓存（NUM_WINDOWS 控制窗口数）
#                 注意：新版为 8 帧多步长窗口，与旧 5 帧缓存不兼容
#   MAX_LOGS      数据准备时最多扫描的日志数（默认 2000）
#   EXTRA         追加给 md_train 的额外参数（如 --limit-batches 4 --no-tb）
#   DRY_RUN=1     只打印命令不执行
#
# 轨迹表征 / top-k / 几何重打分开关（TopoCurve arXiv:2610.09118 / MGMP arXiv:2610.10646 借鉴，默认开）：
#   TRAJ_REPR     轨迹表征：bezier（默认，Bézier 控制点）/ raw（旧逐点回归）
#   BEZIER_CTRL   控制点数（默认 5；4 = 三次曲线）
#   TOPK          每层监督 top-k（默认 5；1 = 旧 WTA）
#   TOPK_DECAY    top-k 第 j 名权重衰减（默认 0.5）
#   GEO_RESCORE=0  关推理几何重打分
#   MAP_CCL=1      开地图 PA+BANE 辅助损失（默认 0；仅 stage2/3 训练）
#   MAP_CCL_WEIGHT 辅助损失权重（默认 0.1）
#   TOAD=1         开测试时控制空间 CEM（默认 0；仅 eval，不改训练损失）
#   TOAD_ITERS / TOAD_SAMPLES / TOAD_ELITES / TOAD_STD_SCALE
#   TOAD_ANCHOR / TOAD_COMFORT / TOAD_SEED
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-/home/qwc/navsim_venv}"
source "$VENV/bin/activate"

WORK_ROOT="${MD_WORK_ROOT:-my_drive/workspace}"
CACHE="${CACHE:-$WORK_ROOT/cache}"
PRETEXT_CACHE="${PRETEXT_CACHE:-$WORK_ROOT/cache_pretext}"
RUNS="${RUNS:-$WORK_ROOT/runs}"

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
STAGE1_CKPT="${STAGE1_CKPT:-}"
SKIP_PRETEXT="${SKIP_PRETEXT:-0}"
SKIP_GRPO="${SKIP_GRPO:-0}"
FORESIGHT_TARGET="${FORESIGHT_TARGET:-proj}"
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
TRAJ_REPR="${TRAJ_REPR:-bezier}"
BEZIER_CTRL="${BEZIER_CTRL:-5}"
TOPK="${TOPK:-5}"
TOPK_DECAY="${TOPK_DECAY:-0.5}"
GEO_RESCORE="${GEO_RESCORE:-1}"
MAP_CCL="${MAP_CCL:-0}"
MAP_CCL_WEIGHT="${MAP_CCL_WEIGHT:-0.1}"
# TOAD 测试时 CEM 优化（仅 eval；默认关，TOAD=1 开启）
TOAD="${TOAD:-0}"
TOAD_ITERS="${TOAD_ITERS:-5}"
TOAD_SAMPLES="${TOAD_SAMPLES:-64}"
TOAD_ELITES="${TOAD_ELITES:-8}"
TOAD_STD_SCALE="${TOAD_STD_SCALE:-0.5}"
TOAD_ANCHOR="${TOAD_ANCHOR:-0.5}"
TOAD_COMFORT="${TOAD_COMFORT:-0.05}"
TOAD_SEED="${TOAD_SEED:-2}"

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

# stage 1-3 附加参数（轨迹表征 / top-k 监督 / 几何重打分）
SFT_EXTRA=(--traj-repr "$TRAJ_REPR" --bezier-ctrl "$BEZIER_CTRL"
           --topk "$TOPK" --topk-decay "$TOPK_DECAY")
if [ "$GEO_RESCORE" = "0" ]; then SFT_EXTRA+=(--geo-rescore-off); fi
if [ "$MAP_CCL" = "1" ]; then
    SFT_EXTRA+=(--map-ccl --map-ccl-weight "$MAP_CCL_WEIGHT")
fi
if [ "$TOAD" = "1" ]; then
    SFT_EXTRA+=(--toad --toad-iters "$TOAD_ITERS" --toad-samples "$TOAD_SAMPLES"
                --toad-elites "$TOAD_ELITES" --toad-std-scale "$TOAD_STD_SCALE"
                --toad-anchor-weight "$TOAD_ANCHOR"
                --toad-comfort-weight "$TOAD_COMFORT" --toad-seed "$TOAD_SEED")
fi

INIT1=()
if [ -n "$STAGE0_CKPT" ]; then
    # 外部 pretext 权重：跳过 stage 0 训练
    INIT1=(--init-from "$STAGE0_CKPT")
elif [ "${SKIP_PRETEXT}" = "1" ]; then
    echo "stage 0 跳过（SKIP_PRETEXT=1）→ stage 1 从随机初始化开始"
elif [ -f "$PRETEXT_CACHE/index.json" ] && has_val "$PRETEXT_CACHE/index.json"; then
    echo "==================== stage 0: pretext（未来特征预测，target=${FORESIGHT_TARGET}）===================="
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

# V2：占据监督依赖 cache/rays；缺失时默认直接失败，避免“以为在跑 V2、实际 occ 全跳过”。
if [ ! -d "$CACHE/rays" ]; then
    if [ "${ALLOW_NO_OCC:-0}" = "1" ]; then
        echo "警告：$CACHE/rays 不存在，ALLOW_NO_OCC=1 → 跳过占据监督"
    else
        echo "错误：$CACHE/rays 不存在，V2 的占据监督会静默失效。"
        echo "      先生成射线（python -m my_drive.data.md_prepare_rays_lidar ...），"
        echo "      或显式设置 ALLOW_NO_OCC=1 只跑轨迹/BEV 消融。"
        exit 1
    fi
fi

if [ -n "$STAGE1_CKPT" ]; then
    echo "stage 1 跳过（STAGE1_CKPT=$STAGE1_CKPT）"
else
    echo "==================== stage 1: LoRA + Planner（冻结 DINO）===================="
    run python -m my_drive.train.md_train --stage 1 --epochs "$EPOCHS1" --lr "$LR1" \
        "${INIT1[@]}" "${SFT_EXTRA[@]}" \
        --cache-root "$CACHE" --out-dir "$RUNS/stage1" \
        --batch-size $BATCH --grad-accum $ACCUM --num-workers $WORKERS --patience $PATIENCE $EXTRA
fi
STAGE1_REF="${STAGE1_CKPT:-$(best_of "$RUNS/stage1")}"


echo "==================== stage 2: 仅 BEV（冻结 DINO+LoRA+Planner）===================="
run python -m my_drive.train.md_train --stage 2 --epochs "$EPOCHS2" --lr "$LR2" \
    --init-from "$STAGE1_REF" "${SFT_EXTRA[@]}" \
    --cache-root "$CACHE" --out-dir "$RUNS/stage2" \
    --batch-size $BATCH --grad-accum $ACCUM --num-workers $WORKERS --patience $PATIENCE $EXTRA

echo "==================== stage 3: LoRA + Planner + BEV 联合 ===================="
run python -m my_drive.train.md_train --stage 3 --epochs "$EPOCHS3" --lr "$LR3" \
    --init-from "$(best_of "$RUNS/stage2")" "${SFT_EXTRA[@]}" \
    --cache-root "$CACHE" --out-dir "$RUNS/stage3" \
    --batch-size $BATCH --grad-accum $ACCUM --num-workers $WORKERS --patience $PATIENCE $EXTRA

if [ "${SKIP_GRPO}" = "1" ]; then
    echo "stage 3.5 GRPO 跳过（SKIP_GRPO=1）"
elif has_val "$CACHE/index.json"; then
    echo "==================== stage 3.5: GRPO（scorer 强化微调）===================="
    GRPO_EXTRA=(--traj-repr "$TRAJ_REPR" --bezier-ctrl "$BEZIER_CTRL")
    run python -m my_drive.train.md_grpo --init-from "$RUNS/stage3/last.pt"         "${GRPO_EXTRA[@]}"         --cache-root "$CACHE" --out-dir "$RUNS/grpo" --epochs "$GRPO_EPOCHS"         --batch-size $BATCH --grad-accum $ACCUM --num-workers $WORKERS
else
    echo "stage 3.5 GRPO 跳过（缓存无 val 划分）"
fi

echo "全部完成 → $RUNS"
