# 半合成 Power 分析方案与当前问题总结

本文档总结当前在 `semisynthetic_power.py` 中实现的半合成 power 分析思路、已经观察到的问题，以及后续修改重点。核心背景是：原始 bootstrap 在目标样本量超过真实每组样本数时会重复抽到真实样本，导致大样本外推不可靠。因此新脚本尝试先用预实验数据生成更大的 semi-synthetic sample pool，再在该 pool 上评估 power 曲线。

## 当前方案

### 总体思路

新算法不修改 `phylopower/core.py`，而是在独立脚本 `semisynthetic_power.py` 中实现实验入口。

基本流程是：

1. 读取用户预实验数据和分组信息。
2. 根据真实数据估计组内变异、组间中心差异、零值或非零结构。
3. 生成每组更大的 synthetic pool。
4. 对不同 effect level 计算距离矩阵。
5. 在 synthetic pool 上进行有放回 bootstrap，抽取目标样本量。
6. 用 PERMANOVA p 值估计 power，并用 anchored sigmoid 经过 `(0, alpha)` 拟合 omega²-power 曲线。

正式样本量外推时，不使用真实 observed_n 范围内的 power 校准未来大样本。`pilot_n=5/10/17 -> eval_n=17` 只作为开发验证：看不同 pilot size 生成的曲线是否稳定，以及生成器是否能复现真实 17v17 的结构。

### 基因组数据的半合成生成

基因组数据当前按照 feature x sample count table 处理。生成器主要保留 compositional count/CLR 结构：

- 估计每组 library size 分布。
- 估计 feature prevalence 和零值结构。
- 在 CLR 空间估计组中心和组内 residual。
- 生成 synthetic count table 后，再调用现有 Gemelli/RPCA 距离计算。
- effect 调节沿用现有几何调节逻辑。

这个方向相对合理，因为基因组数据主要是一维 feature abundance table。核心难点是 compositionality、zero inflation、library size 和组内距离结构。当前结果显示，基因组用半合成/bootstrapping 后的偏差相对可接受，没有蛋白质组那么明显。

### 蛋白质组数据的半合成生成

蛋白质组数据是 Taxon-Function 双维长表，不能简单当作普通 feature table。当前已将蛋白质组生成器从独立 Bernoulli presence 改为 `template-mask`：

- 每个 synthetic sample 从同组真实样本中抽一个 template sample。
- 完全复制该 template 的 Taxon-Function 非零边集合。
- 因此每个 synthetic sample 的 edge count、taxon degree、function degree 和缺失块结构与 template 一致。
- 只在 template 的非零边上生成 log-positive abundance。
- 零边保持为零。
- 生成后按 template total abundance 或同组经验 total abundance 重标定。
- effect 调节仍使用 `core.apply_rwct_effect_scaling`。

这个修改的目的，是优先保留蛋白质组的双维网络拓扑，而不是让每个 Taxon-Function pair 独立决定是否出现。之前 Bernoulli 生成器会破坏 degree distribution 和 edge co-occurrence，导致蛋白质组 power 明显高估。

## 当前运行结果

最近主要跑的是蛋白质组 sensitivity：

- `pilot_n=5,10,17`
- `eval_n=17`
- `boot=200`
- `protein-generator=template-mask`
- 有放回 bootstrap
- effect level 使用等距点
- increase: `0..2.4`
- decrease: `0..-2`
- increase/decrease 各 15 个点
- sigmoid 拟合默认使用所有有限点，包括 `power=1` 的点

结果图：

- 合图：`/Users/liyutong/Desktop/figure/semisynthetic_pilot_sensitivity_b200/protein_equal_range_template_mask/taxon-function_equal_range_combined_common_axis.png`
- `pilot_n=5`：`/Users/liyutong/Desktop/figure/semisynthetic_pilot_sensitivity_b200/protein_equal_range_template_mask/taxon-function_equal_range_pilot_n5_common_axis.png`
- `pilot_n=10`：`/Users/liyutong/Desktop/figure/semisynthetic_pilot_sensitivity_b200/protein_equal_range_template_mask/taxon-function_equal_range_pilot_n10_common_axis.png`
- `pilot_n=17`：`/Users/liyutong/Desktop/figure/semisynthetic_pilot_sensitivity_b200/protein_equal_range_template_mask/taxon-function_equal_range_pilot_n17_common_axis.png`

关键数值：

| pilot_n | baseline omega² | baseline power | transition points | fitted power at real 17v17 omega² |
|---:|---:|---:|---:|---:|
| 5 | 0.128 | 1.00 | 0/30 | 0.992 |
| 10 | 0.020 | 0.58 | 26/30 | 0.815 |
| 17 | 0.032 | 0.52 | 27/30 | 0.691 |

真实 17v17 点大约为：

- `omega²≈0.045`
- `power≈0.55`

因此，`pilot_n=17` 的 baseline power 已经接近真实点，但 sigmoid 在真实 omega² 附近仍偏高。`pilot_n=5` 基本失败：它从起点就已经 power=1，曲线没有可识别的上升过程。

## 基因组当前问题

基因组数据目前问题相对温和，主要是常规 semi-synthetic microbiome 模拟问题：

1. 小 pilot 下组中心估计不稳定。
   当每组样本很少时，CLR 组中心可能包含抽样噪声，生成器会把这部分噪声当成真实组间差异外推。

2. 零值和 prevalence 的外推仍较简化。
   当前模型能保留一定 zero/prevalence 结构，但没有显式建模复杂的 feature-feature correlation 或生态共现结构。

3. 大样本外推依赖生成器质量。
   n 超过 observed_n 后，power 曲线不再由真实样本重复抽样驱动，而是由 synthetic pool 决定。因此生成器如果低估组内变异或高估组间差异，会直接影响样本量估计。

当前基因组 sensitivity 已新增 `ordination` engine，作为和蛋白质组一致的主线：

- pilot count table 先计算 Gemelli/RPCA distance。
- 对 pilot distance 做 PCoA。
- 每组估计 centroid 和 Ledoit-Wolf covariance。
- 生成 large MVN distance pool。
- 用 centroid-separation scale 增强/减弱效应。
- 从 large pool 有放回抽目标样本量，PERMANOVA 显著比例作为 power。
- `--effect-grid adaptive` 时按 `eval_n` 收缩 scale 范围，并加密低 scale/低 omega 转折区。

最新修改中，ordination/MVN 的中心放置新增 `--center-mode`：

- `observed`：scale=1 保留 pilot distance 在 PCoA/Gemelli loading 空间中实际观察到的组间 centroid separation。它只能表示“原始中心距离”，不能保证保留 pilot 原始 omega²。
- `debiased`：原 positive-part shrinkage，按 centroid 不确定性扣除小样本中心距离膨胀。它更适合做保守的未来样本量外推，但不适合解释为“原始点”。
- `empirical-bayes`：连续的 signal/(signal+noise) shrinkage，避免小 pilot 被 positive-part 一次性压得过低；后续可作为 prospective power 的候选主线测试。
- `omega-calibrated`：先计算 pilot 自己的 PERMANOVA omega²，再反解 synthetic large-pool 在 scale=1 时应有的 centroid separation。它比 `observed` 更适合作为“原始效应锚点”，因为 `observed` 只保留 centroid distance，生成大 pool 后可能显著放大 omega²。

Gemelli/RPCA 的小样本曲线必须区分这两类问题：`observed` 先验证“scale=1 是否贴住 pilot 原始效应”，`debiased/empirical-bayes` 再验证“用 pilot 推未来目标样本量是否稳健”。不能把去偏后的 scale=1 点拿来当原始 pilot 效应点。

更新后的判断：真正的“原始效应曲线”应优先使用 `omega-calibrated`，而不是 `observed`。原因是 power 曲线横坐标是 omega²，omega² 同时受到 centroid separation、组内 covariance 和样本量校正项影响；仅保留 centroid separation 不等于保留原始 omega²。

`omega-calibrated` 之后 effect range 也必须同步改单位：外部的 `--ordination-enhance-max` 仍表示相对原始 observed centroid 的最大增强，但内部 scale 是相对校准后的 centroid。若校准 shrinkage 为 `s`，内部最大 scale 应约为 `ordination_enhance_max / s`。否则像 `pilot_n=4` 这种 `s≈0.29` 的情况，固定 scale `0..3` 实际只覆盖到原始 observed centroid 的 `0.87` 倍，所有点会挤在低 omega 端，曲线看起来反而更差。当前实现已按 shrinkage 自动扩展增强范围，并用 `--omega-calibrated-max-scale` 做安全上限。

进一步新增 `--effect-grid omega-uniform`：先用 dense candidate scale 做一次不跑 bootstrap 的 preview，只计算每个 candidate synthetic pool 的 omega²，然后反选出 omega² 上更均匀的一组 scale。这个模式用于避免 scale 很小的多个点都被 omega² 校正截到 0，导致图上大量点堆在横坐标低端。它比纯 scale-uniform/adaptive grid 更适合论文图中展示 omega²-power sigmoid 曲线。

进一步新增 `sensitivity-taxon --engine gemelli-loading`，用于让基因组模拟更贴近 Gemelli 自身机制：

- 直接按 Gemelli 源码链路计算 pilot 的 phylogenetic rclr table。
- 使用 `MatrixCompletion` 的 sample weights `U` 作为模拟坐标。
- 这与 Gemelli 输出 distance matrix 的定义一致，因为 Gemelli distance 是 `cdist(U, U)`，不是最终 biplot 样本坐标的距离。
- 后续仍在该坐标中估计 Ledoit-Wolf covariance、按 `--center-mode` 调制 centroid separation、生成 large pool，并有放回 bootstrap。

因此，`gemelli-loading` 是比 `ordination` 更 Gemelli-native 的快速模拟器；`ordination` 是从 Gemelli distance 再 PCoA 的等距近似。二者若结果接近，说明当前误差主要不是 PCoA 嵌入造成；若差异主要出现在低效应/低 pilot 子集，则需要进一步用 pilot-repeat ensemble 和 table-level rerun Gemelli 校验。

原有 CLR residual + library size + prevalence 的 table generator 保留为 `--engine table`，用于生物学诊断和 count-level fidelity 检查。二者的 power 估计框架一致，但数据结构保留重点不同：基因组保留 compositional count、zero/prevalence、library size 与 Gemelli latent geometry；蛋白质组保留 PhyloFunc distance 与 Taxon-Function 双维拓扑信息。

## 蛋白质组当前主要问题

蛋白质组是当前真正的难点。它的问题不是简单的点数、bootstrap 次数或 sigmoid 拟合，而是生成器是否正确保留了 Taxon-Function 双维数据结构。

### 1. 小 pilot 会严重放大组间差异

`pilot_n=5` 时，baseline 已经达到：

- `omega²≈0.128`
- `power=1.00`

这说明只抽 5 个样本时，组中心差异被估计得过强。生成器随后把这个强组间差异当作真实生物信号，生成了一个“组间已经明显分开”的 synthetic population。此时无论 effect 怎么调，曲线都会从高 power 开始，无法再用于判断真实样本量需求。

这个问题不能靠增加 effect 点解决。点再多，起点已经错了。

### 2. template-mask 修复了拓扑，但没有完全修复 abundance 差异

`template-mask` 已经解决了一个重要问题：非零边集合、edge count、taxon degree 和 function degree 不再被 Bernoulli 生成器破坏。

但它仍然可能高估 power，原因是：

- template mask 保留了每个样本的检测结构，但 log-positive abundance 的组中心差异仍可能偏强。
- 如果 pilot 中某些 Taxon-Function edge 恰好组间差异很大，模型会把它当成稳定信号。
- abundance residual 目前主要是围绕组中心生成，可能没有充分保留真实数据中的异质性、批次噪声和 edge-edge covariance。
- Taxon 和 Function 两个层级之间的相关结构还没有被完整建模。

也就是说，当前模型已经保留了“哪里有边”，但“边上的强度如何共同变化”还不够真实。

### 3. 主 power 估计标准

当前主 power 定义统一为 large-pool semi-synthetic bootstrap：

1. 真实 pilot/full data 只用于估计 semi-synthetic population。
2. 每个 effect level 先生成足够大的 synthetic pool。
3. 每次从该 synthetic pool 中有放回抽取目标样本量。
4. 对每个 bootstrap 距离矩阵运行 PERMANOVA。
5. `p < alpha` 的比例定义为 prospective power。

这保留 Kelly/micropower 的 bootstrap-PERMANOVA power 形式，但避免直接从原始小样本池中有放回抽样。输出中记录 `pool_size_per_group`、`draw_n_per_group`、`expected_duplicate_slots_per_group` 和 `expected_duplicate_fraction`，用于确认大池有放回抽样产生的重复样本风险足够低。

原始真实样本的有限池 bootstrap 只作为诊断结果，不再作为主 power 真值。

### 4. RWCT effect 的高 gamma 区域可能不现实

之前使用 increase 到 6 时，很多点已经进入 power=1 平台，而且部分点出现很大的 library drift 或 log shift。强 gamma 点不是没有意义，它们能说明曲线已经覆盖到饱和区；但如果范围过大，会让图像和拟合受到不现实区域影响。

因此目前更合理的策略是：

- 点仍然等距。
- power=1 点保留并参与拟合。
- 重点调整 effect 范围，而不是用 adaptive 点密度。
- 蛋白质组暂时用 `increase 0..2.4`、`decrease 0..-2` 比 `increase 0..6` 更合理。

后续还需要根据不同数据自动判断合理范围，例如要求：

- 至少覆盖低 power、中间转折、高 power 平台。
- 不让过多点进入极端 library drift 区域。
- 不因为范围太窄导致看不到 power=1 平台。

### 4. sigmoid 拟合不是主要矛盾

目前 anchored sigmoid 会经过 `(0, alpha)`，这与 Fig2 风格一致。之前尝试过只用 transition/realistic 点拟合，但这会产生两个问题：

- `power=1` 点被弱化，导致平台信息丢失。
- adaptive 点密度会让单图和合图视觉上不一致。

因此现阶段更合适的做法是：

- 等距 effect points。
- 所有有限点参与拟合。
- diagnostics 只用于解释哪些区域可能不现实，不直接删除 power=1 点。

真正导致拟合偏高的原因，主要仍是 synthetic population 本身偏强，而不是 sigmoid 形式本身。

## 后续修改重点

### 优先级 1：蛋白质组 between-group shrinkage

需要避免小 pilot 把偶然组间差异当作真实差异完全外推。思路不是简单规定“小于多少样本就是小样本”，而是根据数据稳定性自适应 shrinkage。

可考虑：

- 对每个 Taxon-Function edge 的组间差异按不确定性 shrink。
- pilot 越小、组内方差越大、prevalence 越低，shrink 越强。
- 高置信度、高 prevalence、组内稳定的 edge 保留更多差异。
- 低置信度、稀疏、受少数样本驱动的 edge 差异向全局均值收缩。

这样可以避免粗暴地说 `n<10` 就小，而是让每个 edge 根据自身证据强弱决定保留多少组间信号。

### 优先级 2：保留蛋白质组双层相关结构

当前 template-mask 保留了 topology，但 abundance covariance 仍不足。后续应考虑：

- 在 template residual 上保留更多样本级相关结构。
- 区分 taxon-level total、function-level total 和 edge-level abundance。
- 先生成 sample-level/taxon-level/function-level latent factors，再生成 edge abundance。
- 保留同一样本内多个 function 或多个 taxon 的共同波动。

这比把每条 Taxon-Function edge 当作独立 feature 更符合蛋白质组数据本身。

### 优先级 3：effect range 自动选择，而不是 adaptive 点密度

这次尝试说明，adaptive 点密度不适合当前阶段，因为我们需要看完整曲线形态。更好的自动化方向是：

1. 先用少量 preview 点估计合理范围。
2. 确定 increase/decrease 的上下界。
3. 在最终范围内使用等距点正式跑 boot。

最终正式图仍应是等距点，而不是局部加密点。

范围选择目标：

- 包含 baseline 附近。
- 包含 target omega² 附近。
- 包含 power 从 alpha/低值上升到 0.8 的区域。
- 包含一小段 power=1 平台。
- 避免大量点落在极端 library drift 或不现实 log shift 区域。

### 优先级 4：把 diagnostics 从“删点依据”改成“解释依据”

当前可保留这些诊断：

- `edge_count_drift`
- `taxon_degree_drift`
- `function_degree_drift`
- `presence_diff_after_rwct`
- `max_taxon_degree_diff_after_rwct`
- `max_function_degree_diff_after_rwct`
- `rwct_shift_max`
- `rwct_lib_drift_med`
- `within_group_distance_ratio`

但默认不应直接因为 power=1 或 drift 偏大而删除点。更合适的是：

- 图中标注或表中记录。
- summary 中提示某些 effect 区域可能不现实。
- 范围选择时减少极端区域占比。
- 拟合默认仍使用所有有限点，除非用户明确选择过滤。

## 当前结论

当前半合成框架已经解决了原始距离矩阵 bootstrap 在 `n > observed_n` 时重复真实样本的问题。基因组方向基本可用，主要需要继续做诊断和稳健性检查。

蛋白质组方向已经从 Bernoulli presence 改为 `template-mask`，明显改善了 degree distribution 和拓扑保真，但仍存在 power 偏高问题。根本原因是蛋白质组的双维结构和 abundance covariance 没有完全被生成器捕捉，尤其是小 pilot 下组间差异容易被放大。

下一步最应该改的是蛋白质组生成器，而不是 sigmoid 或 bootstrap 本身。具体优先方向是：对组间 abundance 差异做不确定性 shrinkage，并进一步保留 Taxon-Function 双层相关结构；effect points 保持等距，重点自动选择合理范围。
