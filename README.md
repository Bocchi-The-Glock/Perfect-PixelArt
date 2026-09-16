# PixelPerfect Plus

用传统图像处理将 AI 生成的伪像素画恢复为原生低分辨率 PNG。项目从零实现，未复制或导入相邻 `perfectPixel-main` 项目的代码。运行不需要 OpenCV、神经网络、远程 API 或自定义编译扩展。

这是一个待验证的工程算法：恢复结果是对输入网格的解释，不保证存在唯一的原始像素图。程序提供未经概率校准的**启发式置信分数**；缺少有效证据时保留原尺寸，并通过 stderr 说明情况。

## 目录

顶层有三个功能目录，源码放在 `source`，不是旧项目的副本：

```text
perfect_pixel_plus/
  input/                     # 生成的合成输入；也可放自己的图片
  output/                    # 示例 PNG 与显式运行评估得到的报告
  source/
    pixelperfect/            # 独立 Python 包
    tests/                   # pytest 测试
    evaluate.py              # 合成评估与性能测量
  pixelperfect.py            # 直接运行入口
  pyproject.toml
  README.md
```

## 安装与快速使用

需要 Python 3.10 或更高版本。运行依赖只有 NumPy、SciPy、Pillow；测试使用 pytest。SciPy 提供信号处理、稀疏覆盖矩阵与小范围数值优化，Pillow 负责方向修正及 PNG 读写。

```powershell
cd D:\Learn\Program\python\PixelArt\perfect_pixel_plus
python -m pip install -e ".[test]"
python pixelperfect.py -i input/synthetic/clean_integer.png -o output/restored.png --verbose
```

安装后也可以运行：

```powershell
pixelperfect -i input/synthetic/clean_integer.png -o output/restored.png
python -m pixelperfect -i input/synthetic/clean_integer.png -o output/restored.png
```

命令默认只写指定的一张 PNG，不创建报告或预览图。输出是原生低分辨率图；`--scale 4` 才会在保存时使用整数最近邻放大，绝不使用双线性插值放大。

```powershell
python pixelperfect.py -i input/synthetic/clean_integer.png -o output/fixed.png --pixel-size 8
python pixelperfect.py -i input/synthetic/clean_integer.png -o output/target.png --target-size 24x24
python pixelperfect.py -i input/synthetic/clean_integer.png -o output/palette.png --colors 16
python pixelperfect.py -i input/synthetic/clean_integer.png -o output/preview.png --scale 4
python pixelperfect.py -i input/synthetic/clean_integer.png -o output/debugged.png --debug-dir output/debug --verbose
```

附带 `input/synthetic/clean_integer.png` 是 24×24 真值经 8 倍最近邻放大的输入，上述命令可直接运行。自己的输入应使用与其成比例的目标尺寸。

## 参数与默认行为

| 参数 | 默认值 | 含义 |
|---|---|---|
| `-i, --input` | 必填 | 常见静态图片，支持 PNG/JPEG、RGB/RGBA及可转换模式 |
| `-o, --output` | 必填 | 输出必须以 `.png` 结尾，不能与输入同路径 |
| `--pixel-size` | 自动 | 源图像素间距，例如 `8` 或 `8x9`，可以是小数；仍优化起点 |
| `--target-size` | 自动 | 精确输出宽高，例如 `64x64`；与 pixel-size 互斥 |
| `--colors` | 不量化 | 1–256 个可见 RGB 颜色上限；不添加抖动 |
| `--scale` | 1 | 1–64 整数倍最近邻导出，原生结果尺寸不变 |
| `--sampling` | robust | robust、center、median；后两者作为对照 |
| `--local-warp` | auto | auto 或 off；限制为两组一维切割线 |
| `--min-pixel-size` | 2 | 自动搜索最小源图间距 |
| `--max-pixel-size` | 64 | 自动搜索最大源图间距，还会按输入尺寸裁剪 |
| `--square` | 关闭 | 要求横纵名义间距相等；默认只施加接近方形的软偏好 |
| `--debug-dir` | 不写 | 显式保存候选 JSON、网格图、格子置信图、重建图 |
| `--verbose` | 关闭 | 向 stderr 输出主要阶段耗时与结果尺寸 |

### 目标尺寸和图像边缘

- `target-size` 的两个值必须是正整数，且不能超过输入尺寸。放大请使用 `scale`。
- 目标尺寸模式不裁剪、不拉伸输入；纵横缩放间距只允许原图约一像素取整量级的差异。超过此误差的宽高比冲突直接报错，退出码为 2。
- 同时开启 `--square` 时更严格：要求目标尺寸对应的横纵名义间距完全相等。
- 目标尺寸模式固定切割线数量，边界固定为 `0, W` 和 `0, H`；内部起点平移受限于名义间距的 ±30%。因此输出尺寸严格等于指定值。
- 自动或 pixel-size 模式覆盖完整输入。两端小于名义格宽/高 20% 的残片合并到邻格；其余部分格子保留。不会丢弃任何输入面积。
- 每格的覆盖范围由相邻 `x_lines` 与 `y_lines` 唯一决定，坐标对应源图像素方框 `[x,x+1)×[y,y+1)`，不是像素中心。
- 不进行旧式的“差一行就增删一行凑正方形”处理。

### 透明度

完全透明的隐藏 RGB 会清零，不参与梯度、取色、调色板或误差评分。轮廓证据含 alpha 变化。检测副本和重建的轻微模糊使用预乘 alpha，避免隐藏黑色污染边缘。

robust 和 median 的每格 alpha 是按真实覆盖面积计算的平均值。全透明格始终透明，可见格的 RGB 只从可见输入估计；中心采样对照模式使用中心 alpha。输出保留 RGBA，调色板量化不改变 alpha，也不为全透明格分配颜色预算。

一个输出像素不能同时表示格内所有透明与不透明形状：不足一格的小透明孔可能变为半透明覆盖，无法承诺保留任意亚格结构。明显的整格孔洞已有测试。

## Python API

```python
from PIL import Image
from pixelperfect import Config, pixelize, save_result

result = pixelize(Image.open("input/synthetic/clean_integer.png"), Config(
    sampling="robust",
    local_warp="auto",
    # pixel_size=8,             # 或 target_size=(64, 64)，二者互斥
    # colors=32,
))
save_result(result, "output/api.png")
# 显式放大时：save_result(result, "output/api_4x.png", scale=4)
print(result.image.size, result.confidence)
print(result.grid["x_lines"], result.timings)
```

`pixelize` 也接受图片路径、`uint8` RGB/RGBA 数组或取值在 `[0,1]` 的浮点数组。会拒绝 NaN、无效形状和其他整数类型，避免猜测颜色范围。Pillow 路径会修正 EXIF 方向；动画输入被拒绝。颜色计算使用标准化编码 sRGB 欧氏距离，不声称是感知均匀的 ΔE；未做 ICC 色彩管理或线性光照恢复。

返回 `PixelizeResult`：

- `image`：Pillow 图像，**始终为原生低分辨率**；Config.scale 仅供 CLI 保存使用。
- `grid`：间距、起点、实际切割线、输出尺寸、漂移状态和候选元信息。
- `confidence`：0–1 启发式分数，不是正确概率。
- `cell_confidence`：每格颜色恢复的启发式可信度。
- `timings`：各阶段墙钟耗时（秒），含总时间。
- `diagnostics`：各候选分项得分、选择原因、警告、结构检查。

API 不写文件、不依赖命令行，也不会交互询问。低置信警告放在 diagnostics，由 CLI 输出到 stderr。

## 算法与模块

| 模块 | 实现 |
|---|---|
| `image_io.py` | EXIF、RGB/RGBA 标准化、数值检查、PNG/最近邻导出 |
| `features.py` | 原分辨率颜色与 alpha 梯度，最多 6 条带，限幅与等权投影 |
| `grid.py` | 有限周期候选、起点搜索、快速筛选、可选一维动态规划漂移 |
| `sampling.py` | 覆盖面积、稳健取色、center/median 对照和四邻域结构检查 |
| `palette.py` | 低分辨率上确定性、置信度与边缘加权的有限调色板 |
| `scoring.py` | 分数像素覆盖重建、受限模糊、误差/结构/复杂度评分 |
| `pipeline.py` | 串联、计时、选择候选、置信度与回退 |
| `diagnostics.py` | 显式诊断导出，无绘图库依赖 |

### 1. 候选与起点

检测副本仅用 sigma=0.45 源像素的轻度高斯平滑，原图和原始梯度保留用于验证。所有间距在原始坐标估计，没有先随意缩图。条带限制局部强纹理贡献，平坦或透明条带不参与投票。

从边缘投影的 FFT、自相关和间距直方图产生候选，包含半倍、两倍、±0.25 像素邻近值；FFT 与自相关来自同一信号，只作为同一证据族。最多对 3 个领先尺度做连续微调，每次最多 14 次目标评估。每个方向最多保留 6 个尺度。

每个尺度搜索起点，使用截断平方的边缘到网格距离损失，并检查多个条带的一致性。不会要求每条线有可见边缘。最多组合 36 对，快速排序后默认最多 4 个进入后续处理；保留主候选的可用半倍/两倍解释，不让附近小数候选挤满列表。全部快速筛选分数与保留情况记录在元信息中。

完整评分还使用安全下界筛选：由于 J 的各项非负，当某候选的 `0.045 × 格子数 / 输入面积` 已大于现有最佳总分时，无须继续取色重建。诊断标记 `pruned_lower_bound`，记录下界，真实总分保留为 null，不伪造尚未计算的误差。置信间隔仅利用保守下界，不把淘汰的竞争者忽略掉。RGB 取色梯度在候选之间复用。

### 2. 漂移模型

只处理 `x[k]`、`y[j]` 两组全局切线。移动范围不超过 min(3 源像素, 18% 名义间距)，要求至少两个有效条带支持移动。动态规划同时考虑边缘收益、位置偏移与相邻间距代价，保持严格单调和完整覆盖；证据增益不超过复杂度门槛时不移动。

这不支持任意二维弯曲、旋转、透视或局部不同网格方向，也不是精确恢复模型。

### 3. 颜色、细线与量化

默认每格最多 128 个确定性空间分层 RGB 样本，结合覆盖面积、alpha、位置和局部梯度降权。alpha 使用该格完整覆盖统计，不抽样。近单色区域直接恢复代表色；混色格用低成本分桶初始化最多 3 个颜色簇，最多 4 轮更新，并保留少量备选色及空间证据。中心位置不能单独决定结果。不计算全样本两两距离。近单色格按最多 512 格一批向量化处理，复杂格保留逐格聚类；整数/分数切线与透明/不透明输入均测试了与标量参考的一致性。

结构规则统一使用**四邻域**；对角接触不视为连接。仅做一轮保守修正：颜色在格内至少有 8% 可见支持、呈细长形状，且两端邻格有同色证据时，允许提升次要色以接续直线。紧凑、多采样点支持、明显比背景亮且与四邻域背景一致的次要簇，也可作为高光保留；单个中心热噪点不会满足这条规则。记录明显透明孔和孤立细节，将它们标记给可选量化。没有无条件清除孤立像素的规则，也没有语义推理。紧凑成簇噪声仍可能被误认作高光。

量化只在恢复后网格上执行，最多 8192 个训练格、10 轮更新；考虑 alpha、置信度、局部边缘和稀有颜色。最多保留 `min(K//4, 8)` 个显著细节颜色作为固定中心，透明格不占 RGB 预算。颜色预算很小时仍可能合并细节，这会计入诊断，不保证任意 K 都能保持全部颜色关系。

### 4. 重建与评分

通过稀疏的一维像素/格子面积重叠矩阵，在**实际切割线**上重建预乘 RGBA。边缘部分格按实际可见面积计算；不会把不规则网格误当均匀最近邻放大。

比较无模糊与 sigma=0.5 源像素高斯模糊两个退化模型；模糊另加 0.0004 代价，不能任意增加模糊解释错误网格。

采用**越小越好**的代价（等价于最大化负代价的质量分数）：

```text
J = D + blur_penalty + 0.045 R + 0.035 E_structure + 0.020 E_grid
```

- `D`：黑/白背景复合颜色与 alpha 的 Huber 类稳健误差，按全部输入像素面积归一化；隐藏 RGB 不影响它。
- `R`：输出格数/输入面积，加可见颜色表达和横纵邻域颜色变化次数的代理代价。不同尺寸使用同一个输入面积分母，过细网格必须支付更多代价。
- `E_structure`：有输入支持的强边缘在重建中丢失的比例，允许 1 源像素定位误差；包括 alpha 边缘。结合前述格级线/孔/点检查，但不构成严格拓扑等价证明。
- `E_grid`：归一化边缘解释误差与漂移惩罚。

复杂度项是工程代理，不是严格最优编码长度；包含空间变化，不只是直方图熵；不以 PNG 文件大小判断质量。各候选使用相同颜色预算与同一套权重，不能直接跨不同 `--colors` 运行比较裸颜色误差。权重尚未通过大规模数据校准。

### 5. 置信与回退

置信综合有效边缘支持、条带一致性、前两候选差距、重建质量和是否触及搜索边界。高重建质量不能弥补没有边缘证据的情况。

自动低置信时，从与最优得分相近（绝对差≤0.006）的候选中选择格子较多者，减少激进降采样。完全缺少双向有效周期证据则保留原尺寸并返回 0 分，输出仍正常生成。提供间距/目标约束时即使低置信也完成约束下输出。手动指定的格子尺寸不因此被宣称正确。

## 测试与合成评估

```powershell
python -m pytest
python source/evaluate.py
```

评估脚本显式生成 input 内的素材、output 内的恢复结果及 `output/evaluation/` 中的 JSON/Markdown 报告，包括一次 1024×1024 自动处理计时。这是独立评估命令，不是普通 CLI 的默认副作用。

在本次受限环境中，pytest 安装在项目内 `source/.test_deps`，也可使用以下命令复现，无需改全局 Python：

```powershell
$env:PYTHONPATH = "source;source/.test_deps"
python -m pytest
```

测试覆盖干净网格尺寸/起点、半倍与两倍候选、分数覆盖、中心噪点、整格细线/高光/透明孔、隐藏 RGB 不变性、目标尺寸/最近邻、确定性、损坏文件、EXIF、纯色与小图、CLI 参数和默认只保存结果图。非整数缩放、模糊、压缩、缺失边界、重复纹理和漂移另外作为质量探针，不把“生成了有效 PNG”写成“正确恢复了唯一真值”。

实测环境、分项耗时、采样对照和失败案例见 [评估报告](output/evaluation/EVALUATION.md)。测试没有下载图像，也没有用旧项目效果图充当新算法结果。

### 本次验证记录（2026-09-16）

- 89 项自动化测试通过；wheel 实际构建并安装到临时项目内目录，已用安装后的 `python -m pixelperfect` 完成 PNG 导出，随后清理安装检查副本。
- Windows 11、Python 3.12.11、NumPy 1.26.4、SciPy 1.16.2、Pillow 12.1.0；系统报告 20 个逻辑 CPU，具体 CPU 型号未取得。
- 1024×1024 合成输入自动恢复成 64×64：一次预热进程内实测 3.770 秒，不含磁盘读写，RGB/alpha MAE 均为 0。优化前同工作区另一次记录为 20.845 秒；非受控多次统计实验，不宣称稳定倍数或实时。
- 10 个自动探针有 9 个输出尺寸与生成真值一致；尺寸一致不等于内容完全正确。双线性非整数缩放案例保留原尺寸、置信为 0，仍列在报告中。
- 模糊＋JPEG、固定目标网格的 RGB MAE（0–255）：center 5.979，median 5.556，robust 2.732；中心噪点案例分别为 125.142、0、0。

## 已知限制

1. 大片同色、重复纹理、极少边界可能存在多个同样合理的格数。启发式置信未校准，仍可能判断错误。
2. 半倍/两倍候选参与比较，但有限候选预算仍可能遗漏正确尺度；触及默认 2–64 搜索边界时应按输入调整范围。
3. 只支持全局水平/竖直网格及有限一维漂移；不处理任意二维弯曲、透视、旋转网格或多个独立尺度。
4. RGB 代表色为近似统计，源图细节已丢失时不能重造。小于一格的眼睛、高光或透明孔可能被合并。
5. 结构代价主要是边缘与局部四邻域代理；不是语义识别，也不保证所有连通性或孔洞拓扑。
6. 抽样、候选数和迭代数有上限；大幅模糊、强压缩与连续渐变可能让模型失配。复杂混色格比干净格更慢。
7. 大图会保留原始数组、梯度和候选重建，内存随输入面积增长。没有宣称实时、没有 GPU 路径。
8. 首版仅单张静态图；无 GUI、批处理、动画、精灵表分割或自动透视校正。
