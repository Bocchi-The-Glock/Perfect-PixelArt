# PixelPerfect Plus 0.3.0

将伪像素画恢复为原生低分辨率 PNG。运行依赖仅 **NumPy、Pillow**；pytest 只用于测试。不需要 SciPy、OpenCV、Matplotlib 或网络服务。

本版首先改进网格识别，撤回 0.2.1 的整格描黑、轮廓补实规则。算法独立实现；对比脚本直接调用未修改的 perfectPixel-main 公开入口，由原项目选择后端，没有复制其算法代码。

## 运行

在本项目目录执行：

```powershell
python -m pip install -e .
python pixelperfect.py -i .\input\hollow-knight-sprite.png
```

已有 NumPy、Pillow 时，可以直接运行第二条命令。仍只需 `-i`：结果写入本项目的 `output/<输入名>.png`，同名结果会更新。JPEG 等输入也导出同名 PNG；不同目录的同名输入需要用 `-o` 区分。安装后的独立 wheel 默认使用工作目录的 output。

每次命令行导出，同时生成：

```text
output/
  hollow-knight-sprite.png
  hollow-knight-sprite_debug/
    fft.png        原分辨率二维 FFT 的对数幅度；红线是最终间距的倒数频率
    edges.png      RGB + alpha 边缘；红色为 X 差分，绿色为 Y 差分
    grid.png       最终实际切割线
    profiles.png   横纵边缘投影及其 FFT
    curvature.png  插值斜率变化证据；红线是格子中心
    info.json      候选、分数、采用的证据模型、网格、透明度策略和耗时
    info.txt       最终尺寸、间距、起点等摘要
```

输出默认保存原生小图；仅显式指定 `--scale` 时做整数倍最近邻放大。诊断图可能缩小显示，但检测始终使用原分辨率证据。PNG 保留透明通道，诊断图的灰色背景只用于显示。

## 主要参数

| 参数 | 行为 |
|---|---|
| `-i` / `--input` | 必填，静态输入图片 |
| `-o` / `--output` | 可选，指定结果 PNG 路径 |
| `--pixel-size 8` 或 `7.5x8` | 固定横纵源图格距，仍估计起点 |
| `--target-size 64x64` | 精确输出指定尺寸，全幅均分；与 pixel-size 互斥 |
| `--sampling robust` | 默认，中心优先并检查孤立噪点 |
| `--sampling center` | 直接取中心，不做异常检查 |
| `--sampling median` | 内部采样点中值，用于噪声和取色对照 |
| `--alpha-mode auto` | 默认，保留所选样本的 alpha，不自动二值化 |
| `--alpha-mode coverage` | 使用格内实际面积平均 alpha |
| `--alpha-mode binary` | 显式将采样 alpha 按 0.5 阈值二值化；不补描边 |
| `--local-warp auto/off` | 默认 auto，只对少数候选吸附局部边界 |
| `--min-pixel-size / --max-pixel-size` | 自动搜索范围，默认 2～64，按图像尺寸裁剪 |
| `--square` | 强制相同横纵名义格距 |
| `--colors 32` | 可选低分辨率调色板量化，默认不限色、不抖动 |
| `--scale 4` | 整数倍最近邻导出，默认 1 |
| `--debug-dir` | 改变诊断目录，不关闭诊断 |
| `--verbose` | 打印阶段耗时 |

target-size 不允许超过输入尺寸；宽高比冲突超过一个源像素舍入误差时明确报错，不静默裁剪或拉伸。修正 EXIF 方向，拒绝损坏、无效尺寸和多帧输入。

## 算法概要（0.3.0，本次未改动）

### 1. 先区分颜色边界和插值节点

旧检测器会把双线性缩放产生的宽缓坡上的量化起伏当成多条边缘，因此常选到半倍格距。

现在有两种受约束的证据模型：

- **颜色边界模型**：对预乘 RGB 和 alpha 取方向差分；合并没有明显谷底的近等高峰，避免一条宽边缘重复投票。
- **线性插值模型**：在最多 64 条原分辨率横、纵扫描线上计算二阶颜色差分。双线性插值的斜率变化发生在原始像素的中心，须减去半个格距才能得到边界，不能直接拿来切格子。

仅当两个方向的“二阶变化总量 / 一阶变化总量”都小于 0.65 时，尝试第二种模型；它还必须满足网格支持分数至少 0.65、两个方向边缘解释度至少 0.75，且比边界模型高至少 0.08。该模型保持规则格线，不用局部漂移掩盖错误。用户给 target-size 时不启用它。

这里的阈值是工程参数，不是已经校准的概率。诊断保留两个模型的候选和选择原因。

### 2. FFT 提供候选，格距和起点一起验证

一次二维实数 FFT 复用于检测和诊断。图像频谱谷、边缘投影 FFT、可见边缘间距提出有限候选，并显式比较半倍和两倍尺度。

每个候选先估计起点，再计算：

```text
0.48 × 单位边缘间距支持
+ 0.42 × 扣除偶然对齐后的边缘解释度
+ 0.10 × 边缘投影周期强度
```

分数在一维证据上计算；不为每个候选重建整张图。单位间距支持抑制无依据的过细网格，缺失的同色边界不要求补齐。相关证据不能当成独立概率相乘。

默认接近方形；只有两个方向都存在强规则证据、格距比不超过 1.12 时，才允许自动选择轻微不同的横纵间距。显式 pixel-size 可指定矩形格子。

仅排名前三的候选尝试局部边缘吸附，每步范围为格距的 ±24%。强规则网格不做这项调整。递推偏移可能累积；不支持任意二维弯曲或透视。

### 3. robust 改为“中心优先，有证据才拒绝”

每格最多取 5×5 个内部点，并检查中心附近的 3×3 源像素：

- 中心有至少两个相似邻近样本支持时，直接保留中心值，不再平均整个多数颜色簇。
- 孤立中心异常与多数颜色明显不同、且没有邻近支持时，才回退到可见样本中值。
- 连续细线、2×2 高光等有局部支持的中心细节可以保留；不统一删除孤立输出像素。
- 预乘颜色与 alpha 联合用于一致性检查，完全透明像素的隐藏 RGB 不参与判断。
- 很小的源格子不做中心异常剔除，避免把原生细节当成噪点。

中心邻域检查允许八邻域局部支持，但不做连通分量合并。没有全局四/八连通拓扑修复。

默认 alpha 跟随采样：源 alpha=253 就保留 253，不因为主体接近不透明便强制改成 255；透明中心也不会因偏在格子边上的黑线而变成整格黑块。coverage 模式仍可用于面积平均对照，二值化只由显式 binary 参数触发。

### 4. 边缘覆盖、回退和量化

源图全幅覆盖；小于 0.2 格距的边缘残片并入相邻格。名义格距可非整数，最终切线通过 NumPy rint 舍入到整数源边界，每格保存实际半开区间。target-size 的精确数量优先。

没有可靠证据时保留原尺寸，输出低置信度提示和诊断，不交互阻塞。启发式置信分数不能解释为正确概率。

指定 colors 时只对可见低分辨率 RGB 做 Pillow median-cut，透明区域不占无意义的颜色预算。调色板太小仍可能损失高光。当前快速实现不做复杂全图重建评分、全局结构优化或神经网络推理。

## Python API

```python
from pixelperfect import Config, pixelize, save_result

result = pixelize("input/ritsu.png", Config())
print(result.image.size, result.grid, result.confidence, result.timings)
save_result(result, "output/ritsu.png")  # 同时生成诊断
```

pixelize 本身不写文件；返回原生图片、实际切割线、启发式置信分数、诊断与耗时。Config.scale 只影响导出尺寸。

## 测试：仅保留两个入口

```text
src/
  compare_programs.py         唯一的 Plus / main 对比脚本
  tests/test_pixelperfect.py  唯一的普通 pytest 文件
  pixelperfect/              核心算法，本次没有修改
```

原来的 pattern 生成器、benchmark、独立评估/报告脚本和对应产物已删除。普通回归合并到一个文件，覆盖网格、取色、透明度、输入输出、CLI 和对比输入一致性。其小数组与临时图片只用于功能回归，不生成图案评测数据集。

### 普通回归

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

pytest 是测试依赖，不是运行图片处理所必需的依赖。

### 直接对比真实图片

```powershell
python src/compare_programs.py
python src/compare_programs.py -i .\input\hollow-knight-sprite.png
```

不带 -i 时处理 input 中的实际图片。对比脚本：

1. 先修正原图 EXIF 方向，按 alpha 合成到纯白背景，保存一张供两者共同使用的 RGB PNG，不缩放输入。
2. **Plus**：启动现有 `python pixelperfect.py -i 共同白底文件 -o 结果路径`。
3. **main**：导入原项目公开入口，调用 `get_perfect_pixel(rgb)`，传入共同白底文件对应的相同 RGB 像素，所有算法参数采用原代码默认值。
4. 网格大小、起点和切割线全部由各自代码决定；脚本不提供外部格距、目标尺寸或调参配置。
5. 对比图使用纯白背景，左右两列使用相同的整数最近邻倍数。不会对算法输出执行去光晕、抠图或描边。

不能直接丢弃原图 alpha：完全透明像素可能存有黑色或带光晕的隐藏 RGB，丢弃 alpha 会将它们变成可见颜色。这里用 `alpha × RGB + (1-alpha) × 白色` 处理，两者接收同一份可见颜色。半透明像素正常混合，不强制变实。本对比评估白底 RGB 输入，不评估透明通道恢复；直接运行 Plus 处理原始 RGBA 时仍保留其正常透明流程。

这里比较的是同一白底输入经过各程序正常流程的结果。白底合成会改变边缘证据，因此网格可能与直接运行原始 RGBA 时不同。当前环境没有 OpenCV，main 公开入口会按原项目规则回退 NumPy；实际后端记录在图上和 JSON 中。有 OpenCV 时，后端也由原项目自己选择。

输出：

- [comparison.png](output/comparison.png)：**左列 perfectPixel-main，右列 PixelPerfect Plus**。
- [comparison.json](output/comparison.json)：调用命令、后端、返回尺寸、Plus 原程序给出的实际格线，以及错误信息。
- [comparison_native](output/comparison_native)：input_white/ 共同白底输入、main/ 和 plus/ 原生 PNG、每张图的 8 倍对比图；Plus 同时保留正常诊断包。JSON 记录共同输入路径及 RGB 像素 SHA256。

真实 AI 图片没有已知原始网格，不额外编造分数或准确率。原生输出保持各程序生成的尺寸，显示放大只作用于对比图。

## 目录与局限

### src/pixelperfect 模块说明

当前共 10 个 Python 文件、约 909 行（含空行和注释）。这些模块全部有调用用途，没有可以直接删除的闲置模块。本次只调整目录名与引用路径，未改动核心代码。

| 文件 | 作用 | 保留与简化建议 |
|---|---|---|
| `__init__.py` | 导出 Config、pixelize、PixelizeResult、save_result 和版本号 | 保留，只有 7 行，是 Python API 入口。 |
| `__main__.py` | 命令行参数、默认输出目录、错误处理、运行摘要 | 保留，支持命令行和安装后的入口。参数解析与 Config 校验各有职责，不宜一起删除。 |
| `config.py` | 统一保存参数，验证范围、类型与互斥约束 | 保留，CLI 和 API 共用，避免无效参数进入算法。 |
| `pipeline.py` | 串联读取、特征、网格、取色、量化；处理回退、计时和结果对象 | 保留，是 pixelize 的实现。可将回退结果构造抽成小函数，减少分支负担。 |
| `image_io.py` | 读取路径/Pillow/数组，EXIF 修正，处理 alpha，导出 PNG 和诊断 | 保留。Pillow 输入分支中 EXIF 转置执行两次，可在保留现有测试的前提下合并成一次。 |
| `features.py` | RGB/alpha 方向梯度、投影、二维 FFT、插值曲率证据 | 保留，是网格检测和诊断的共同输入。不能只保留灰度边缘，否则可能漏掉颜色边界。 |
| `grid.py` | 候选间距、半倍/两倍比较、起点搜索、评分、有限漂移和插值模型选择 | 保留，是决定格子大小的核心。可复用重复构造的规则格线及候选起点评分，不能直接删候选验证分支。 |
| `sampling.py` | center/median/robust 取色，中心邻域支持检查，alpha 策略和格子置信度 | 保留。可拆出 alpha 应用的小函数；仅需运行摘要的不透明比例统计可按需计算，保留 API 诊断兼容性。 |
| `palette.py` | 指定 --colors 时对可见颜色做无抖动量化 | 默认不执行，但由 pipeline 导入。只有 22 行，可合并到 sampling.py；直接删除会破坏导入和 --colors 功能，合并本身不会明显提速。 |
| `diagnostics.py` | 输出 FFT、边缘、网格、投影、曲率图及 JSON/TXT 信息 | 按当前每次生成诊断的要求必须保留。可清理未使用参数并复用投影 FFT。 |

可优先简化的具体位置：

- `diagnostics.write_debug` 的 `original` 参数没有被函数使用，当前调用也不传它；可移除，但需确认外部调用兼容性。
- `grid._axis` 返回字典中的 `trough` 没有下游读取；可移除这个返回字段。局部 trough 的计算仍用于产生候选，不能连同计算一起删除。
- `grid._detect_grid` 在判断 warped 时对同一组参数重复调用 `make_lines`；可缓存规则切割线。
- `grid._axis` 已算过投影 FFT，`diagnostics.write_debug` 又为作图重算；可复用中间结果，不过一维 FFT 的节省预计有限，需测量确认。
- `image_io.load_image` 的 Pillow 分支重复修正 EXIF，合并能省一次复制。需要保持现有内存 EXIF 测试通过。

`PixelizeResult.cell_confidence` 虽然没有参与最终网格选择，也没有独立导出置信度图，仍属于 API 返回信息；它不是完全无用的变量。`source_near_opaque_fraction` 目前只做诊断，已经不用于强制边缘变实。

文件合并主要减少文件数量，不等于算法提速。优先避免重复数组计算和图像复制；FFT、边缘识别、候选验证以及稳健取色都应保留。若只想减少一个小模块，可考虑合并 palette.py，当前没有必要把全部实现挤进一个文件。

业务目录为 input、output、src。运行依赖仍只有 NumPy 和 Pillow；测试另需 pytest。`src/pixelperfect` 是当前实现，`src/tests/test_pixelperfect.py` 是唯一的普通测试文件，`src/compare_programs.py` 是唯一的程序对比脚本。

仍不能保证任意 AI 图有唯一原始网格。重复纹理、严重模糊、任意二维漂移和偏离中心的极小细节仍可能导致误判。当前插值分支只检验受限的线性插值证据。请结合实际网格叠加图判断。
