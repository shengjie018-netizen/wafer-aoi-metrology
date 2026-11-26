# 半导体 AOI 视觉算法实战学习指南（先跑通 → 再倒推手写）

这份指南专为你（自动化 + CV 背景，冲刺半导体量检测/AOI 核心视觉算法实习）设计。
**核心原则**：不要一行行死背代码，而是按照 **“跑通看现象 → 读懂数据流 → 删掉核心函数自己手撕 → 跑单测验证精度 → 用面试官视角复盘”** 的 5 步闭环来吃透整个仓库。

---

## 一、7 周倒推学习路线（每周约 15 小时）

| 周次 | 核心模块 | 阅读文件 | 必须完成的动手任务 |
| :--- | :--- | :--- | :--- |
| **Week 1** | 现代 C++ 工程骨架与图像容器 | `CMakeLists.txt`, `image.hpp`, `image.cpp` | 亲手编译跑通 `wafer_aoi_tests` 和 `wafer_aoi_demo`；搞懂 `Image<T>` 连续内存布局、`row_ptr` 缓存友好访问、`std::clamp`、PGM/PPM 格式 |
| **Week 2** | 经典底层视觉算子 | `image_ops.hpp`, `image_ops.cpp` | 吃透**可分离高斯卷积（$O(2k)$ vs $O(k^2)$）**、Sobel 归一化、双线性插值、Otsu 大津法、**两遍扫描 + 并查集（Union-Find）连通域标记** → 完成【手撕挑战 1 & 2】 |
| **Week 3** | 金字塔梯度形状匹配 | `shape_matcher.hpp`, `shape_matcher.cpp` | 理解为何梯度单位向量点积抗光照变化、模板特征为什么在 `train()` 里做 `std::shuffle`（配合贪心早停）、OpenMP 多线程并行的粒度设计 → 完成【手撕挑战 3】 |
| **Week 4** | 亚像素边缘与鲁棒几何拟合 | `subpixel_metrology.hpp`, `subpixel_metrology.cpp` | 推导沿梯度法向 3 点抛物线亚像素公式、PCA 总最小二乘直线拟合、Kåsa + Gauss-Newton 几何圆拟合、30 次 GR&R $3\sigma$ 重复性含义 → 完成【手撕挑战 4】 |
| **Week 5** | 2D-FFT 与亚像素相位相关 | `d2d_inspector.cpp`（前半部分） | 吃透 Cooley-Tukey 蝶形变换、Hann 窗去频谱泄漏、正则化互功率谱（为何 `eps` 不能设为 `1e-12`）、**Guizar-Sicairos 矩阵乘上采样 DFT** → 完成【手撕挑战 5】 |
| **Week 6** | 频域陷波与 D2D 缺陷检出 | `d2d_inspector.cpp`（后半部分） | 理解径向频环统计自动陷波原理、**梯度自适应局部容差阈值**如何压低边缘过杀（Overkill）、**PCA 二阶矩长宽比**为何比水平外接框更适合斜向划痕 → 完成【手撕挑战 6】 |
| **Week 7** | 并发流水线与面试演练 | `thread_pool.hpp`, `main_demo.cpp` | 手写一遍 `ThreadPool`（`mutex + condition_variable + packaged_task`）与 `BoundedQueue`，脱离文档口述一遍下方 15 道面试高频题 |

---

## 二、6 个“删掉重写”手撕挑战（最有效的掌握方式）

把对应函数体清空（或新建 `git checkout -b practice/my-impl` 分支），凭理解自己重写，直到 `wafer_aoi_tests` 重新 **12/12 PASS**！

1. **挑战 1（`src/core/image_ops.cpp` -> `gaussian_blur` & `bilinear_sample`）**
   * 要求：先水平一维卷积再垂直一维卷积，处理镜像边界 `reflect_index`。
   * 验收：单测全过，思考面试追问：“为什么可分离卷积快？缓存命中率在垂直 pass 时怎么优化？”
2. **挑战 2（`src/core/image_ops.cpp` -> `connected_components`）**
   * 要求：手写 8 邻域两遍扫描 + 带路径压缩的 Union-Find，并统计面积、质心与外接框。
   * 验收：`CCL finds 2 components` 与 `CCL areas exact` 通过。
3. **挑战 3（`src/algo/shape_matcher.cpp` -> `score_pose`）**
   * 要求：实现单位梯度点积打分 + 极性反转选项 + 每 16 个特征检查一次精确剪枝与贪心早停。
   * 验收：`ShapeBasedMatcher pose accuracy` 通过，且 `main_demo` 耗时保持在 75 ms 以内。
4. **挑战 4（`src/algo/subpixel_metrology.cpp` -> `extract_subpixel_edges` & `fit_line_ransac`）**
   * 要求：实现法向 NMS + 3 点二次插值亚像素坐标，以及 RANSAC 内点筛选 + TLS 精修。
   * 验收：BGA 圆和 IC 引脚间距误差均 `< 0.10 px`。
5. **挑战 5（`src/algo/d2d_inspector.cpp` -> `phase_correlate_subpixel`）**
   * 要求：自己实现 Hann 窗 → 2D FFT → 正则化互功率谱 → IFFT 找整像素峰 → Guizar-Sicairos 上采样 DFT 精修。
   * 验收：`(6.35, -4.72)` 测试位移误差 `< 0.05 px`。
   * 💡 *对比实验*：你可以把 Guizar-Sicairos 临时换回普通 3 点抛物线插值，亲眼观察误差为何从 `0.02 px` 恶化到 `0.20 px`——这个实验细节在面试中极有说服力！
6. **挑战 6（`src/algo/d2d_inspector.cpp` -> `inspect_die_to_die` 的边缘自适应阈值与 PCA 划痕分类）**
   * 要求：重写 `local_thresh = base + k * grad_mag` 以及基于二阶中心矩 $\mu_{20}, \mu_{02}, \mu_{11}$ 的特征值长宽比公式。
   * 验收：`main_demo` 精确检出 `1 Scratch + 2 Particle + 1 Missing/LargeDefect` 且 0 过杀。

---

## 三、半导体量检测 / AOI 面试高频 15 问（直接对照本仓库作答）

### A. 形状匹配与定位
1. **为什么半导体 AOI 定位普遍用梯度形状匹配（Shape-Based），而不用灰度相关性（NCC/SSD）？**
   * 答：产线存在光源衰减、晶圆膜厚色差、明暗场反射率变化甚至局部对比度反转。梯度单位向量点积 $\mathbf{n}_t \cdot \mathbf{n}_s$ 对全局线性亮度偏移与增益变化完全不变；如果取绝对值 $|\mathbf{n}_t \cdot \mathbf{n}_s|$，甚至能抗极性反转（`ignore_polarity`）。
2. **模板匹配搜索 $360^\circ$ 很慢，你做了哪些加速？**
   * 答：四级加速——① 高斯金字塔由粗到细（Coarse-to-Fine）；② 离线预计算粗层旋转姿态表；③ 随机打乱特征点顺序 + 贪心早停（Greediness），99% 背景点算十几个特征就剪枝退出；④ OpenMP 在姿态维度动态调度并行（实测加速 `5.01×`）。
3. **如何获得亚像素和亚角度定位精度？**
   * 答：在 Level 0 整数峰值周围取 3 邻域（$x\pm 1, y\pm 1, \theta\pm \Delta\theta$）做 1D 二次抛物线顶点插值 $\delta = \frac{s_{-1} - s_{+1}}{2(s_{-1} - 2s_0 + s_{+1})}$。

### B. 亚像素量测与几何拟合
4. **普通 Canny 边缘精度只能到 1 像素，你的亚像素边缘是怎么做到 $0.05\text{ px}$ 以内的？**
   * 答：在梯度幅值图上沿连续单位法向 $(n_x, n_y)$ 双线性采样 $m(-1), m(0), m(+1)$，先做方向非极大值抑制找脊线，再对三点拟合抛物线求极大值偏移 $\delta \in [-0.5, 0.5]$，输出 $(x + \delta n_x, y + \delta n_y)$。
5. **为什么直线拟合用 PCA（总最小二乘 TLS）而不是普通最小二乘 $y = kx + b$？**
   * 答：普通最小二乘只最小化竖直误差 $\sum (y_i - kx_i - b)^2$，当 IC 引脚是竖直线（$k \to \infty$）时直接数值崩溃；TLS 最小化点到直线的正交几何距离，对任意角度完全旋转不变。
6. **圆拟合为什么先用 Kåsa 代数拟合再用 Gauss-Newton 精修？为什么还要套 RANSAC？**
   * 答：BGA 焊球常带毛刺/连锡等外点（Outlier），直接最小二乘会被外点严重拉偏，所以必须先用 RANSAC 剔除毛刺外点；对内点先用中心化 Kåsa 代数法快速给初值，再用 Gauss-Newton 直接最小化几何径向误差 $\sum (\|p_i - c\| - r)^2$，消除代数拟合在弧段不完整时的偏置。
7. **工业量测里说的 GR&R 和 $3\sigma$ 重复性精度是什么意思？**
   * 答：对同一标准件在真实传感器噪声下连续重复测量 30 次，计算测量结果标准差的 3 倍（$3\sigma$），它衡量算法自身的稳定度上限。本项目中 30 帧噪声测试下圆半径 $3\sigma = 0.0049\text{ px}$。

### C. 频域处理与 Die-to-Die 缺陷检测
8. **为什么相位相关（Phase Correlation）前要加 Hann 窗？**
   * 答：2D-FFT 隐含周期延拓假设，图像上下/左右边界的灰度跳变会在频域引入强烈的十字形频谱泄漏，干扰相位峰；加 2D Hann 窗让边界平滑衰减到 0，显著提升峰值信噪比。
9. **相位相关求亚像素位移时，为什么不能简单对 IFFT 峰值做 3 点抛物线拟合？**
   * 答：理想相位相关的空间峰值是 Sinc 核（加 Hann 窗后呈展宽主瓣），而非抛物线，3 点抛物线会产生显著向整数收缩的系统误差（本项目实测约 `0.2 px`）；采用 **Guizar-Sicairos (2008) 矩阵乘上采样 DFT** 在粗峰附近做局部精确带限插值，可将误差压到 `< 0.02 px`，且时间复杂度仅需两次可分离 1D 矩阵乘。
10. **互功率谱归一化时为什么分母要加正则化项 `eps = 1e-4 * max_mag`？**
    * 答：高频（接近 Nyquist）处的真实图像信号能量极小，如果直接除以 `|F| + 1e-12`，会把高频相机噪声和插值混叠放大成单位幅值，污染相位峰；加入比例正则化项可平滑抑制低信噪比高频。
11. **FFT 频域自动陷波滤波时，为什么按“径向频环”而不是全局均值标准差找周期尖峰？**
    * 答：自然与晶圆图像的功率谱本身随频率半径 $r$ 呈 $1/f$ 衰减。全局阈值会导致低频全被误判、高频谐波全漏判；按半径分环统计 $(\mu_r, \sigma_r)$ 再算 Z-score，才能跨频段精准抓出周期电路的各阶谐波。
12. **Die-to-Die 差分检测中，高对比度电路边缘最容易产生误报（过杀 Overkill），你是怎么解决的？**
    * 答：因为即使对齐误差只有 `0.02 px`，在梯度极陡的电路边缘处依然会产生一阶泰勒残差 $\Delta I \approx \|\nabla I\| \cdot \|\Delta x\|$。本仓库引入了**梯度自适应局部阈值** $T(x,y) = T_{\text{base}} + k \cdot \|\nabla I_{\text{golden}}(x,y)\|$，在平坦区保持高灵敏度抓微小缺陷，在电路边缘放宽容差，实测把高敏模式下的边缘过杀直接降为 0。
13. **为什么不能用外接矩形的长宽比（`w/h`）来判断划痕（Scratch）？**
    * 答：一条 $45^\circ$ 斜向细长划痕的水平外接框接近正方形（`w/h ≈ 1.4`），会被误判为块状缺陷。计算连通域的二阶中心矩协方差矩阵特征值比 $\sqrt{\lambda_1 / \lambda_2}$（PCA 长短轴比）才是真正**旋转不变**的细长率指标。

### D. C++ 工程与并发优化
14. **你的 `ThreadPool` 是怎么避免忙等（CPU 空转）和虚假唤醒（Spurious Wakeup）的？**
    * 答：工作线程通过 `std::condition_variable::wait(lock, predicate)` 阻塞挂起，不占 CPU；传入 lambda 谓词 `[this]{ return stop_ || !tasks_.empty(); }` 循环检查条件，彻底防止虚假唤醒；析构时加锁置 `stop_ = true` 后 `notify_all()` 再逐一 `join()`，确保已入队任务安全执行完毕不丢帧。
15. **如果要把这套算法继续加速 5~10 倍放到真实 $4096\times 4096$ 高速晶圆产线上，你下一步怎么做？**
    * 答：① 内存池复用（预分配每帧缓冲，杜绝循环内 `malloc/free`）；② 关键内层循环（梯度点积、Sobel、残差计算）用 **AVX2 SIMD** 向量化；③ 把 2D-FFT（`cuFFT`）、双线性对齐与金字塔匹配移植到 **CUDA GPU**，配合 `Pinned Memory + 多 CUDA Stream` 隐藏 H2D/D2H 拷贝延迟（这也正是你研二上升级到项目 2 / CUDA 阶段的演进方向！）。
