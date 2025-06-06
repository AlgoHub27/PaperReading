### **MSE（均方误差）失真**

#### **1. 计算公式**
MSE（Mean Squared Error）定义为原始信号 $X$ 与重建信号 $\hat{X}$ 之间像素级差异的平方均值：
$\Delta_{\text{MSE}}(X, \hat{X}) = \mathbb{E}\left[ \|X - \hat{X}\|^2 \right]$
- 对离散信号（如图像），计算公式为：
  $\text{MSE} = \frac{1}{N} \sum_{i=1}^N (x_i - \hat{x}_i)^2$
  其中 $N$ 为像素总数，$x_i$ 和 $\hat{x}_i$ 分别为原始和重建像素值。

#### **2. 含义**
- **数学意义**：量化信号重建的像素级精度，值越小表示重建信号与原始信号越接近。
- **局限性**：MSE 与人类主观感知相关性较弱。例如，模糊化可能降低MSE但导致感知质量下降（见图6右）。

#### **3. 论文中的应用**
- 作为基础失真度量（如第4.1节实验），用于对比传统压缩与感知优化压缩的性能。
- 定理2中证明：完美感知质量（$P=0$）时，MSE允许最多增加2倍（PSNR降低3dB）。

---

### **SSIM（结构相似性）失真**

#### **1. 计算公式**
SSIM（Structural Similarity Index）通过比较亮度（Luminance）、对比度（Contrast）和结构（Structure）衡量两信号的相似性：
$\text{SSIM}(X, \hat{X}) = \frac{(2\mu_X \mu_{\hat{X}} + C_1)(2\sigma_{X\hat{X}} + C_2)}{(\mu_X^2 + \mu_{\hat{X}}^2 + C_1)(\sigma_X^2 + \sigma_{\hat{X}}^2 + C_2)}$
其中：
- $\mu_X, \mu_{\hat{X}}$：$X$ 和 $\hat{X}$ 的均值；
- $\sigma_X^2, \sigma_{\hat{X}}^2$：方差；
- $\sigma_{X\hat{X}}$：协方差；
- $C_1, C_2$：避免除零的常数。

**作为失真度量**：  
SSIM 取值范围为 $[-1, 1]$（值越大相似性越高），但需转换为非负失真指标：
$\Delta_{\text{SSIM}}(X, \hat{X}) = 1 - \text{SSIM}(X, \hat{X})$

#### **2. 含义**
- **数学意义**：捕捉人类视觉系统对亮度、对比度和结构的敏感度，值越小表示感知相似性越高。
- **优势**：比MSE更符合主观质量评价（如论文第4.2节实验）。

#### **3. 论文中的应用**
- 论文指出，即使使用SSIM作为失真度量，仍无法避免感知质量与码率/失真的权衡（见图8）。
- **关键结论**：仅优化SSIM仍会导致解码分布偏离源分布（即感知质量下降），需显式约束分布散度（如Wasserstein距离）。

---

### **MSE与SSIM的对比**
| **指标** | MSE                          | SSIM                          |
|----------|------------------------------|-------------------------------|
| **焦点** | 像素级精度                   | 结构相似性                    |
| **范围** | $[0, +\infty)$             | $[0, 2]$（转换后）          |
| **感知关联性** | 弱（易与人类评价脱节） | 强（更贴近主观质量）          |
| **计算复杂度** | 低                         | 高（需计算均值、方差、协方差）|
| **论文角色** | 基础失真度量，用于理论分析 | 先进失真度量，验证权衡普适性  |

---

### **总结**
- **MSE**：简单易计算，但忽略感知特性，易导致模糊化重建。
- **SSIM**：更贴近人类视觉，但需转换为失真度量，且无法单独解决感知-失真矛盾。
- **论文启示**：无论使用何种失真度量，均需显式优化分布散度以提升感知质量（如结合GAN）。 