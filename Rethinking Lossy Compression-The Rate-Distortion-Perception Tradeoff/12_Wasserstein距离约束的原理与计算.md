### **Wasserstein距离约束的原理与计算**
- CSDN博客补充：[Wasserstein距离（最优传输距离：-CSDN博客](https://blog.csdn.net/weixin_44862361/article/details/125505769)
---

#### **1. Wasserstein距离的基本原理**
**Wasserstein距离**（Earth Mover's Distance, EMD）衡量两个概率分布 $p_X$（源分布）和 $p_{\hat{X}}$（重建分布）之间的差异。其核心思想是计算将 $p_X$ “搬运”到 $p_{\hat{X}}$ 所需的最小“工作量”（成本），同时考虑分布的几何结构。

**数学定义**：  
对于 $p$-Wasserstein距离（ $p \geq 1$ ）：
$W_p(p_X, p_{\hat{X}}) = \left( \inf_{\gamma \in \Gamma(p_X, p_{\hat{X}})} \int \|x - \hat{x}\|^p \, d\gamma(x, \hat{x}) \right)^{1/p}$
- $\Gamma(p_X, p_{\hat{X}})$：所有联合分布 $\gamma(x, \hat{x})$ 的集合，其边缘分布分别为 $p_X$ 和 $p_{\hat{X}}$；
- $\|x - \hat{x}\|^p$：将 $x$ 传输到 $\hat{x}$ 的成本（通常取欧氏距离的 $p$ 次方）。

**直观解释**：  
Wasserstein距离不仅考虑分布的重叠程度，还考虑分布间样本的几何关系，因此对非重叠分布的差异更敏感。

---

#### **2. 论文中的Wasserstein约束原理**
在论文中，Wasserstein距离被用作感知质量约束，目标是最小化 $W_1(p_X, p_{\hat{X}})$。具体实现方式：
1. **优化目标**：在率失真优化中，同时约束：
   $\mathbb{E}[\Delta(X, \hat{X})] \leq D \quad \text{and} \quad W_1(p_X, p_{\hat{X}}) \leq P$
2. **GAN框架**：利用Wasserstein GAN（WGAN）的判别器（Critic）近似计算Wasserstein距离，并通过对抗训练优化生成器（解码器）。

---

#### **3. 具体计算方法**
论文中采用 **Wasserstein GAN** 的优化策略，具体步骤如下：

##### **步骤1：定义判别器（Critic）网络**
- **输入**：原始样本 $X$ 或重建样本 $\hat{X}$；
- **输出**：标量值，表示样本属于自然分布 $p_X$ 的置信度；
- **约束**：判别器需是 **1-Lipschitz连续函数**（通过梯度惩罚实现）。

##### **步骤2：计算Wasserstein距离**
Wasserstein距离的对偶形式为：

   ${W_1(p_X, p_{\hat{X}}) = \sup_{\|f\|_{L} \leq 1} \left( \mathbb{E}_{X \sim p_X}[f(X)] - \mathbb{E}_{\hat{X} \sim p_{\hat{X}}}[f(\hat{X})] \right)}$                
   
- $f$ 是1-Lipschitz函数，由判别器网络近似。

##### **步骤3：联合优化编码器-解码器**
- **损失函数**：
  $\mathcal{L} = \mathbb{E}[\Delta(X, \hat{X})] + \lambda \cdot W_1(p_X, p_{\hat{X}})$
- **训练流程**：
  1. 更新判别器：最大化 $\mathbb{E}[f(X)] - \mathbb{E}[f(\hat{X})]$；
  2. 更新生成器（解码器）：最小化 $\mathcal{L}$；
  3. 交替迭代直至收敛。

---

#### **4. 算法图例**
```plaintext
输入数据 X ──→ 编码器 ──→ 量化 ──→ 解码器（生成器） ──→ 重建数据 X̂
                              │
                              └── 噪声注入（提升多样性）
                                    │
                                    ↓
判别器（Critic）───→ 计算 Wasserstein距离 W₁(p_X, p_X̂)
                                    │
                                    └── 反馈梯度，优化编码器/解码器
```

#### **5. 示例与论文结果**

- **MNIST实验**（图6）：
    
    - **无约束（λ=0）**：重建图像模糊，感知质量差（高Wasserstein距离）。
        
    - **Wasserstein约束（λ>0）**：重建图像更自然，但需接受更高失真（如MSE增加）。
        
- **结论**：Wasserstein距离约束能有效提升感知质量，但需权衡码率或失真。
    

---

### **总结**

- **原理**：Wasserstein距离通过最优传输理论量化分布差异，结合GAN框架实现高效优化。
    
- **计算**：利用判别器网络近似对偶形式，通过梯度惩罚保证Lipschitz连续性。
    
- **效果**：在低码率下显著改善感知质量，但需权衡失真（如MSE增加）。