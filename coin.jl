### A Pluto.jl notebook ###
# v0.11.8

using Markdown
using InteractiveUtils

# ╔═╡ 842295c6-e3c3-11ea-1cb4-a5066a8c5c46
using DrWatson; quickactivate(@__DIR__)

# ╔═╡ f406b586-e3ee-11ea-1d63-894add7085c9
using Random, Turing, Plots, StatsPlots; theme(:bright; size=(400, 300))

# ╔═╡ b5646d60-e3f2-11ea-16aa-f7b1df7bb884
md"""
## Demo 1: 丢硬币
"""

# ╔═╡ 763835f4-e3f2-11ea-3def-31d64ac6496a
md"""
小刚跟小明打赌，猜他的硬币丢一百次能有几次是正面。

小明知道小刚有一枚“作弊”硬币：丢 100 次只有 25 次是正面。

小刚这次用的正是这枚硬币，**但是**小明不知道这次用的就是这枚硬币。

于是小明让小刚先丢 10 次硬币看看，结果如下。
- 0 代表反面
- 1 代表正面
"""

# ╔═╡ 3a439974-e4b7-11ea-22ff-4b5ccaa0c2a4
data = let N = 10
	Random.seed!(900)
	x = rand(Bernoulli(1 / 4), N)
	(x = x, h = sum(x .== 1), t = sum(x .== 0))
end

# ╔═╡ c90c698a-e3f2-11ea-0047-7537d36e5668
md"""
小明准备使用刚才提到的伯努利模型和一个贝塔先验来帮助自己判断。

$\theta \sim \mathcal{B}eta(2, 6)$ 
$x_i \sim \mathcal{B}er(\theta)\quad \text{for}\;i \in 1,\dots,N$
"""

# ╔═╡ 7a73fd6c-e3f0-11ea-2728-d526a502b56e
硬币先验 = Beta(2, 6)

# ╔═╡ bd220b8e-e3f0-11ea-15fc-fbe80370c5a9
# 众数和期望
mode(硬币先验), mean(硬币先验)

# ╔═╡ 3138cf80-e3f3-11ea-0d31-ede384782c79
md"""
因为小明之前就知道小刚有一枚作弊硬币，所以使用了很**强**的**先验** -- 在没有数据的情况下就觉得投出 1 的概率很小。

这是贝叶斯方法的一个特点：允许在建模过程中提供已知信息。
"""

# ╔═╡ 159e0562-e3f4-11ea-3f5d-cb2c7e82119b
md"""
小明知道贝塔分布是伯努利分布的共轭先验，所以通过 slides 里的公式可以得到后验是：
"""

# ╔═╡ d356c9dc-e3f0-11ea-2b9e-0526ed460521
硬币后验 = let h = data.h, t = data.t
	Beta(2 + h, 6 + t)
end

# ╔═╡ ef424660-e3f0-11ea-209a-d3920e7cba75
mode(硬币后验), mean(硬币后验)

# ╔═╡ 66b13448-e3fa-11ea-3531-678ed289fbc0
md"""
如果不用贝叶斯方法，光“数数”的话，会得到正面的概率是 0.4, 这个结果和最大似然估计 (maximam likelihood estimation, MLE) 一致。
"""

# ╔═╡ 665ab664-e3f4-11ea-00f9-1742f414d077
md"""
我们可以可视化看一下这两个分布分布长什么样。
"""

# ╔═╡ f74caf14-e3f0-11ea-1db1-b5f6b08fef8a
let p = plot()
	plot!(硬币先验; label="Prior")
	plot!(硬币后验; label="Posterior")
	vline!([0.4]; linestyle=:dash, label="MLE")
	xlabel!("θ")
end

# ╔═╡ 76a151b8-e3f4-11ea-381e-e1f1ac9d2e5f
md"""
如果小明不知道后验的公式，或者使用了一个没有解析解的先验-似然组合呢？

这时候 Turing.jl 就派上用处了!
"""

# ╔═╡ b2b34508-e3f4-11ea-1537-c577aa2a2dee
md"""
小明使用的硬币模型在 Turing.jl 中可以这样表示：
"""

# ╔═╡ 32712354-e3f1-11ea-193c-d7ab9f328ef4
@model function 硬币贝塔伯努利(x, α=2, β=6)
	θ ~ Beta(α, β)
	硬币 = Bernoulli(θ)
	for i in eachindex(x)
		x[i] ~ 硬币
	end
	return x
end

# ╔═╡ cf39e1f0-e3f4-11ea-1433-797e2f8b11e1
md"""
在这里
- 把 `@model` 放在函数定义前来声明模型
- `硬币贝塔伯努利` 是模型的名字
- 任何函数的参数会被视为数据，比如此处的 `x`
- 函数里的 `~` (tilde) 来表示某个变量服从某一个分布
    - `θ` 服从 `硬币先验 = Beta(2, 6)`
    - `x[i]` 服从 `Bernoulli(θ)`
    - 因为 `x` 是函数参数，所以被视为了数据。
    - 任何不是数据的变量都会被视为模型参数，如此处的 `θ`
"""

# ╔═╡ 6bbaa866-e3f5-11ea-36bb-9f6982695ecc
md"""
刚才我说过

> 概率编程框架=有概率语义的编程 (即建模) + 自动贝叶斯推断

现在我们已经完成来建模，接下来就可以“一键贝叶斯推断”了！
"""

# ╔═╡ 78fc2012-e3f1-11ea-21f6-fb4c3182d30d
chain = sample(硬币贝塔伯努利(data.x), HMC(0.1, 10), 5_000)

# ╔═╡ 9760141a-e3f5-11ea-110d-43a52620b037
md"""
在 Turing.jl 中，贝叶斯推断由 `sample` 函数调用 MCMC 算法完成
- `硬币贝塔伯努利(data.x)` 是我们的模型 + 数据
- `HMC(0.1, 10)` 是我们的 MCMC 算法 (Hamiltonian Monte Carlo)
- `1_000` 是我们要求采样的数目
"""

# ╔═╡ 12cd5d30-e3f6-11ea-3a6a-b1fe7ec4c5c1
md"""
现在我们来可视化以下看看自动推断和“手动推断”的结果是否一样：
"""

# ╔═╡ b5a5db8e-e3f1-11ea-23f9-0bb32a509cfb
let p = plot()
	plot!(硬币先验; label="Prior")
	plot!(硬币后验; label="Posterior (Analytical)")
	density!(Array(chain[:θ]); label="Posterior (MCMC)")
	xlabel!("θ")
end

# ╔═╡ 326a28ce-e3f6-11ea-38b3-9149528b7f59
md"""
可以看到，MCMC 采样的结果（绿色）和手算的结果（红色）是一致的！
"""

# ╔═╡ 3039c296-e3f9-11ea-2d12-e320d346ed16
md"""
最后，我们来用后验做一下预测，看看丢一百次硬币的话会有多少次正面。

$\int p(\theta \mid \mathcal{D}) f(\theta) \mathrm{d}\theta \quad\text{where}\quad f(\theta) = 100\theta$

我们可以用蒙特卡洛来计算积分：

$\frac{1}{N} \sum_i 100\theta_i \quad\text{where}\quad \theta_i \sim p(\theta \mid \mathcal{D})$
"""

# ╔═╡ 471ce07e-e3f9-11ea-2b7a-0731c5885f6c
let samples = Array(chain[:θ])
	sum([100θ for θ in samples]) / length(samples)
end

# ╔═╡ 24b99d32-e3fa-11ea-35c5-1324214e65bb
md"""
如果最大似然估计, 就会预测出现 40 次正面。
这里可以看到，因为加入了先验，在数据有限的情况下，小明的估计比没有先验会更靠近真实值 (25)。
"""

# ╔═╡ eeb5bf96-e5ef-11ea-0fdf-5f4495021bf7
md"""
### 关于 `return` 和 `missing`

在提问环节有人提问了 `return` 怎么用，当时没有把例子写完，现在在这里展示一个 `return` 和 `missing` 的例子。
"""

# ╔═╡ 2344987e-e5f0-11ea-0b90-d73304df0637
model_with_missing = 硬币贝塔伯努利(fill(missing, 5))

# ╔═╡ 505702fc-e5f0-11ea-254d-71207088208e
md"""
在这里，我已经在之前的模型定义中加入了 `return x`。

如果传入的 data 是 `missing` 或者 `Vector{Missing}`，Turing 就会对数据进行采样。
这里
- `x = fill(missing, 5)` 就是 `Vector{Missing}` 类型的
- `model_with_missing` 在这里其实就是一个仿真器 simulator，当被调用的时候就会随机产生数据。
"""

# ╔═╡ ad370a76-e5f0-11ea-22ae-83fc4caaf1b6
[model_with_missing() for _ in 1:3]

# ╔═╡ Cell order:
# ╠═842295c6-e3c3-11ea-1cb4-a5066a8c5c46
# ╟─b5646d60-e3f2-11ea-16aa-f7b1df7bb884
# ╠═f406b586-e3ee-11ea-1d63-894add7085c9
# ╟─763835f4-e3f2-11ea-3def-31d64ac6496a
# ╠═3a439974-e4b7-11ea-22ff-4b5ccaa0c2a4
# ╟─c90c698a-e3f2-11ea-0047-7537d36e5668
# ╠═7a73fd6c-e3f0-11ea-2728-d526a502b56e
# ╠═bd220b8e-e3f0-11ea-15fc-fbe80370c5a9
# ╟─3138cf80-e3f3-11ea-0d31-ede384782c79
# ╟─159e0562-e3f4-11ea-3f5d-cb2c7e82119b
# ╠═d356c9dc-e3f0-11ea-2b9e-0526ed460521
# ╠═ef424660-e3f0-11ea-209a-d3920e7cba75
# ╟─66b13448-e3fa-11ea-3531-678ed289fbc0
# ╟─665ab664-e3f4-11ea-00f9-1742f414d077
# ╠═f74caf14-e3f0-11ea-1db1-b5f6b08fef8a
# ╟─76a151b8-e3f4-11ea-381e-e1f1ac9d2e5f
# ╟─b2b34508-e3f4-11ea-1537-c577aa2a2dee
# ╠═32712354-e3f1-11ea-193c-d7ab9f328ef4
# ╟─cf39e1f0-e3f4-11ea-1433-797e2f8b11e1
# ╟─6bbaa866-e3f5-11ea-36bb-9f6982695ecc
# ╠═78fc2012-e3f1-11ea-21f6-fb4c3182d30d
# ╟─9760141a-e3f5-11ea-110d-43a52620b037
# ╟─12cd5d30-e3f6-11ea-3a6a-b1fe7ec4c5c1
# ╠═b5a5db8e-e3f1-11ea-23f9-0bb32a509cfb
# ╟─326a28ce-e3f6-11ea-38b3-9149528b7f59
# ╟─3039c296-e3f9-11ea-2d12-e320d346ed16
# ╠═471ce07e-e3f9-11ea-2b7a-0731c5885f6c
# ╟─24b99d32-e3fa-11ea-35c5-1324214e65bb
# ╟─eeb5bf96-e5ef-11ea-0fdf-5f4495021bf7
# ╠═2344987e-e5f0-11ea-0b90-d73304df0637
# ╟─505702fc-e5f0-11ea-254d-71207088208e
# ╠═ad370a76-e5f0-11ea-22ae-83fc4caaf1b6
