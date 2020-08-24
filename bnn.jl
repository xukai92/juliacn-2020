### A Pluto.jl notebook ###
# v0.11.8

using Markdown
using InteractiveUtils

# ╔═╡ 15745074-e4b8-11ea-3411-db4855a833f8
using DrWatson; quickactivate(@__DIR__)

# ╔═╡ 5ea2b6fa-e4b8-11ea-2a25-17153795dc1c
begin
	using Random, RDatasets, MLDataUtils, Distributions, Flux
	using Turing, ReverseDiff; Turing.setadbackend(:reversediff)
	using Plots; theme(:bright; size=(600, 300))
end

# ╔═╡ 798395e8-e4b8-11ea-1615-45cda6d2547c
md"""
## Demo 2: Bayesian Neural Nets 贝叶斯神经网络

基于 Turing.jl 的两个教程改编
- https://turing.ml/dev/tutorials/3-bayesnn/
- https://turing.ml/dev/tutorials/8-multinomiallogisticregression/
"""

# ╔═╡ f41393d0-e4b8-11ea-03da-db12c88fee22
md"""
在这个教程里我们考虑使用经典的 IRIS 数据集，是一个常用的分类实验数据集。

IRIS 数据集里包含了一些花的特征，包括
- 花萼长度 sepal length
- 花萼宽度 sepal width
- 花瓣长度 petal length
- 花瓣宽度 petal width

而我们想要预测的标签就是花的种类，包括
- 山鸢尾 Iris Setosa
- 杂色鸢尾 Iris Versicolor
- 维吉尼亚鸢尾 Iris Virginica

注：翻译来自百度百科：https://baike.baidu.com/item/IRIS/4061453
"""

# ╔═╡ e505f776-e4d0-11ea-0d11-bfe02c0857ea
md"""
我们首先用 RDatasets.jl 读取数据集。
数据集会以 `DataFrame` 的形式被读取。
"""

# ╔═╡ 5aff404e-e4b9-11ea-1b1c-5b17fd0dd55e
data_raw = RDatasets.dataset("datasets", "iris"); first(data_raw, 5)

# ╔═╡ 759e2d6c-e4b8-11ea-107a-c1ee5218c8c7
md"""
### Data cleaning

作为准备，我们需要
- 把花的种类变成数字，即
    - `"Setosa"` -> `1`
    - `"Versicolor"` -> `2`
    - `"Virginica"` -> `3`
- 把花的种类做 one-hot encoding，即
    - `1` -> `[1, 0, 0]`
    - `2` -> `[0, 1, 0]`
    - `3` -> `[0, 0, 1]`
- 把 `DataFrame` 里的特征和目标转换成 `Matrix` 方便训练和推断
- 把数据集分成 train 和 test 两部分
- 最后放在一个 `NamedTuple` 里方便使用
"""

# ╔═╡ eedbd9b0-e4b8-11ea-1ce0-3fb9a0edbbcf
data = let data = copy(data_raw)
	Random.seed!(1)
	
	# Encode the species columns
	data[!,:C1] = [r.Species == "setosa"     ? 1.0 : 0.0 for r in eachrow(data)]
	data[!,:C2] = [r.Species == "versicolor" ? 1.0 : 0.0 for r in eachrow(data)]
	data[!,:C3] = [r.Species == "virginica"  ? 1.0 : 0.0 for r in eachrow(data)]

	# Delete the old column
	select!(data, Not([:Species]))
	
	# `DataFrame` to `Matrix`
	features = [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]
	target = [:C1, :C2, :C3]
	X, Y = Matrix(data[:,features])', Matrix(data[:,target])'
	
	# Shuffle
	X, Y = shuffleobs((X, Y))
	
	# Split train and test
	(X_train, Y_train), (X_test, Y_test) = splitobs((X, Y); at=0.5)
	X_train, Y_train, X_test, Y_test = 
		Matrix(X_train), Matrix(Y_train), Matrix(X_test), Matrix(Y_test)
	y_train, y_test = Flux.onecold(Y_train), Flux.onecold(Y_test)
	
	# Put in a `NamedTuple`
	@ntuple(X_train, Y_train, y_train, X_test, Y_test, y_test)
end

# ╔═╡ e7c4ab86-e4b9-11ea-185b-cdb98f88b778
md"""
### 神经网络 via Flux.jl

数据准备好了，我先来看看用 Flux.jl 训练一个普通的神经网络效果如何。

这里我们准备一个神经网络
- Input layer 大小为 4 和特征数目一致
- Hidden layer 大小为 20
- Output layer 大小为 3 和种类数目一致
- Activation function 为 `relu`
"""

# ╔═╡ bb52b846-e4e0-11ea-3d90-cdfd0f10bf67
make_nn(h=20) = Chain(Dense(4, h, relu),Dense(h, h, relu), Dense(h, 3))

# ╔═╡ e9069d8e-e4d1-11ea-36f3-c318e3d02ed7
md"""
并且定义一下 loss 函数，这里使用常见的交叉熵 cross entropy。
"""

# ╔═╡ 87b3da0a-e4d2-11ea-3098-bb619eee0493
md"""
训练 1,000 次并且可视化一下训练过程中 loss 的变化。
"""

# ╔═╡ 295ffc24-e4bc-11ea-3f61-250e3bdada9e
begin
	Random.seed!(1)
	
	nn = make_nn(20)
	
	loss(x, y) = Flux.logitcrossentropy(nn(x), y)
	
	let ps = Flux.params(nn), data_train = (data.X_train, data.Y_train), lr = 2f-3
		
		losses_train = [loss(data_train...)]
		for i_epoch in 1:100
			Flux.train!(loss, ps, [data_train], Descent(lr))
			push!(losses_train, loss(data_train...))
		end
		
		plot(losses_train; label=nothing, xlabel="Epoch", ylabel="Loss")
	end
end

# ╔═╡ a3cfe706-e4d2-11ea-1805-6b595985a3a5
md"""
最后在测试集上看一下预测的准确率。
"""

# ╔═╡ 8736fff0-e4bf-11ea-2aca-abfda7137a44
let P_test = softmax(nn(data.X_test))
	
	mean(Flux.onecold(P_test) .== data.y_test)
end

# ╔═╡ 63c9b0f4-e4bb-11ea-1e3c-734358a3369e
md"""
### NNs $\to$ Bayesian NNs via Turing.jl

现在我们来用 Turing.jl 把这个神经网络变成**贝叶斯神经网络**。

首先，什么是贝叶斯神经网络？

PS: 贝叶斯神经网络和贝叶斯网络 Bayes nets 不是一个东西
"""

# ╔═╡ 390eb738-e4d4-11ea-3842-bbffb433cc64
md"""
#### 神经网络分类器的概率诠释 Probabilistic interpretation of neural classifiers

神经网络分类器 + cross entropy loss function 等价于以下过程

`p = softmax(NN(x, weights))`

`y ~ Categorical(p)`

其中

- `x` 是特征
- `softmax` 是 softmax 函数，能把一个向量变成一个概率向量 (加起来是 1)
- `p` 是每个类别的概率
- `Categorical(p)` 是类别分布 categorical distribution
- `y` 是标签 1, 2 或 3

以上部分代表了我们之前提到的 $ p(\mathcal{D} \mid \theta) $ 其中 $\theta$ 就是我们神经网络的权重，所以我们剩下的任务就是选择一个先验。

这里我们选择最简单的先验 $ p(\theta_i) = \mathcal{N}(0, \sigma^2) $。

我们可以想象以下在这里的标签的生成过程

1. 从先验采样神经网络的权重
2. 用神经网络算出概率
3. 从类别分布采样标签

"""

# ╔═╡ eb667a64-e4d5-11ea-2452-41da7e85a45b
md"""
在用 Turing.jl 建模之前，我们做一些准备工作，其中

- `reconstruct` 是用来给定任意 `weights` 重建神经网络的函数，由 Flux.jl 提供
- `LogCategorical` 是一个把概率存在 log domain 里的 `Categorical`，更稳定一些
    - 任何用户通过 Distributions.jl 接口自定义的分布都能够在 Turing.jl 里使用
"""

# ╔═╡ 6469172a-e4c5-11ea-269b-07ab35eb52ac
begin
	_weights, reconstruct = Flux.destructure(nn)
	
	struct LogCategorical{T} <: DiscreteUnivariateDistribution
		logp::T
	end
	
	Distributions.logpdf(d::LogCategorical, x::Int) = d.logp[x]
end

# ╔═╡ 9242cd06-e4d6-11ea-29ae-4b1d652eb3fe
md"""
现在我们就可以用 Turing.jl 实现贝叶斯神经网络了，模型在做什么一目了然。
"""

# ╔═╡ 50e7ecbc-e4c5-11ea-0df5-43978cefd73f
@model function bayes_nn(X, y, n_weights=length(_weights))
	weights ~ filldist(Normal(0, 5), n_weights)
	bnn = reconstruct(weights)
	logP = logsoftmax(bnn(X))
	for i in 1:length(y)
		y[i] ~ LogCategorical(logP[:,i])
	end
end

# ╔═╡ b02c4e32-e4d6-11ea-0961-7b03d1950350
md"""
接下来我们调用 `sample` 来从后验采样，并且可视化 joint probability 随着采样过程的变化。
"""

# ╔═╡ f6467368-e4c5-11ea-16c0-ab42bf07da1d
begin
	Random.seed!(1)
	
	chain = let data_train = (data.X_train, Flux.onecold(data.Y_train))
		sample(bayes_nn(data_train...), HMC(1e-2, 4), 200; init_theta=_weights)
	end
	
	let log_density = vec(only(get(chain, :log_density)))
		plot(log_density; label=nothing, xlabel="Sample", ylabel="Log-joint")
	end
end

# ╔═╡ 336d31ca-e4d8-11ea-3a08-750699ecabca
md"""
我们来看一下在后验下的预测情况。

在使用 MCMC 的样本的时候
- 通常会丢掉一开始的样本 `n_warmup`
- 每个几个样本使用（稀释） `n_thinning`
"""

# ╔═╡ 462d9268-e4e8-11ea-2066-53f737ff74a4
md"""
这里我们获得了一些样本神经网络的输出，放在 `logitP_list` 里。
"""

# ╔═╡ 2bccb446-e4d7-11ea-2de1-cb9705394ed5
logitP_list = let samples = Array(chain[:weights]), n_warmup = 100, n_thinning = 10
	
	weights_list = [samples[i,:] for i in n_warmup+1:n_thinning:size(samples, 1)]
	logitP_list = []
	for weights in weights_list
		bnn = reconstruct(weights)
		logitP = bnn(data.X_test)
		push!(logitP_list, logitP)
	end
	logitP_list
end

# ╔═╡ fa8b4e18-e4e2-11ea-0283-11afa19524a3
md"""
平均概率的准确率：每个神经网络都求预测概率，然后取平均，最后判断对错。
"""

# ╔═╡ 7037a20e-e4e2-11ea-2872-2b23dc9b050a
mean(Flux.onecold(mean(softmax.(logitP_list))) .== Flux.onecold(data.Y_test))

# ╔═╡ c97d6a7a-e4e4-11ea-0163-6f3555c2b28e
md"""
似乎比普通的神经网络好，是因为我们多做了一些计算吗？

让我们把神经网络多训练100步看看。
"""

# ╔═╡ 4050a96c-e4d8-11ea-27d0-370a05813d94
md"""
### 贝叶斯神经网络和普通神经网络的比较

1. 先验和正则：防止过拟合
    - 高斯先验等价于 L2 正则
    - 大的模型会显著
2. 考虑模型不确定性 p(θ | D) - 更鲁棒 robust
3. 优化 vs 采样
4. 贝叶斯并不总是会更好
"""

# ╔═╡ 9ecdfc0a-e4d9-11ea-2f26-c97bae79502b
md"""
最后：可视化模型不确定性

- 这里模型的不确定性是在样本中模型输出概率的标准差
- "确定的错误“ vs ”不确定的错误"
- "蒙对的"

知道什么时候自己不知道很重要
"""

# ╔═╡ 031d86ba-e4d9-11ea-1c84-27ec38bf9173
let uncertainty = std(softmax.(logitP_list))
	
	P_test = softmax(nn(data.X_test))
	idx_mistake = findall(i -> !i, Flux.onecold(P_test) .== data.y_test)
	
	p1 = bar(uncertainty[1,:]; label="Setosa")
	vline!(idx_mistake; label="Mistake", style=:dash)
	p2 = bar(uncertainty[2,:]; label="Versicolor")
	vline!(idx_mistake; label="Mistake", style=:dash)
	p3 = bar(uncertainty[3,:]; label="Virginica")
	vline!(idx_mistake; label="Mistake", style=:dash)
	plot(p1, p2, p3; layout=@layout([p1; p2; p3]), size=(600, 450))
end

# ╔═╡ e956defa-e4e7-11ea-35de-7711f250e3b1
md"""
跟预测熵 predictive entropy 比较一下。
- 这里不确定性就是神经网络最后输出概率 `p` 的熵

$-\sum_i p_i \log p_i$

- 通常用于衡量非贝叶斯神经网络的不确定性
- 熵越大不确定性越大
"""

# ╔═╡ 6b4ab680-e4e7-11ea-0b15-2bec0accf9e2
let P_test = softmax(nn(data.X_test))
	
	uncertainty = -vec(sum(P_test .* log.(P_test); dims=1))
	idx_mistake = findall(i -> !i, Flux.onecold(P_test) .== data.y_test)
	
	bar(uncertainty; label="Predictive entropy", size=(600, 200))
	vline!(idx_mistake; label="Mistake", style=:dash)
	ylims!(0, -log(1/3))
end

# ╔═╡ de8a32a0-e4e8-11ea-36ea-5f10d6d7af4f
md"""
可以看到很多数据的 predictive entropy 都差不多，而且跟最后对错的相关性没那么高。
"""

# ╔═╡ Cell order:
# ╠═15745074-e4b8-11ea-3411-db4855a833f8
# ╟─798395e8-e4b8-11ea-1615-45cda6d2547c
# ╠═5ea2b6fa-e4b8-11ea-2a25-17153795dc1c
# ╟─f41393d0-e4b8-11ea-03da-db12c88fee22
# ╟─e505f776-e4d0-11ea-0d11-bfe02c0857ea
# ╠═5aff404e-e4b9-11ea-1b1c-5b17fd0dd55e
# ╟─759e2d6c-e4b8-11ea-107a-c1ee5218c8c7
# ╠═eedbd9b0-e4b8-11ea-1ce0-3fb9a0edbbcf
# ╟─e7c4ab86-e4b9-11ea-185b-cdb98f88b778
# ╠═bb52b846-e4e0-11ea-3d90-cdfd0f10bf67
# ╟─e9069d8e-e4d1-11ea-36f3-c318e3d02ed7
# ╟─87b3da0a-e4d2-11ea-3098-bb619eee0493
# ╠═295ffc24-e4bc-11ea-3f61-250e3bdada9e
# ╟─a3cfe706-e4d2-11ea-1805-6b595985a3a5
# ╠═8736fff0-e4bf-11ea-2aca-abfda7137a44
# ╟─63c9b0f4-e4bb-11ea-1e3c-734358a3369e
# ╟─390eb738-e4d4-11ea-3842-bbffb433cc64
# ╟─eb667a64-e4d5-11ea-2452-41da7e85a45b
# ╠═6469172a-e4c5-11ea-269b-07ab35eb52ac
# ╟─9242cd06-e4d6-11ea-29ae-4b1d652eb3fe
# ╠═50e7ecbc-e4c5-11ea-0df5-43978cefd73f
# ╟─b02c4e32-e4d6-11ea-0961-7b03d1950350
# ╠═f6467368-e4c5-11ea-16c0-ab42bf07da1d
# ╟─336d31ca-e4d8-11ea-3a08-750699ecabca
# ╟─462d9268-e4e8-11ea-2066-53f737ff74a4
# ╠═2bccb446-e4d7-11ea-2de1-cb9705394ed5
# ╟─fa8b4e18-e4e2-11ea-0283-11afa19524a3
# ╠═7037a20e-e4e2-11ea-2872-2b23dc9b050a
# ╟─c97d6a7a-e4e4-11ea-0163-6f3555c2b28e
# ╟─4050a96c-e4d8-11ea-27d0-370a05813d94
# ╟─9ecdfc0a-e4d9-11ea-2f26-c97bae79502b
# ╠═031d86ba-e4d9-11ea-1c84-27ec38bf9173
# ╟─e956defa-e4e7-11ea-35de-7711f250e3b1
# ╠═6b4ab680-e4e7-11ea-0b15-2bec0accf9e2
# ╟─de8a32a0-e4e8-11ea-36ea-5f10d6d7af4f
