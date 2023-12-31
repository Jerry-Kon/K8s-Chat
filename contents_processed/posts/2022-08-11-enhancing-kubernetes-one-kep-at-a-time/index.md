---
layout: blog
title: "逐个 KEP 地增强 Kubernetes"
date: 2022-08-11
slug: enhancing-kubernetes-one-kep-at-a-time
---

**作者：** Ryler Hockenbury（Mastercard）

你是否知道 Kubernetes v1.24 有
[46 个增强特性](https://kubernetes.io/zh-cn/blog/2022/05/03/kubernetes-1-24-release-announcement/)？
在为期 4 个月的发布周期内包含了大量新特性。
Kubernetes 发布团队协调发布的后勤工作，从修复测试问题到发布更新的文档。他们需要完成成吨的工作，但发布团队总是能按期交付。

发布团队由大约 30 人组成，分布在六个子团队：Bug Triage、CI Signal、Enhancements、Release Notes、Communications 和 Docs。
每个子团队负责管理发布的一个组件。这篇博文将重点介绍增强子团队的角色以及你如何能够参与其中。

## 增强子团队是什么？

好问题。我们稍后会讨论这个问题，但首先让我们谈谈 Kubernetes 中是如何管理功能特性的。

每个新特性都需要一个 [Kubernetes 增强提案](https://github.com/kubernetes/enhancements/blob/master/keps/README.md)，
简称为 KEP。KEP 是一些小型结构化设计文档，提供了一种提出和协调新特性的方法。
KEP 作者描述其提案动机、设计理念（和替代方案）、风险和测试，然后社区成员会提供反馈以达成共识。

你可以通过 [Kubernetes/enhancements 仓库](https://github.com/kubernetes/enhancements)的拉取请求（PR）工作流来提交和更新 KEP。
每个功能特性始于 Alpha 阶段，随着不断成熟，经由毕业流程进入 Beta 和 Stable 阶段。
这里有一个很酷的 KEP 例子，是关于 [Windows Server 上的特权容器支持](https://github.com/kubernetes/enhancements/blob/master/keps/sig-windows/1981-windows-privileged-container-support/kep.yaml)。
这个 KEP 在 Kubernetes v1.22 中作为 Alpha 引入，并在 v1.23 中进入 Beta 阶段。

现在回到上一个问题：增强子团队如何协调每个版本的 KEP 生命周期跟踪。
每个 KEP 都必须满足一组清晰具体的要求，才能被纳入一个发布版本中。
增强子团队负责验证每个 KEP 的要求并跟踪其状态。

在一个发行版本启动时，各个 [Kubernetes 特别兴趣小组](https://github.com/kubernetes/community/blob/master/sig-list.md) (SIG)
会提交各自的增强特性以进入某版本发布。通常一个版本最初可能有 60 到 90 个增强特性。随后，许多增强特性会被过滤掉。
这是因为有些不完全符合 KEP 要求，而另一些还未完成代码的实现。最初选择加入的 KEP 中大约有 60% - 70% 将进入最终发布。

## 增强子团队做什么？

这是另一个很好的问题，切中了要点！增强特性的团队在每个版本中会涉及两个重要的里程碑：增强特性冻结和代码冻结。

#### 增强特性冻结

增强特性冻结是一个 KEP 按序完成增强特性并纳入一个发布版本的最后期限。
这是一个质量门控，用于强制对齐与 KEP 维护和更新相关的事项。
最值得注意的要求是
(1) [生产就绪审查](https://github.com/kubernetes/community/blob/master/sig-architecture/production-readiness.md)(PRR)
和 (2) 附带完整测试计划和毕业标准的 [KEP 文件](https://github.com/kubernetes/enhancements/tree/master/keps/NNNN-kep-template)。

增强子团队通过在 Github 上对 KEP 问题发表评论与每位 KEP 作者进行沟通。
作为第一步，子团队成员将检查 KEP 状态并确认其是否符合要求。
KEP 在满足要求后被标记为已被跟踪（Tracked）；否则，它会被认为有风险。
如果在增强特性冻结生效时 KEP 仍然存在风险，该 KEP 将被从发布版本中移除。

在发布周期的这个阶段，增强子团队通常是最繁忙的，因为他们要梳理大量的 KEP，可能需要反复审查每个 KEP 才能验证某个 KEP 是否满足要求。

#### 代码冻结

代码冻结是从代码上实现所有增强特性的最后期限。
如果某增强特性的代码需要更改或更新，则必须在这个时间节点完成所有代码实现、代码审查和代码合并工作。
版本发布的最后三个工作专注于稳定代码库：修复测试问题，解决各种回归并准备文档。而在此之前，所有代码必须就位。

增强子团队将验证某增强特性相关的所有 PR 均已合并到 [Kubernetes 代码库](https://github.com/kubernetes/kubernetes) (k/k)。
在此期间，子团队将联系 KEP 作者以了解哪些 PR 是 KEP 的一部分，检查这些 PR 是否已合并，然后更新 KEP 的状态。
如果在代码冻结的最后期限之前这些代码还未全部合并，该增强特性将从发布版本中移除。

## 我如何才能参与发布团队？

很高兴你提出这个问题。
最直接的方式就是申请成为一名[发布团队影子](https://github.com/kubernetes/sig-release/blob/master/release-team/shadows.md)。
影子角色是一个见习职位，旨在帮助个人在发布团队中担任领导职位做好准备。许多影子角色是非技术性的，且不需要事先对 Kubernetes 代码库做出贡献。

Kubernetes 每年发布 3 个版本，每个版本大约有 25 个影子，发布团队总是需要愿意做出贡献的人。
在每个发布周期之前，发布团队都会为影子计划打开申请渠道。当申请渠道上线时，
会公布在 [Kubernetes 开发邮件清单](https://groups.google.com/a/kubernetes.io/g/dev)中。
你可以订阅该列表中的通知（或定期查看！），以了解申请渠道何时开通。该公告通常会在 4 月中旬、7 月中旬和 12 月中旬发布，
或者在每个版本开始前大约一个月时发布。

## 我怎样才能找到更多信息？

如果你对所有 Kubernetes 发布子团队的详情感到好奇，
请查阅[角色手册](https://github.com/kubernetes/sig-release/tree/master/release-team/role-handbooks)。
这些手册记录了每个子团队的后勤工作，包括每周对子团队活动的细分任务。这是更好地了解每个团队的绝佳参考。

你还可以查看与发布相关的 Kubernetes slack 频道，特别是 #release、#sig-release 和 #sig-arch。
这些频道围绕发布的许多方面进行了讨论和更新。
