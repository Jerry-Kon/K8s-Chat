---
title: 进阶贡献
slug: advanced
content_type: concept
weight: 100
---


如果你已经了解如何[贡献新内容](/zh-cn/docs/contribute/new-content/)和
[评阅他人工作](/zh-cn/docs/contribute/review/reviewing-prs/)，并准备了解更多贡献的途径，
请阅读此文。你需要使用 Git 命令行工具和其他工具做这些工作。


## 提出改进建议   {#propose-improvements}

SIG Docs 的[成员](/zh-cn/docs/contribute/participate/roles-and-responsibilities/#members)可以提出改进建议。

在对 Kubernetes 文档贡献了一段时间后，你可能会对[样式指南](/zh-cn/docs/contribute/style/style-guide/)、
[内容指南](/zh-cn/docs/contribute/style/content-guide/)、用于构建文档的工具链、网站样式、
评审和合并 PR 的流程或者文档的其他方面产生改进的想法。
为了尽可能透明化，这些提议都需要在 SIG Docs 会议或
[kubernetes-sig-docs 邮件列表](https://groups.google.com/forum/#!forum/kubernetes-sig-docs)上讨论。
此外，在提出全面的改进之前，这些讨论能帮助我们了解有关“当前工作如何运作”和“以往的决定是为何做出”的背景。
想了解文档的当前运作方式，最快的途径是咨询 [kubernetes.slack.com](https://kubernetes.slack.com)
中的 `#sig-docs` 聊天群组。

在进行了讨论并且 SIG 就期望的结果达成一致之后，你就能以最合理的方式处理改进建议了。
例如，样式指南或网站功能的更新可能涉及 PR 的新增，而与文档测试相关的更改可能涉及 sig-testing。

## 为 Kubernetes 版本发布协调文档工作   {#coordinate-docs-for-a-kubernetes-release}

SIG Docs 的[批准人（Approver）](/zh-cn/docs/contribute/participate/roles-and-responsibilities/#approvers)
可以为 Kubernetes 版本发布协调文档工作。

每一个 Kubernetes 版本都是由参与 sig-release 的 SIG（特别兴趣小组）的一个团队协调的。
指定版本的发布团队中还包括总体发布牵头人，以及来自 sig-testing 的代表等。
要了解更多关于 Kubernetes 版本发布的流程，请参考
[https://github.com/kubernetes/sig-release](https://github.com/kubernetes/sig-release)。

SIG Docs 团队的代表需要为一个指定的版本协调以下工作：

- 通过特性跟踪表来监视新功能特性或现有功能特性的修改。
  如果版本的某个功能特性的文档没有为发布做好准备，那么该功能特性不允许进入发布版本。
- 定期参加 sig-release 会议并汇报文档的发布状态。
- 评审和修改由负责实现某功能特性的 SIG 起草的功能特性文档。
- 合入版本发布相关的 PR，并为对应发布版本维护 Git 特性分支。
- 指导那些想学习并有意愿担当该角色的 SIG Docs 贡献者。这就是我们常说的“实习”。
- 发布版本的组件发布时，相关的文档更新也需要发布。

协调一个版本发布通常需要 3-4 个月的时间投入，该任务由 SIG Docs 批准人轮流承担。

## 担任新的贡献者大使   {#serve-as-a-new-contributor-ambassador}

SIG Docs [批准人（Approver）](/zh-cn/docs/contribute/participate/roles-and-responsibilities/#approvers)
可以担任新的贡献者大使。

新的贡献者大使欢迎 SIG-Docs 的新贡献者，对新贡献者的 PR 提出建议，
以及在前几份 PR 提交中指导新贡献者。

新的贡献者大使的职责包括：

- 监听 [Kubernetes #sig-docs 频道](https://kubernetes.slack.com) 上新贡献者的 Issue。
- 与 PR 管理者合作为新参与者寻找[合适的第一个 issue](https://kubernetes.dev/docs/guide/help-wanted/#good-first-issue)。
- 通过前几个 PR 指导新贡献者为文档存储库作贡献。
- 帮助新的贡献者创建成为 Kubernetes 成员所需的更复杂的 PR。
- [为贡献者提供保荐](#sponsor-a-new-contributor)，使其成为 Kubernetes 成员。
- 每月召开一次会议，帮助和指导新的贡献者。

当前新贡献者大使将在每次 SIG 文档会议上以及 [Kubernetes #sig-docs 频道](https://kubernetes.slack.com)中宣布。

## 为新的贡献者提供保荐 {#sponsor-a-new-contributor}

SIG Docs 的[评审人（Reviewer）](/zh-cn/docs/contribute/participate/roles-and-responsibilities/#reviewers)
可以为新的贡献者提供保荐。

新的贡献者针对一个或多个 Kubernetes 项目仓库成功提交了 5 个实质性 PR 之后，
就有资格申请 Kubernetes 组织的[成员身份](/zh-cn/docs/contribute/participate/roles-and-responsibilities/#members)。
贡献者的成员资格需要同时得到两位评审人的保荐。

新的文档贡献者可以通过咨询 [Kubernetes Slack 实例](https://kubernetes.slack.com)
上的 #sig-docs 频道或者 [SIG Docs 邮件列表](https://groups.google.com/forum/#!forum/kubernetes-sig-docs)
来请求评审者保荐。如果你对申请人的工作充满信心，你自愿保荐他们。
当他们提交成员资格申请时，回复 “+1” 并详细说明为什么你认为申请人适合加入 Kubernetes 组织。

## 担任 SIG 联合主席   {#sponsor-a-new-contributor}

SIG Docs [成员（Member）](/zh-cn/docs/contribute/participate/roles-and-responsibilities/#members)
可以担任 SIG Docs 的联合主席。

### 前提条件   {#prerequisites}

Kubernetes 成员必须满足以下要求才能成为联合主席：

- 理解 SIG Docs 工作流程和工具：git、Hugo、本地化、博客子项目
- 理解其他 Kubernetes SIG 和仓库会如何影响 SIG Docs 工作流程，包括：
  [k/org 中的团队](https://github.com/kubernetes/org/blob/master/config/kubernetes/sig-docs/teams.yaml)、
  [k/community 中的流程](https://github.com/kubernetes/community/tree/master/sig-docs)、
  [k/test-infra](https://github.com/kubernetes/test-infra/) 中的插件、
  [SIG Architecture](https://github.com/kubernetes/community/tree/main/sig-architecture) 中的角色。
  此外，了解 [Kubernetes 文档发布流程](/zh-cn/docs/contribute/advanced/#coordinate-docs-for-a-kubernetes-release)的工作原理。
- 由 SIG Docs 社区直接或通过惰性共识批准。
- 在至少 6 个月的时段内，确保每周至少投入 5 个小时（通常更多）

### 职责范围   {#responsibilities}

联合主席的角色提供以下服务：

- 拓展贡献者规模
- 处理流程和政策
- 安排时间和召开会议
- 安排 PR 管理员
- 在 Kubernetes 社区中提出文档倡议
- 确保文档在 Kubernetes 发布周期中符合预期
- 让 SIG Docs 专注于有效的优先事项

职责范围包括：

- 保持 SIG Docs 专注于通过出色的文档最大限度地提高开发人员的满意度
- 以身作则，践行[社区行为准则](https://github.com/cncf/foundation/blob/main/code-of-conduct.md)，
  并要求 SIG 成员对自身行为负责
- 通过更新贡献指南，为 SIG 学习并设置最佳实践
- 安排和举行 SIG 会议：每周状态更新，每季度回顾/计划会议以及其他需要的会议
- 在 KubeCon 活动和其他会议上安排和负责文档工作
- 与 {{< glossary_tooltip text="CNCF" term_id="cncf" >}} 及其尊贵合作伙伴
  （包括 Google、Oracle、Azure、IBM 和华为）一起以 SIG Docs 的身份招募和宣传
- 负责 SIG 正常运行

### 召开高效的会议   {#running-effective-meetings}

为了安排和召开高效的会议，这些指南说明了如何做、怎样做以及原因。

**坚持[社区行为准则](https://github.com/cncf/foundation/blob/main/code-of-conduct.md)**：

- 相互尊重地、包容地进行讨论。

**设定明确的议程**：

- 设定清晰的主题议程
- 提前发布议程

对于每周一次的会议，请将前一周的笔记复制并粘贴到笔记的“过去的会议”部分中

**通过协作，完成准确的记录**：

- 记录会议讨论
- 考虑委派笔记记录员的角色

**清晰准确地分配执行项目**：

- 记录操作项，分配给它的人员以及预期的完成日期

**根据需要来进行协调**：

- 如果讨论偏离议程，请让参与者重新关注当前主题
- 为不同的讨论风格留出空间，同时保持讨论重点并尊重人们的时间

**尊重大家的时间**:

按时开始和结束会议

**有效利用 Zoom**：

- 熟悉 [Kubernetes Zoom 指南](https://github.com/kubernetes/community/blob/master/communication/zoom-guidelines.md)
- 输入主持人密钥登录时声明主持人角色

<img src="/images/docs/contribute/claim-host.png" width="75%" alt="声明 Zoom 主持人角色" />

### 录制 Zoom 会议   {#recording-meetings-on-zoom}

准备开始录制时，请单击“录制到云”。

准备停止录制时，请单击“停止”。

视频会自动上传到 YouTube。

### SIG 联合主席 (Emeritus) 离职  {#offboarding-a-sig-cochair}

参见 [k/community/sig-docs/offboarding.md](https://github.com/kubernetes/community/blob/master/sig-docs/offboarding.md)
