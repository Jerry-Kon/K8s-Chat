---
title: Kubernetes API
content_type: concept
weight: 40
description: >
  Kubernetes API 使你可以查询和操纵 Kubernetes 中对象的状态。
  Kubernetes 控制平面的核心是 API 服务器和它暴露的 HTTP API。
  用户、集群的不同部分以及外部组件都通过 API 服务器相互通信。
card:
  name: concepts
  weight: 30
---


Kubernetes {{< glossary_tooltip text="控制面" term_id="control-plane" >}}的核心是
{{< glossary_tooltip text="API 服务器" term_id="kube-apiserver" >}}。
API 服务器负责提供 HTTP API，以供用户、集群中的不同部分和集群外部组件相互通信。

Kubernetes API 使你可以在 Kubernetes 中查询和操纵 API 对象
（例如 Pod、Namespace、ConfigMap 和 Event）的状态。

大部分操作都可以通过 [kubectl](/zh-cn/docs/reference/kubectl/) 命令行接口或类似
[kubeadm](/zh-cn/docs/reference/setup-tools/kubeadm/) 这类命令行工具来执行，
这些工具在背后也是调用 API。不过，你也可以使用 REST 调用来访问这些 API。

如果你正在编写程序来访问 Kubernetes API，
可以考虑使用[客户端库](/zh-cn/docs/reference/using-api/client-libraries/)之一。


## OpenAPI 规范     {#api-specification}

完整的 API 细节是用 [OpenAPI](https://www.openapis.org/) 来表述的。

### OpenAPI v2

Kubernetes API 服务器通过 `/openapi/v2` 端点提供聚合的 OpenAPI v2 规范。
你可以按照下表所给的请求头部，指定响应的格式：

<table>
  <caption style="display:none">OpenAPI v2 查询请求的合法头部值</caption>
  <thead>
     <tr>
        <th>头部</th>
        <th style="min-width: 50%;">可选值</th>
        <th>说明</th>
     </tr>
  </thead>
  <tbody>
     <tr>
        <td><code>Accept-Encoding</code></td>
        <td><code>gzip</code></td>
        <td><em>不指定此头部也是可以的</em></td>
     </tr>
     <tr>
        <td rowspan="3"><code>Accept</code></td>
        <td><code>application/com.github.proto-openapi.spec.v2@v1.0+protobuf</code></td>
        <td><em>主要用于集群内部</em></td>
     </tr>
     <tr>
        <td><code>application/json</code></td>
        <td><em>默认值</em></td>
     </tr>
     <tr>
        <td><code>*</code></td>
        <td><em>提供</em><code>application/json</code></td>
     </tr>
  </tbody>
</table>

Kubernetes 为 API 实现了一种基于 Protobuf 的序列化格式，主要用于集群内部通信。
关于此格式的详细信息，可参考
[Kubernetes Protobuf 序列化](https://git.k8s.io/design-proposals-archive/api-machinery/protobuf.md)设计提案。
每种模式对应的接口描述语言（IDL）位于定义 API 对象的 Go 包中。

### OpenAPI v3

{{< feature-state state="stable"  for_k8s_version="v1.27" >}}

Kubernetes 支持将其 API 的描述以 OpenAPI v3 形式发布。

发现端点 `/openapi/v3` 被提供用来查看可用的所有组、版本列表。
此列表仅返回 JSON。这些组、版本以下面的格式提供：

```yaml
{
    "paths": {
        ...,
        "api/v1": {
            "serverRelativeURL": "/openapi/v3/api/v1?hash=CC0E9BFD992D8C59AEC98A1E2336F899E8318D3CF4C68944C3DEC640AF5AB52D864AC50DAA8D145B3494F75FA3CFF939FCBDDA431DAD3CA79738B297795818CF"
        },
        "apis/admissionregistration.k8s.io/v1": {
            "serverRelativeURL": "/openapi/v3/apis/admissionregistration.k8s.io/v1?hash=E19CC93A116982CE5422FC42B590A8AFAD92CDE9AE4D59B5CAAD568F083AD07946E6CB5817531680BCE6E215C16973CD39003B0425F3477CFD854E89A9DB6597"
        },
        ....
    }
}
```

为了改进客户端缓存，相对的 URL 会指向不可变的 OpenAPI 描述。
为了此目的，API 服务器也会设置正确的 HTTP 缓存标头
（`Expires` 为未来 1 年，和 `Cache-Control` 为 `immutable`）。
当一个过时的 URL 被使用时，API 服务器会返回一个指向最新 URL 的重定向。

Kubernetes API 服务器会在端点 `/openapi/v3/apis/<group>/<version>?hash=<hash>`
发布一个 Kubernetes 组版本的 OpenAPI v3 规范。

请参阅下表了解可接受的请求头部。

<table>
  <thead>
     <tr>
     </tr>
  </thead>
  <tbody>
     <tr>
        <td><code>Accept-Encoding</code></td>
        <td><code>gzip</code></td>
     </tr>
     <tr>
        <td rowspan="3"><code>Accept</code></td>
        <td><code>application/com.github.proto-openapi.spec.v3@v1.0+protobuf</code></td>
     </tr>
     <tr>
        <td><code>application/json</code></td>
     </tr>
     <tr>
        <td><code>*</code></td>
     </tr>
  </tbody>
</table>

`k8s.io/client-go/openapi3` 包中提供了获取 OpenAPI v3 的 Golang 实现。

## 持久化   {#persistence}

Kubernetes 通过将序列化状态的对象写入到 {{< glossary_tooltip term_id="etcd" >}} 中完成存储操作。

## API 发现   {#api-discovery}

集群支持的所有组版本列表被发布在 `/api` 和 `/apis` 端点。
每个组版本还会通过 `/apis/<group>/<version>`
（例如 `/apis/rbac.authorization.k8s.io/v1alpha1`）广播支持的资源列表。
这些端点由 kubectl 用于获取集群支持的资源列表。

### 聚合发现   {#aggregated-discovery}

{{< feature-state state="beta"  for_k8s_version="v1.27" >}}

Kubernetes 对聚合发现提供 Beta 支持，通过两个端点（`/api` 和 `/apis`）
发布集群支持的所有资源，而不是每个组版本都需要一个端点。
请求此端点显著减少了获取平均 Kubernetes 集群发现而发送的请求数量。
通过请求各自的端点并附带表明聚合发现资源
`Accept: application/json;v=v2beta1;g=apidiscovery.k8s.io;as=APIGroupDiscoveryList`
的 Accept 头部来进行访问。

该端点还支持 ETag 和 protobuf 编码。

## API 组和版本控制 {#api-groups-and-versioning}

为了更容易消除字段或重组资源的呈现方式，Kubernetes 支持多个 API 版本，每个版本位于不同的 API 路径，
例如 `/api/v1` 或 `/apis/rbac.authorization.k8s.io/v1alpha1`。

版本控制是在 API 级别而不是在资源或字段级别完成的，以确保 API 呈现出清晰、一致的系统资源和行为视图，
并能够控制对生命结束和/或实验性 API 的访问。

为了更容易演进和扩展其 API，Kubernetes 实现了 [API 组](/zh-cn/docs/reference/using-api/#api-groups)，
这些 API 组可以被[启用或禁用](/zh-cn/docs/reference/using-api/#enabling-or-disabling)。

API 资源通过其 API 组、资源类型、名字空间（用于名字空间作用域的资源）和名称来区分。
API 服务器透明地处理 API 版本之间的转换：所有不同的版本实际上都是相同持久化数据的呈现。
API 服务器可以通过多个 API 版本提供相同的底层数据。

例如，假设针对相同的资源有两个 API 版本：`v1` 和 `v1beta1`。
如果你最初使用其 API 的 `v1beta1` 版本创建了一个对象，
你稍后可以使用 `v1beta1` 或 `v1` API 版本来读取、更新或删除该对象，
直到 `v1beta1` 版本被废弃和移除为止。此后，你可以使用 `v1` API 继续访问和修改该对象。

### API 变更     {#api-changes}

任何成功的系统都要随着新的使用案例的出现和现有案例的变化来成长和变化。
为此，Kubernetes 已设计了 Kubernetes API 来持续变更和成长。
Kubernetes 项目的目标是 **不要** 给现有客户端带来兼容性问题，并在一定的时期内维持这种兼容性，
以便其他项目有机会作出适应性变更。

一般而言，新的 API 资源和新的资源字段可以被频繁地添加进来。
删除资源或者字段则要遵从
[API 废弃策略](/zh-cn/docs/reference/using-api/deprecation-policy/)。

Kubernetes 对维护达到正式发布（GA）阶段的官方 API 的兼容性有着很强的承诺，通常这一 API 版本为 `v1`。
此外，Kubernetes 保持与 Kubernetes 官方 API 的 **Beta** API 版本持久化数据的兼容性，
并确保在该功能特性已进入稳定期时数据可以通过 GA API 版本进行转换和访问。

如果你采用一个 Beta API 版本，一旦该 API 进阶，你将需要转换到后续的 Beta 或稳定的 API 版本。
执行此操作的最佳时间是 Beta API 处于弃用期，因为此时可以通过两个 API 版本同时访问那些对象。
一旦 Beta API 结束其弃用期并且不再提供服务，则必须使用替换的 API 版本。

{{< note >}}
尽管 Kubernetes 也努力为 **Alpha** API 版本维护兼容性，在有些场合兼容性是无法做到的。
如果你使用了任何 Alpha API 版本，需要在升级集群时查看 Kubernetes 发布说明，
如果 API 确实以不兼容的方式发生变更，则需要在升级之前删除所有现有的 Alpha 对象。
{{< /note >}}

关于 API 版本分级的定义细节，请参阅
[API 版本参考](/zh-cn/docs/reference/using-api/#api-versioning)页面。

## API 扩展  {#api-extension}

有两种途径来扩展 Kubernetes API：

1. 你可以使用[自定义资源](/zh-cn/docs/concepts/extend-kubernetes/api-extension/custom-resources/)来以声明式方式定义
   API 服务器如何提供你所选择的资源 API。
1. 你也可以选择实现自己的[聚合层](/zh-cn/docs/concepts/extend-kubernetes/api-extension/apiserver-aggregation/)来扩展
   Kubernetes API。

## {{% heading "whatsnext" %}}

- 了解如何通过添加你自己的
  [CustomResourceDefinition](/zh-cn/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/)
  来扩展 Kubernetes API。
- [控制 Kubernetes API 访问](/zh-cn/docs/concepts/security/controlling-access/)页面描述了集群如何针对
  API 访问管理身份认证和鉴权。
- 通过阅读 [API 参考](/zh-cn/docs/reference/kubernetes-api/)了解 API 端点、资源类型以及示例。
- 阅读 [API 变更（英文）](https://git.k8s.io/community/contributors/devel/sig-architecture/api_changes.md#readme)
  以了解什么是兼容性的变更以及如何变更 API。
