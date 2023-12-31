---
title: 使用配置文件对 Kubernetes 对象进行命令式管理
content_type: task
weight: 40
---

可以使用 `kubectl` 命令行工具以及用 YAML 或 JSON 编写的对象配置文件来创建、更新和删除 Kubernetes 对象。
本文档说明了如何使用配置文件定义和管理对象。

## {{% heading "prerequisites" %}}

安装 [`kubectl`](/zh-cn/docs/tasks/tools/) 。

{{< include "task-tutorial-prereqs.md" >}} {{< version-check >}}


## 权衡

`kubectl` 工具支持三种对象管理：

* 命令式命令
* 命令式对象配置
* 声明式对象配置

参看 [Kubernetes 对象管理](/zh-cn/docs/concepts/overview/working-with-objects/object-management/)
中关于每种对象管理的优缺点的讨论。

## 如何创建对象

你可以使用 `kubectl create -f` 从配置文件创建一个对象。
请参考 [kubernetes API 参考](/docs/reference/generated/kubernetes-api/{{< param "version" >}}/) 有关详细信息。

* `kubectl create -f <filename|url>`

## 如何更新对象

{{< warning >}}
使用 `replace` 命令更新对象会删除所有未在配置文件中指定的规范的某些部分。
不应将其规范由集群部分管理的对象使用，比如类型为 `LoadBalancer` 的服务，
其中 `externalIPs` 字段独立于配置文件进行管理。
必须将独立管理的字段复制到配置文件中，以防止 `replace` 删除它们。
{{< /warning >}}

你可以使用 `kubectl replace -f` 根据配置文件更新活动对象。 

* `kubectl replace -f <filename|url>`

## 如何删除对象

你可以使用 `kubectl delete -f` 删除配置文件中描述的对象。

* `kubectl delete -f <filename|url>`

{{< note >}}
如果配置文件在 `metadata` 节中设置了 `generateName` 字段而非 `name` 字段，
你无法使用 `kubectl delete -f <filename|url>` 来删除该对象。
你必须使用其他标志才能删除对象。例如：

```shell
kubectl delete <type> <name>
kubectl delete <type> -l <label>
```
{{< /note >}}

## 如何查看对象

你可以使用 `kubectl get -f` 查看有关配置文件中描述的对象的信息。

* `kubectl get -f <filename|url> -o yaml`

`-o yaml` 标志指定打印完整的对象配置。
使用 `kubectl get -h` 查看选项列表。

## 局限性

当完全定义每个对象的配置并将其记录在其配置文件中时，`create`、 `replace` 和`delete` 命令会很好的工作。
但是，当更新一个活动对象，并且更新没有合并到其配置文件中时，下一次执行 `replace` 时，更新将丢失。
如果控制器,例如 HorizontalPodAutoscaler ,直接对活动对象进行更新，则会发生这种情况。
这有一个例子：

1. 从配置文件创建一个对象。
1. 另一个源通过更改某些字段来更新对象。
1. 从配置文件中替换对象。在步骤2中所做的其他源的更改将丢失。

如果需要支持同一对象的多个编写器，则可以使用 `kubectl apply` 来管理该对象。

## 从 URL 创建和编辑对象而不保存配置

假设你具有对象配置文件的 URL。
你可以在创建对象之前使用 `kubectl create --edit` 对配置进行更改。
这对于指向可以由读者修改的配置文件的教程和任务特别有用。

```shell
kubectl create -f <url> --edit
```

## 从命令式命令迁移到命令式对象配置

从命令式命令迁移到命令式对象配置涉及几个手动步骤。

1. 将活动对象导出到本地对象配置文件：

   ```shell
   kubectl get <kind>/<name> -o yaml > <kind>_<name>.yaml
   ```

2. 从对象配置文件中手动删除状态字段。

3. 对于后续的对象管理，只能使用 `replace` 。

   ```shell
   kubectl replace -f <kind>_<name>.yaml
   ```

## 定义控制器选择器和 PodTemplate 标签

{{< warning >}}
不建议在控制器上更新选择器。
{{< /warning >}}

推荐的方法是定义单个不变的 PodTemplate 标签，该标签仅由控制器选择器使用，而没有其他语义。

标签示例：

```yaml
selector:
  matchLabels:
      controller-selector: "apps/v1/deployment/nginx"
template:
  metadata:
    labels:
      controller-selector: "apps/v1/deployment/nginx"
```

## {{% heading "whatsnext" %}}

* [使用命令式命令管理 Kubernetes 对象](/zh-cn/docs/tasks/manage-kubernetes-objects/imperative-command/)
* [使用配置文件对 Kubernetes 对象进行声明式管理](/zh-cn/docs/tasks/manage-kubernetes-objects/declarative-config/)
* [Kubectl 命令参考](/docs/reference/generated/kubectl/kubectl-commands/)
* [Kubernetes API 参考](/docs/reference/generated/kubernetes-api/{{< param "version" >}}/)


