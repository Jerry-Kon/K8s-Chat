---
api_metadata:
  apiVersion: "v1"
  import: "k8s.io/api/core/v1"
  kind: "ResourceQuota"
content_type: "api_reference"
description: "ResourceQuota 设置每个命名空间强制执行的聚合配额限制。"
title: "ResourceQuota"
weight: 2
---


`apiVersion: v1`

`import "k8s.io/api/core/v1"`

## ResourceQuota {#ResourceQuota}

ResourceQuota 设置每个命名空间强制执行的聚合配额限制。

<hr>

- **apiVersion**: v1

- **kind**: ResourceQuota

- **metadata** (<a href="{{< ref "../common-definitions/object-meta#ObjectMeta" >}}">ObjectMeta</a>)


  标准的对象元数据。
  更多信息： https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata

- **spec** (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuotaSpec" >}}">ResourceQuotaSpec</a>)


  spec 定义所需的配额。
  https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status

- **status** (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuotaStatus" >}}">ResourceQuotaStatus</a>)


  status 定义实际执行的配额及其当前使用情况。
  https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status

## ResourceQuotaSpec {#ResourceQuotaSpec}

ResourceQuotaSpec 定义为 Quota 强制执行所需的硬限制。

<hr>

- **hard** (map[string]<a href="{{< ref "../common-definitions/quantity#Quantity" >}}">Quantity</a>)


  hard 是每种指定资源所需的硬性限制集合。
  更多信息： https://kubernetes.io/docs/concepts/policy/resource-quotas/

- **scopeSelector** (ScopeSelector)


  scopeSelector 也是一组过滤器的集合，和 scopes 类似，
  必须匹配配额所跟踪的每个对象，但使用 ScopeSelectorOperator 结合可能的值来表示。
  对于要匹配的资源，必须同时匹配 scopes 和 scopeSelector（如果在 spec 中设置了的话）。

  <a name="ScopeSelector"></a>

  scope 选择算符表示的是由限定范围的资源选择算符进行 **逻辑与** 计算得出的结果。

  - **scopeSelector.matchExpressions** ([]ScopedResourceSelectorRequirement)


    按资源范围划分的范围选择算符需求列表。

    <a name="ScopedResourceSelectorRequirement"></a>

    限定范围的资源选择算符需求是一种选择算符，包含值、范围名称和将二者关联起来的运算符。

    - **scopeSelector.matchExpressions.operator** (string)，必需


      表示范围与一组值之间的关系。有效的运算符为 In、NotIn、Exists、DoesNotExist。

    - **scopeSelector.matchExpressions.scopeName** (string)，必需


      选择器所适用的范围的名称。

    - **scopeSelector.matchExpressions.values** ([]string)


      字符串值数组。
      如果操作符是 In 或 NotIn，values 数组必须是非空的。
      如果操作符是 Exists 或 DoesNotExist，values 数组必须为空。
      该数组将在策略性合并补丁操作期间被替换。

- **scopes** ([]string)


  一个匹配被配额跟踪的所有对象的过滤器集合。
  如果没有指定，则默认匹配所有对象。

## ResourceQuotaStatus {#ResourceQuotaStatus}

ResourceQuotaStatus 定义硬性限制和观测到的用量。

<hr>

- **hard** (map[string]<a href="{{< ref "../common-definitions/quantity#Quantity" >}}">Quantity</a>)


  hard 是每种指定资源所强制实施的硬性限制集合。
  更多信息： https://kubernetes.io/docs/concepts/policy/resource-quotas/

- **used** (map[string]<a href="{{< ref "../common-definitions/quantity#Quantity" >}}">Quantity</a>)


  used 是当前命名空间中所观察到的资源总用量。

## ResourceQuotaList {#ResourceQuotaList}

ResourceQuotaList 是 ResourceQuota 列表。

<hr>

- **apiVersion**：v1

- **kind**：ResourceQuotaList

- **metadata** (<a href="{{< ref "../common-definitions/list-meta#ListMeta" >}}">ListMeta</a>)


  标准列表元数据。
  更多信息： https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds

- **items** ([]<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>)，必需


  items 是 ResourceQuota 对象的列表。
  更多信息： https://kubernetes.io/docs/concepts/policy/resource-quotas/

## 操作 {#Operations}

<hr>

### `get` 读取指定的 ResourceQuota

#### HTTP 请求

GET /api/v1/namespaces/{namespace}/resourcequotas/{name}

#### 参数

- **name** （**路径参数**）: string, 必需

  ResourceQuota 的名称

- **namespace** （**路径参数**）: string, 必需

  <a href="{{< ref "../common-parameters/common-parameters#namespace" >}}">namespace</a>

- **pretty** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

#### 响应

200 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): OK

401: Unauthorized

### `get` 读取指定的 ResourceQuota 的状态

#### HTTP 请求

GET /api/v1/namespaces/{namespace}/resourcequotas/{name}/status

#### 参数

- **name** （**路径参数**）: string, 必需

  ResourceQuota 的名称

- **namespace** （**路径参数**）: string, 必需

  <a href="{{< ref "../common-parameters/common-parameters#namespace" >}}">namespace</a>

- **pretty** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

#### 响应

200 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): OK

401: Unauthorized

### `list` 列出或监视 ResourceQuota 类别的对象

#### HTTP 请求

GET /api/v1/namespaces/{namespace}/resourcequotas

#### 参数

- **namespace** （**路径参数**）: string, 必需

  <a href="{{< ref "../common-parameters/common-parameters#namespace" >}}">namespace</a>

- **allowWatchBookmarks** （**查询参数**）: boolean

  <a href="{{< ref "../common-parameters/common-parameters#allowWatchBookmarks" >}}">allowWatchBookmarks</a>

- **continue** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#continue" >}}">continue</a>

- **fieldSelector** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#fieldSelector" >}}">fieldSelector</a>

- **labelSelector** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#labelSelector" >}}">labelSelector</a>

- **limit** （**查询参数**）: integer

  <a href="{{< ref "../common-parameters/common-parameters#limit" >}}">limit</a>

- **pretty** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

- **resourceVersion** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#resourceVersion" >}}">resourceVersion</a>

- **resourceVersionMatch** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#resourceVersionMatch" >}}">resourceVersionMatch</a>

- **sendInitialEvents** (**查询参数**): boolean

  <a href="{{< ref "../common-parameters/common-parameters#sendInitialEvents" >}}">sendInitialEvents</a>

- **timeoutSeconds** （**查询参数**）: integer

  <a href="{{< ref "../common-parameters/common-parameters#timeoutSeconds" >}}">timeoutSeconds</a>

- **watch** （**查询参数**）: boolean

  <a href="{{< ref "../common-parameters/common-parameters#watch" >}}">watch</a>

#### 响应

200 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuotaList" >}}">ResourceQuotaList</a>): OK

401: Unauthorized

### `list` 列出或监视 ResourceQuota 类别的对象

#### HTTP 请求

GET /api/v1/resourcequotas

#### 参数

- **allowWatchBookmarks** （**查询参数**）: boolean

  <a href="{{< ref "../common-parameters/common-parameters#allowWatchBookmarks" >}}">allowWatchBookmarks</a>

- **continue** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#continue" >}}">continue</a>

- **fieldSelector** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#fieldSelector" >}}">fieldSelector</a>

- **labelSelector** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#labelSelector" >}}">labelSelector</a>

- **limit** （**查询参数**）: integer

  <a href="{{< ref "../common-parameters/common-parameters#limit" >}}">limit</a>

- **pretty** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

- **resourceVersion** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#resourceVersion" >}}">resourceVersion</a>

- **resourceVersionMatch** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#resourceVersionMatch" >}}">resourceVersionMatch</a>

- **sendInitialEvents** (**查询参数**): boolean

  <a href="{{< ref "../common-parameters/common-parameters#sendInitialEvents" >}}">sendInitialEvents</a>

- **timeoutSeconds** （**查询参数**）: integer

  <a href="{{< ref "../common-parameters/common-parameters#timeoutSeconds" >}}">timeoutSeconds</a>

- **watch** （**查询参数**）: boolean

  <a href="{{< ref "../common-parameters/common-parameters#watch" >}}">watch</a>

#### 响应

200 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuotaList" >}}">ResourceQuotaList</a>): OK

401: Unauthorized

### `create` 创建一个 ResourceQuota

#### HTTP 请求

POST /api/v1/namespaces/{namespace}/resourcequotas

#### 参数

- **namespace** （**路径参数**）: string, 必需

  <a href="{{< ref "../common-parameters/common-parameters#namespace" >}}">namespace</a>

- **body**: <a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>, 必需

- **dryRun** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#dryRun" >}}">dryRun</a>

- **fieldManager** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#fieldManager" >}}">fieldManager</a>

- **fieldValidation** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#fieldValidation" >}}">fieldValidation</a>

- **pretty** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

#### 响应

200 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): OK

201 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): Created

202 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): Accepted

401: Unauthorized

### `update` 更新指定的 ResourceQuota

#### HTTP 请求

PUT /api/v1/namespaces/{namespace}/resourcequotas/{name}

#### 参数

- **name** （**路径参数**）: string, 必需

  ResourceQuota 的名称

- **namespace** （**路径参数**）: string, 必需

  <a href="{{< ref "../common-parameters/common-parameters#namespace" >}}">namespace</a>

- **body**: <a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>, 必需

- **dryRun** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#dryRun" >}}">dryRun</a>

- **fieldManager** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#fieldManager" >}}">fieldManager</a>

- **fieldValidation** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#fieldValidation" >}}">fieldValidation</a>

- **pretty** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

#### 响应

200 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): OK

201 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): Created

401: Unauthorized

### `update` 更新指定 ResourceQuota 的状态

#### HTTP 请求

PUT /api/v1/namespaces/{namespace}/resourcequotas/{name}/status

#### 参数

- **name** （**路径参数**）: string, 必需

  ResourceQuota 的名称

- **namespace** （**路径参数**）: string, 必需

  <a href="{{< ref "../common-parameters/common-parameters#namespace" >}}">namespace</a>

- **body**: <a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>, 必需

- **dryRun** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#dryRun" >}}">dryRun</a>

- **fieldManager** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#fieldManager" >}}">fieldManager</a>

- **fieldValidation** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#fieldValidation" >}}">fieldValidation</a>

- **pretty** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

#### 响应

200 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): OK

201 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): Created

401: Unauthorized

### `patch` 部分更新指定的 ResourceQuota

#### HTTP 请求

PATCH /api/v1/namespaces/{namespace}/resourcequotas/{name}

#### 参数

- **name** （**路径参数**）: string, 必需

  ResourceQuota 的名称

- **namespace** （**路径参数**）: string, 必需

  <a href="{{< ref "../common-parameters/common-parameters#namespace" >}}">namespace</a>

- **body**: <a href="{{< ref "../common-definitions/patch#Patch" >}}">Patch</a>, 必需

- **dryRun** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#dryRun" >}}">dryRun</a>

- **fieldManager** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#fieldManager" >}}">fieldManager</a>

- **fieldValidation** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#fieldValidation" >}}">fieldValidation</a>

- **force** （**查询参数**）: boolean

  <a href="{{< ref "../common-parameters/common-parameters#force" >}}">force</a>

- **pretty** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

#### 响应

200 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): OK

201 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): Created

401: Unauthorized

### `patch` 部分更新指定 ResourceQuota 的状态

#### HTTP 请求

PATCH /api/v1/namespaces/{namespace}/resourcequotas/{name}/status

#### 参数

- **name** （**路径参数**）: string, 必需

  ResourceQuota 的名称

- **namespace** （**路径参数**）: string, 必需

  <a href="{{< ref "../common-parameters/common-parameters#namespace" >}}">namespace</a>

- **body**: <a href="{{< ref "../common-definitions/patch#Patch" >}}">Patch</a>, 必需

- **dryRun** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#dryRun" >}}">dryRun</a>

- **fieldManager** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#fieldManager" >}}">fieldManager</a>

- **fieldValidation** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#fieldValidation" >}}">fieldValidation</a>

- **force** （**查询参数**）: boolean

  <a href="{{< ref "../common-parameters/common-parameters#force" >}}">force</a>

- **pretty** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

#### 响应

200 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): OK

201 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): Created

401: Unauthorized

### `delete` 删除 ResourceQuota

#### HTTP 请求

DELETE /api/v1/namespaces/{namespace}/resourcequotas/{name}

#### 参数

- **name** （**路径参数**）: string, 必需

  ResourceQuota 的名称

- **namespace** （**路径参数**）: string, 必需

  <a href="{{< ref "../common-parameters/common-parameters#namespace" >}}">namespace</a>

- **body**: <a href="{{< ref "../common-definitions/delete-options#DeleteOptions" >}}">DeleteOptions</a>

- **dryRun** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#dryRun" >}}">dryRun</a>

- **gracePeriodSeconds** （**查询参数**）: integer

  <a href="{{< ref "../common-parameters/common-parameters#gracePeriodSeconds" >}}">gracePeriodSeconds</a>

- **pretty** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

- **propagationPolicy** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#propagationPolicy" >}}">propagationPolicy</a>

#### 响应

200 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): OK

202 (<a href="{{< ref "../policy-resources/resource-quota-v1#ResourceQuota" >}}">ResourceQuota</a>): Accepted

401: Unauthorized

### `deletecollection` 删除 ResourceQuota 的集合

#### HTTP 请求

DELETE /api/v1/namespaces/{namespace}/resourcequotas

#### 参数

- **namespace** （**路径参数**）: string, 必需

  <a href="{{< ref "../common-parameters/common-parameters#namespace" >}}">namespace</a>

- **body**: <a href="{{< ref "../common-definitions/delete-options#DeleteOptions" >}}">DeleteOptions</a>

- **continue** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#continue" >}}">continue</a>

- **dryRun** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#dryRun" >}}">dryRun</a>

- **fieldSelector** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#fieldSelector" >}}">fieldSelector</a>

- **gracePeriodSeconds** （**查询参数**）: integer

  <a href="{{< ref "../common-parameters/common-parameters#gracePeriodSeconds" >}}">gracePeriodSeconds</a>

- **labelSelector** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#labelSelector" >}}">labelSelector</a>

- **limit** （**查询参数**）: integer

  <a href="{{< ref "../common-parameters/common-parameters#limit" >}}">limit</a>

- **pretty** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

- **propagationPolicy** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#propagationPolicy" >}}">propagationPolicy</a>

- **resourceVersion** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#resourceVersion" >}}">resourceVersion</a>

- **resourceVersionMatch** （**查询参数**）: string

  <a href="{{< ref "../common-parameters/common-parameters#resourceVersionMatch" >}}">resourceVersionMatch</a>

- **sendInitialEvents** (*查询参数*): boolean

  <a href="{{< ref "../common-parameters/common-parameters#sendInitialEvents" >}}">sendInitialEvents</a>

- **timeoutSeconds** （**查询参数**）: integer

  <a href="{{< ref "../common-parameters/common-parameters#timeoutSeconds" >}}">timeoutSeconds</a>

#### 响应

200 (<a href="{{< ref "../common-definitions/status#Status" >}}">Status</a>): OK

401: Unauthorized

