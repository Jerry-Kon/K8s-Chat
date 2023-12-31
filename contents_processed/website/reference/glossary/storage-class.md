---
title: StorageClass
id: storageclass
date: 2018-04-12
full_link: /zh-cn/docs/concepts/storage/storage-classes/
short_description: >
  StorageClass 是管理员用来描述可用的不同存储类型的一种方法。

aka: 
tags:
- core-object
- storage
---



 StorageClass 是管理员用来描述不同的可用存储类型的一种方法。



StorageClass 可以映射到服务质量等级（QoS）、备份策略、或者管理员任意定义的策略。
每个 StorageClass 对象包含的字段有 `provisioner`、`parameters` 和 `reclaimPolicy`。
动态制备该存储类别的{{< glossary_tooltip text="持久卷" term_id="persistent-volume" >}}时需要用到这些字段值。
通过设置 StorageClass 对象的名称，用户可以请求特定存储类别。