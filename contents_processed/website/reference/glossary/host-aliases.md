---
title: HostAliases
id: HostAliases
date: 2019-01-31
full_link: /docs/reference/generated/kubernetes-api/{{< param "version" >}}/#hostalias-v1-core
short_description: >
  主机别名 (HostAliases) 是一组 IP 地址和主机名的映射，用于注入到 Pod 内的 hosts 文件。

aka:
tags:
- operation
---

 主机别名 (HostAliases) 是一组 IP 地址和主机名的映射，用于注入到 {{< glossary_tooltip text="Pod" term_id="pod" >}} 内的 hosts 文件。



[HostAliases](/docs/reference/generated/kubernetes-api/{{< param "version" >}}/#hostalias-v1-core)
是一个包含主机名和 IP 地址的可选列表，配置后将被注入到 Pod 内的 hosts 文件中。
该选项仅适用于没有配置 hostNetwork 的 Pod。
