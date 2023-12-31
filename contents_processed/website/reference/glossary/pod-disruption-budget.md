---
id: pod-disruption-budget
title: Pod Disruption Budget
full-link: /zh-cn/docs/concepts/workloads/pods/disruptions/
date: 2019-02-12
short_description: >
 Pod Disruption Budget 是这样一种对象：它保证在主动中断（ voluntary disruptions）时，多实例应用的 {{< glossary_tooltip text="Pod" term_id="pod" >}} 不会少于一定的数量。

aka:
 - PDB
related:
 - pod
 - container
tags:
 - operation
---


 [Pod 干扰预算（Pod Disruption Budget，PDB）](/zh-cn/docs/concepts/workloads/pods/disruptions/)
 使应用所有者能够为多实例应用创建一个对象，来确保一定数量的具有指定标签的 Pod 在任何时候都不会被主动驱逐。
PDB 无法防止非主动的中断，但是会计入预算（budget）。
