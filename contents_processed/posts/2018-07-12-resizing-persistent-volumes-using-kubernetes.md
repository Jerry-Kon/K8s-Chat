---
layout: blog
title: '使用 Kubernetes 调整 PersistentVolume 的大小'
date: 2018-07-12
slug: resize-pv-using-k8s
---

**作者**: Hemant Kumar (Red Hat)

**编者注：这篇博客是[深度文章系列](https://kubernetes.io/blog/2018/06/27/kubernetes-1.11-release-announcement/)的一部分，这个系列介绍了 Kubernetes 1.11 中的新增特性**

在 Kubernetes v1.11 中，持久化卷扩展功能升级为 Beta。
该功能允许用户通过编辑 `PersistentVolumeClaim`（PVC）对象，轻松调整已存在数据卷的大小。
用户不再需要手动与存储后端交互，或者删除再重建 PV 和 PVC 对象来增加卷的大小。缩减持久化卷暂不支持。

卷扩展是在 v1.8 版本中作为 Alpha 功能引入的，
在 v1.11 之前的版本都需要开启特性门控 `ExpandPersistentVolumes` 以及准入控制器 `PersistentVolumeClaimResize`（防止扩展底层存储供应商不支持调整大小的 PVC）。
在 Kubernetes v1.11+ 中，特性门控和准入控制器都是默认启用的。

虽然该功能默认是启用的，但集群管理员必须选择允许用户调整数据卷的大小。
Kubernetes v1.11 为以下树内卷插件提供了卷扩展支持：
AWS-EBS、GCE-PD、Azure Disk、Azure File、Glusterfs、Cinder、Portworx 和 Ceph RBD。
一旦管理员确定底层供应商支持卷扩展，
就可以通过在 `StorageClass` 对象中设置  `allowVolumeExpansion` 字段为  `true`，让用户可以使用该功能。
只有由这个 `StorageClass` 创建的 PVC 才能触发卷扩展。

```
~> cat standard.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: standard
parameters:
  type: pd-standard
provisioner: kubernetes.io/gce-pd
allowVolumeExpansion: true
reclaimPolicy: Delete
```
从这个 `StorageClass` 创建的任何 PVC 都可以被编辑（如下图所示）以请求更多的空间。
Kubernetes 会将存储字段的变化解释为对更多空间的请求，并触发卷大小的自动调整。

![PVC StorageClass](/images/blog/2018-07-12-resizing-persistent-volumes-using-kubernetes/pvc-storageclass.png)

## 文件系统扩展 {#file-system-expansion}

如 GCE-PD、AWS-EBS、Azure Disk、Cinder 和 Ceph RBD 这类的块存储卷类型，
通常需要在扩展卷的额外空间被 Pod 使用之前进行文件系统扩展。
Kubernetes 会在引用数据卷的 Pod 重新启动时自动处理这个问题。

网络附加文件系统（如 Glusterfs 和 Azure File）可以被扩展，而不需要重新启动引用的 Pod，
因为这些系统不需要特殊的文件系统扩展。

文件系统扩展必须通过终止使用该卷的 Pod 来触发。更具体地说：

* 编辑 PVC 以请求更多的空间。
* 一旦底层卷被存储提供商扩展后， PersistentVolume 对象将反映更新的大小，PVC 会有 `FileSystemResizePending`  状态。

你可以通过运行 `kubectl get pvc <pvc_name> -o yaml` 来验证这一点。

```
~> kubectl get pvc myclaim -o yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: myclaim
  namespace: default
  uid: 02d4aa83-83cd-11e8-909d-42010af00004
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 14Gi
  storageClassName: standard
  volumeName: pvc-xxx
status:
  capacity:
    storage: 9G
  conditions:
  - lastProbeTime: null
    lastTransitionTime: 2018-07-11T14:51:10Z
    message: Waiting for user to (re-)start a pod to finish file system resize of
      volume on node.
    status: "True"
    type: FileSystemResizePending
  phase: Bound
```
* 一旦 PVC 具有 `FileSystemResizePending` 状态 ，就可以重启使用该 PVC 的 Pod 以完成节点上的文件系统大小调整。
重新启动可以通过删除并重新创建 Pod，或者通过 Deployment 缩容后再扩容来实现。
* 一旦文件系统的大小调整完成，PVC 将自动更新以展现新的大小。

在扩展文件系统时遇到的任何错误都应作为 Pod 的事件而存在。

## 在线文件系统扩展 {#online-file-system-expansion}

Kubernetes v1.11 里还引入了一个名为在线文件系统扩展的 Alpha 功能。
该功能可以让一个正在被 Pod 使用的卷进行文件系统扩展。
因为这个功能是 Alpha 阶段，所以它需要启用特性门控 `ExpandInUsePersistentVolumes`。
树内卷插件 GCE-PD、AWS-EBS、Cinder 和 Ceph RBD 都支持该功能。
当这个功能被启用时，引用调整后的卷的 Pod 不需要被重新启动。
相反，在使用中文件系统将作为卷扩展的一部分自动调整大小。
文件系统的扩展是在一个 Pod 引用调整后的卷时才发生的，所以如果没有引用卷的 Pod 在运行，文件系统的扩展就不会发生。

## 更多信息 {#how-can-i-learn-more}

在这里查看有关这一特性的其他文档：
https://kubernetes.io/zh-cn/docs/concepts/storage/persistent-volumes/