---
title: " 使用 Kubernetes Pet Sets 和 Datera Elastic Data Fabric 的 FlexVolume 扩展有状态的应用程序 "
date: 2016-08-29
slug: stateful-applications-using-kubernetes-datera
---

_编者注：今天的邀请帖子来自 Datera 公司的软件架构师 Shailesh Mittal 和高级产品总监 Ashok Rajagopalan，介绍在 Datera Elastic Data Fabric 上用 Kubernetes 配置状态应用程序。_

**简介** 

用户从无状态工作负载转移到运行有状态应用程序，Kubernetes 中的持久卷是基础。虽然 Kubernetes 早已支持有状态的应用程序，比如 MySQL、Kafka、Cassandra 和 Couchbase，但是 Pet Sets 的引入明显改善了情况。特别是，[Pet Sets](/docs/user-guide/petset/) 具有持续扩展和关联的能力，在配置和启动的顺序过程中，可以自动缩放“Pets”（需要连续处理和持久放置的应用程序）。

Datera 是用于云部署的弹性块存储，可以通过 [FlexVolume](/docs/user-guide/volumes/#flexvolume) 框架与 [Kubernetes 无缝集成](http://datera.io/blog-library/8/19/datera-simplifies-stateful-containers-on-kubernetes-13)。基于容器的基本原则，Datera 允许应用程序的资源配置与底层物理基础架构分离，为有状态的应用程序提供简洁的协议（也就是说，不依赖底层物理基础结构及其相关内容）、声明式格式和最后移植的能力。

Kubernetes 可以通过 yaml 配置来灵活定义底层应用程序基础架构，而 Datera 可以将该配置传递给存储基础结构以提供持久性。通过 Datera AppTemplates 声明，在 Kubernetes 环境中，有状态的应用程序可以自动扩展。




**部署永久性存储**



永久性存储是通过 Kubernetes 的子系统 [PersistentVolume](/docs/user-guide/persistent-volumes/#persistent-volumes) 定义的。PersistentVolumes 是卷插件，它定义的卷的生命周期和使用它的 Pod 相互独立。PersistentVolumes 由 NFS、iSCSI 或云提供商的特定存储系统实现。Datera 开发了用于 PersistentVolumes 的卷插件，可以在 Datera Data Fabric 上为 Kubernetes 的 Pod 配置 iSCSI 块存储。


Datera 卷插件从 minion nodes 上的 kubelet 调用，并通过 REST API 回传到 Datera Data Fabric。以下是带有 Datera 插件的 PersistentVolume 的部署示例：


 ```
  apiVersion: v1

  kind: PersistentVolume

  metadata:

    name: pv-datera-0

  spec:

    capacity:

      storage: 100Gi

    accessModes:

      - ReadWriteOnce

    persistentVolumeReclaimPolicy: Retain

    flexVolume:

      driver: "datera/iscsi"

      fsType: "xfs"

      options:

        volumeID: "kube-pv-datera-0"

        size: “100"

        replica: "3"

        backstoreServer: "[tlx170.tlx.daterainc.com](http://tlx170.tlx.daterainc.com/):7717”
  ```


为 Pod 申请 PersistentVolume，要按照以下清单在  Datera Data Fabric 中配置 100 GB 的 PersistentVolume。



 ```
[root@tlx241 /]# kubectl get pv

NAME          CAPACITY   ACCESSMODES   STATUS      CLAIM     REASON    AGE

pv-datera-0   100Gi        RWO         Available                       8s

pv-datera-1   100Gi        RWO         Available                       2s

pv-datera-2   100Gi        RWO         Available                       7s

pv-datera-3   100Gi        RWO         Available                       4s
  ```


**配置**



Datera PersistenceVolume 插件安装在所有 minion node 上。minion node 的声明是绑定到之前设置的永久性存储上的，当 Pod 进入具备有效声明的 minion node 上时，Datera 插件会转发请求，从而在 Datera Data Fabric 上创建卷。根据配置请求，PersistentVolume 清单中所有指定的选项都将发送到插件。

在 Datera Data Fabric 中配置的卷会作为 iSCSI 块设备呈现给 minion node，并且 kubelet 将该设备安装到容器（在 Pod 中）进行访问。

 ![](https://lh4.googleusercontent.com/ILlUm1HrWhGa8uTt97dQ786Gn20FHFZkavfucz05NHv6moZWiGDG7GlELM6o4CSzANWvZckoAVug5o4jMg17a-PbrfD1FRbDPeUCIc8fKVmVBNUsUPshWanXYkBa3gIJy5BnhLmZ)


**使用永久性存储**



Kubernetes PersistentVolumes 与具备 PersistentVolume Claims 的 Pod 一起使用。定义声明后，会被绑定到与声明规范匹配的 PersistentVolume 上。上面提到的定义 PersistentVolume 的典型声明如下所示：



 ```
kind: PersistentVolumeClaim

apiVersion: v1

metadata:

  name: pv-claim-test-petset-0

spec:

  accessModes:

    - ReadWriteOnce

  resources:

    requests:

      storage: 100Gi
  ```


定义这个声明并将其绑定到 PersistentVolume 时，资源与 Pod 规范可以一起使用：



 ```
[root@tlx241 /]# kubectl get pv

NAME          CAPACITY   ACCESSMODES   STATUS      CLAIM                            REASON    AGE

pv-datera-0   100Gi      RWO           Bound       default/pv-claim-test-petset-0             6m

pv-datera-1   100Gi      RWO           Bound       default/pv-claim-test-petset-1             6m

pv-datera-2   100Gi      RWO           Available                                              7s

pv-datera-3   100Gi      RWO           Available                                              4s


[root@tlx241 /]# kubectl get pvc

NAME                     STATUS    VOLUME        CAPACITY   ACCESSMODES   AGE

pv-claim-test-petset-0   Bound     pv-datera-0   0                        3m

pv-claim-test-petset-1   Bound     pv-datera-1   0                        3m
  ```


Pod 可以使用 PersistentVolume 声明，如下所示：


 ```
apiVersion: v1

kind: Pod

metadata:

  name: kube-pv-demo

spec:

  containers:

  - name: data-pv-demo

    image: nginx

    volumeMounts:

    - name: test-kube-pv1

      mountPath: /data

    ports:

    - containerPort: 80

  volumes:

  - name: test-kube-pv1

    persistentVolumeClaim:

      claimName: pv-claim-test-petset-0
  ```


程序的结果是 Pod 将 PersistentVolume Claim 作为卷。依次将请求发送到 Datera 卷插件，然后在 Datera Data Fabric 中配置存储。



 ```
[root@tlx241 /]# kubectl describe pods kube-pv-demo

Name:       kube-pv-demo

Namespace:  default

Node:       tlx243/172.19.1.243

Start Time: Sun, 14 Aug 2016 19:17:31 -0700

Labels:     \<none\>

Status:     Running

IP:         10.40.0.3

Controllers: \<none\>

Containers:

  data-pv-demo:

    Container ID: [docker://ae2a50c25e03143d0dd721cafdcc6543fac85a301531110e938a8e0433f74447](about:blank)

    Image:   nginx

    Image ID: [docker://sha256:0d409d33b27e47423b049f7f863faa08655a8c901749c2b25b93ca67d01a470d](about:blank)

    Port:    80/TCP

    State:   Running

      Started:  Sun, 14 Aug 2016 19:17:34 -0700

    Ready:   True

    Restart Count:  0

    Environment Variables:  \<none\>

Conditions:

  Type           Status

  Initialized    True

  Ready          True

  PodScheduled   True

Volumes:

  test-kube-pv1:

    Type:  PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)

    ClaimName:   pv-claim-test-petset-0

    ReadOnly:    false

  default-token-q3eva:

    Type:        Secret (a volume populated by a Secret)

    SecretName:  default-token-q3eva

    QoS Tier:  BestEffort

Events:

  FirstSeen LastSeen Count From SubobjectPath Type Reason Message

  --------- -------- ----- ---- ------------- -------- ------ -------

  43s 43s 1 {default-scheduler } Normal Scheduled Successfully assigned kube-pv-demo to tlx243

  42s 42s 1 {kubelet tlx243} spec.containers{data-pv-demo} Normal Pulling pulling image "nginx"

  40s 40s 1 {kubelet tlx243} spec.containers{data-pv-demo} Normal Pulled Successfully pulled image "nginx"

  40s 40s 1 {kubelet tlx243} spec.containers{data-pv-demo} Normal Created Created container with docker id ae2a50c25e03

  40s 40s 1 {kubelet tlx243} spec.containers{data-pv-demo} Normal Started Started container with docker id ae2a50c25e03
  ```


永久卷在 minion node（在本例中为 tlx243）中显示为 iSCSI 设备：


 ```
[root@tlx243 ~]# lsscsi

[0:2:0:0]    disk    SMC      SMC2208          3.24  /dev/sda

[11:0:0:0]   disk    DATERA   IBLOCK           4.0   /dev/sdb


[root@tlx243 datera~iscsi]# mount  ``` grep sdb

/dev/sdb on /var/lib/kubelet/pods/6b99bd2a-628e-11e6-8463-0cc47ab41442/volumes/datera~iscsi/pv-datera-0 type xfs (rw,relatime,attr2,inode64,noquota)
  ```


在 Pod 中运行的容器按照清单中将设备安装在 /data 上：


 ```
[root@tlx241 /]# kubectl exec kube-pv-demo -c data-pv-demo -it bash

root@kube-pv-demo:/# mount  ``` grep data

/dev/sdb on /data type xfs (rw,relatime,attr2,inode64,noquota)
  ```



**使用 Pet Sets**



通常，Pod 被视为无状态单元，因此，如果其中之一状态异常或被取代，Kubernetes 会将其丢弃。相反，PetSet 是一组有状态的 Pod，具有更强的身份概念。PetSet 可以将标识分配给应用程序的各个实例，这些应用程序没有与底层物理结构连接，PetSet 可以消除这种依赖性。



每个 PetSet 需要{0..n-1}个 Pet。每个 Pet 都有一个确定的名字、PetSetName-Ordinal 和唯一的身份。每个 Pet 最多有一个 Pod，每个 PetSet 最多包含一个给定身份的 Pet。要确保每个 PetSet 在任何特定时间运行时，具有唯一标识的“pet”的数量都是确定的。Pet 的身份标识包括以下几点：

- 一个稳定的主机名，可以在 DNS 中使用
- 一个序号索引
- 稳定的存储：链接到序号和主机名


使用 PersistentVolume Claim 定义 PetSet 的典型例子如下所示：


 ```
# A headless service to create DNS records

apiVersion: v1

kind: Service

metadata:

  name: test-service

  labels:

    app: nginx

spec:

  ports:

  - port: 80

    name: web

  clusterIP: None

  selector:

    app: nginx

---

apiVersion: apps/v1alpha1

kind: PetSet

metadata:

  name: test-petset

spec:

  serviceName: "test-service"

  replicas: 2

  template:

    metadata:

      labels:

        app: nginx

      annotations:

        [pod.alpha.kubernetes.io/initialized:](http://pod.alpha.kubernetes.io/initialized:) "true"

    spec:

      terminationGracePeriodSeconds: 0

      containers:

      - name: nginx

        image: [gcr.io/google\_containers/nginx-slim:0.8](http://gcr.io/google_containers/nginx-slim:0.8)

        ports:

        - containerPort: 80

          name: web

        volumeMounts:

        - name: pv-claim

          mountPath: /data

  volumeClaimTemplates:

  - metadata:

      name: pv-claim

      annotations:

        [volume.alpha.kubernetes.io/storage-class:](http://volume.alpha.kubernetes.io/storage-class:) anything

    spec:

      accessModes: ["ReadWriteOnce"]

      resources:

        requests:

          storage: 100Gi
  ```


我们提供以下 PersistentVolume Claim：


 ```
[root@tlx241 /]# kubectl get pvc

NAME                     STATUS    VOLUME        CAPACITY   ACCESSMODES   AGE

pv-claim-test-petset-0   Bound     pv-datera-0   0                        41m

pv-claim-test-petset-1   Bound     pv-datera-1   0                        41m

pv-claim-test-petset-2   Bound     pv-datera-2   0                        5s

pv-claim-test-petset-3   Bound     pv-datera-3   0                        2s
  ```


配置 PetSet 时，将实例化两个 Pod：


 ```
[root@tlx241 /]# kubectl get pods

NAMESPACE     NAME                        READY     STATUS    RESTARTS   AGE

default       test-petset-0               1/1       Running   0          7s

default       test-petset-1               1/1       Running   0          3s
  ```


以下是一个 PetSet：test-petset 实例化之前的样子：



 ```
[root@tlx241 /]# kubectl describe petset test-petset

Name: test-petset

Namespace: default

Image(s): [gcr.io/google\_containers/nginx-slim:0.8](http://gcr.io/google_containers/nginx-slim:0.8)

Selector: app=nginx

Labels: app=nginx

Replicas: 2 current / 2 desired

Annotations: \<none\>

CreationTimestamp: Sun, 14 Aug 2016 19:46:30 -0700

Pods Status: 2 Running / 0 Waiting / 0 Succeeded / 0 Failed

No volumes.

No events.
  ```


一旦实例化 PetSet（例如下面的 test-petset），随着副本数（从 PetSet 的初始 Pod 数量算起）的增加，实例化的 Pod 将变得更多，并且更多的 PersistentVolume Claim 会绑定到新的 Pod 上：


 ```
[root@tlx241 /]# kubectl patch petset test-petset -p'{"spec":{"replicas":"3"}}'

"test-petset” patched


[root@tlx241 /]# kubectl describe petset test-petset

Name: test-petset

Namespace: default

Image(s): [gcr.io/google\_containers/nginx-slim:0.8](http://gcr.io/google_containers/nginx-slim:0.8)

Selector: app=nginx

Labels: app=nginx

Replicas: 3 current / 3 desired

Annotations: \<none\>

CreationTimestamp: Sun, 14 Aug 2016 19:46:30 -0700

Pods Status: 3 Running / 0 Waiting / 0 Succeeded / 0 Failed

No volumes.

No events.


[root@tlx241 /]# kubectl get pods

NAME                        READY     STATUS    RESTARTS   AGE

test-petset-0               1/1       Running   0          29m

test-petset-1               1/1       Running   0          28m

test-petset-2               1/1       Running   0          9s
  ```


现在，应用修补程序后，PetSet 正在运行3个 Pod。



当上述 PetSet 定义修补完成，会产生另一个副本，PetSet 将在系统中引入另一个 pod。反之，这会导致在 Datera Data Fabric 上配置更多的卷。因此，在 PetSet 进行扩展时，要配置动态卷并将其附加到 Pod 上。


为了平衡持久性和一致性的概念，如果 Pod 从一个 Minion 转移到另一个，卷确实会附加（安装）到新的 minion node 上，并与旧的 Minion 分离（卸载），从而实现对数据的持久访问。



**结论**



本文展示了具备 Pet Sets 的 Kubernetes 协调有状态和无状态工作负载。当 Kubernetes 社区致力于扩展 FlexVolume 框架的功能时，我们很高兴这个解决方案使 Kubernetes 能够在数据中心广泛运行。


加入我们并作出贡献：Kubernetes [Storage SIG](https://groups.google.com/forum/#!forum/kubernetes-sig-storage).



- [下载 Kubernetes](http://get.k8s.io/)
- 参与 Kubernetes 项目 [GitHub](https://github.com/kubernetes/kubernetes)
- 发布问题（或者回答问题） [Stack Overflow](http://stackoverflow.com/questions/tagged/kubernetes)
- 联系社区 [k8s Slack](http://slack.k8s.io/)
- 在 Twitter 上关注我们 [@Kubernetesio](https://twitter.com/kubernetesio) for latest updates
