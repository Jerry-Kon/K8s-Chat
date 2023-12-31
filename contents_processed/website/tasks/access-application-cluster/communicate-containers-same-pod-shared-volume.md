---
title: 同 Pod 内的容器使用共享卷通信
content_type: task
weight: 120
---

本文旨在说明如何让一个 Pod 内的两个容器使用一个卷（Volume）进行通信。
参阅如何让两个进程跨容器通过
[共享进程名字空间](/zh-cn/docs/tasks/configure-pod-container/share-process-namespace/)。

## {{% heading "prerequisites" %}}

{{< include "task-tutorial-prereqs.md" >}} {{< version-check >}}


## 创建一个包含两个容器的 Pod   {#creating-a-pod-that-runs-two-containers}

在这个练习中，你会创建一个包含两个容器的 Pod。两个容器共享一个卷用于他们之间的通信。
Pod 的配置文件如下：

{{< codenew file="pods/two-container-pod.yaml" >}}

在配置文件中，你可以看到 Pod 有一个共享卷，名为 `shared-data`。

配置文件中的第一个容器运行了一个 nginx 服务器。共享卷的挂载路径是 `/usr/share/nginx/html`。
第二个容器是基于 debian 镜像的，有一个 `/pod-data` 的挂载路径。第二个容器运行了下面的命令然后终止。

```shell
echo Hello from the debian container > /pod-data/index.html
```

注意，第二个容器在 nginx 服务器的根目录下写了 `index.html` 文件。

创建一个包含两个容器的 Pod：

```shell
kubectl apply -f https://k8s.io/examples/pods/two-container-pod.yaml
```

查看 Pod 和容器的信息：

```shell
kubectl get pod two-containers --output=yaml
```

这是输出的一部分：

```yaml
apiVersion: v1
kind: Pod
metadata:
  ...
  name: two-containers
  namespace: default
  ...
spec:
  ...
  containerStatuses:

  - containerID: docker://c1d8abd1 ...
    image: debian
    ...
    lastState:
      terminated:
        ...
    name: debian-container
    ...

  - containerID: docker://96c1ff2c5bb ...
    image: nginx
    ...
    name: nginx-container
    ...
    state:
      running:
    ...
```

你可以看到 debian 容器已经被终止了，而 nginx 服务器依然在运行。

进入 nginx 容器的 shell：

```shell
kubectl exec -it two-containers -c nginx-container -- /bin/bash
```

在 shell 中，确认 nginx 还在运行。

```
root@two-containers:/# apt-get update
root@two-containers:/# apt-get install curl procps
root@two-containers:/# ps aux
```

输出类似于这样：

```
USER       PID  ...  STAT START   TIME COMMAND
root         1  ...  Ss   21:12   0:00 nginx: master process nginx -g daemon off;
```

回忆一下，debian 容器在 nginx 的根目录下创建了 `index.html` 文件。
使用 `curl` 向 nginx 服务器发送一个 GET 请求：

```
root@two-containers:/# curl localhost
```

输出表示 nginx 向外提供了 debian 容器所写就的页面：

```
Hello from the debian container
```

## 讨论   {#discussion}

Pod 能有多个容器的主要原因是为了支持辅助应用（helper applications），以协助主应用（primary application）。
辅助应用的典型例子是数据抽取，数据推送和代理。辅助应用和主应用经常需要相互通信。
就如这个练习所示，通信通常是通过共享文件系统完成的，或者，也通过回环网络接口 localhost 完成。
举个网络接口的例子，web 服务器带有一个协助程序用于拉取 Git 仓库的更新。

在本练习中的卷为 Pod 生命周期中的容器相互通信提供了一种方法。如果 Pod 被删除或者重建了，
任何共享卷中的数据都会丢失。

## {{% heading "whatsnext" %}}

* 进一步了解[复合容器的模式](/blog/2015/06/the-distributed-system-toolkit-patterns/)
* 学习[模块化架构中的复合容器](https://www.slideshare.net/Docker/slideshare-burns)
* 参见[配置 Pod 使用卷来存储数据](/zh-cn/docs/tasks/configure-pod-container/configure-volume-storage/)
* 参考[在 Pod 中的容器之间共享进程命名空间](/zh-cn/docs/tasks/configure-pod-container/share-process-namespace/)
* 参考 [Volume](/docs/reference/generated/kubernetes-api/{{< param "version" >}}/#volume-v1-core)
* 参考 [Pod](/docs/reference/generated/kubernetes-api/{{< param "version" >}}/#pod-v1-core)

