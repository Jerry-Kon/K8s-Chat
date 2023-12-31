
生成控制器管理器要使用的 kubeconfig 文件


### 概要


生成控制器管理器要使用的 kubeconfig 文件，并保存到 controller-manager.conf 文件中。

```
kubeadm init phase kubeconfig controller-manager [flags]
```


### 选项

   <table style="width: 100%; table-layout: fixed;">
<colgroup>
<col span="1" style="width: 10px;" />
<col span="1" />
</colgroup>
<tbody>

<tr>
<td colspan="2">--apiserver-advertise-address string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
<p>API 服务器所公布的其正在监听的 IP 地址。如果未设置，则使用默认的网络接口。</p>
</td>
</tr>

<tr>
<td colspan="2">
--apiserver-bind-port int32&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;默认值：6443
</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
<p>要绑定到 API 服务器的端口。</p>
</td>
</tr>

<tr>
<td colspan="2">
--cert-dir string&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;默认值："/etc/kubernetes/pki"
</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
<p>保存和存储证书的路径。</p>
</td>
</tr>

<tr>
<td colspan="2">--config string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
<p>kubeadm 配置文件的路径。</p>
</td>
</tr>
<tr>
<td colspan="2">--control-plane-endpoint string</td>
</tr>

</tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
<p>为控制平面指定一个稳定的 IP 地址或 DNS 名称。</p>
</td>
</tr>    

<tr>
<td colspan="2">--dry-run</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
<p>不做任何更改；只输出将要执行的操作。</p>
</td>
</tr>

<tr>
<td colspan="2">-h, --help</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;"> 
<p>controller-manager 操作的帮助命令</p>
</td>
</tr>

<tr>
<td colspan="2">
--kubeconfig-dir string&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;默认值："/etc/kubernetes"
</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
<p>kubeconfig 文件的保存路径。</p>
</td>
</tr>

<tr>
<td colspan="2">
--kubernetes-version string&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;默认值："stable-1"
</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
<p>为控制平面指定特定的 Kubernetes 版本。</p>
</td>
</tr>
</tbody>
</table>


### 继承于父命令的选项

   <table style="width: 100%; table-layout: fixed;">
<colgroup>
<col span="1" style="width: 10px;" />
<col span="1" />
</colgroup>
<tbody>

<tr>
<td colspan="2">--rootfs 字符串</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
<p>[实验] 到 '真实' 主机根文件系统的路径。</p>
</td>
</tr>

</tbody>
</table>

