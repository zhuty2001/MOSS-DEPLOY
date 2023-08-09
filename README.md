# Moss Deploy
将moss框架部署到本地的教程

## 介绍
本文档完整介绍了将moss框架部署到本地GPU服务器上所需要的环境配置、部署过程、使用方法以及可能遇到的问题。

## 环境配置

### 键盘输入编码
由于在使用moss时需要频繁使用中文，且本文使用的是FinalShell连接服务器，需要将FinalShell的文字编码格式与本地系统的编码格式相匹配，否则会出现中文字符乱码的问题，以FinalShell和windows系统为例：

可以看到FinalShell默认的编码格式为UTF-8

![image](https://github.com/zhuty2001/moss_deploy/assets/68087747/304face5-0731-47fa-9e75-f82a05a2fbcf)

因此我们需要将windows系统本地环境中将键盘格式同样改为UTF-8，开始菜单->时间和语言->语言和区域->管理语言设置->更改系统区域设置，勾选以下选项并重启即可。

![image](https://github.com/zhuty2001/moss_deploy/assets/68087747/2dc6863f-b560-4497-af69-72a9071062a8)

### 本地环境
首先，本地部署时使用的适配环境是FinalShell,ssh远程连接GPU服务器，其本地环境为：
* Ubuntu 20.04
* cuda 12.2
* cudnn 8.9.3
* 四张 NVIDIA Corporation TU104GL [Tesla T4] (rev a1)

#### 1.查看本地环境版本
查看服务器是否有合适的GPU：

`
lspci | grep -i nvidia
`

查看系统版本：

`
uname -m && cat /etc/*release
`

验证系统GCC版本：

`
gcc --version
`
#### 2.CUDA安装
首先前往以下网址找到对应本地环境的CUDA版本：
https://developer.nvidia.com/cuda-downloads

根据本地环境，本文的版本选择如下：

![image](https://github.com/zhuty2001/moss_deploy/assets/68087747/491fee0e-eacc-4c8a-a4f8-759318c5497b)

在命令行中输入对应的命令下载相应的CUDA文件：

`
wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda_12.2.1_535.86.10_linux.run
`

禁用系统自带的显卡驱动，在命令行输入：

`
sudo touch /etc/modprobe.d/blacklist-nouveau.conf
`

`
sudo vim /etc/modprobe.d/blacklist-nouveau.conf
`

然后将下面的内容添加到/etc/modprobe.d/blacklist-nouveau.conf中并保存:

`
blacklist nouveau
options nouveau modeset=0
`

更新一下：

`
sudo update-initramfs -u
`

输出：

`
update-initramfs: Generating /boot/initrd.img-5.4.0-110-generic
`

完成上述步骤后需重启系统。

完成重启之后即可进行CUDA的安装，同样对应相应的CUDA版本输入以下指令进行安装：
`
sudo sh cuda_12.2.1_535.86.10_linux.run
`






