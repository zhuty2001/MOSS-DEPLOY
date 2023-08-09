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

### 

