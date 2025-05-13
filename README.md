<div align="center">
<h1>Kea</h1>

</div>

### 简介

Kea 是一个通用的测试工具，通过[基于性质的测试](https://en.wikipedia.org/wiki/Software_testing#Property_testing) 发现移动（GUI）应用中的功能性错误。

我们小组采用以下思路设计了我们改进过后的Kea

1. 基于RAG和Prompt工程，将背景、当前状态、任务传递给LLM

2. 直接使用LLM的RAW输出计算做某个动作的概率

3. 选择概率最大的行动，并根据运行结果给出一些reward或punish

4. 基于PPO算法的强化学习训练过程提升我们LLM解决问题的能力

   我们的PPO仓库：[Luinoa/PPOLLM](https://github.com/Luinoa/PPOLLM)

### 安装和使用

**环境配置**

- Python 3.11
- `adb` or `hdc` cmd tools available
- Connect an Android / HarmonyOS device or emulator to your PC
- CUDA（需要配合30系及以上显卡使用）

**安装**

输入以下命令安装 修改过后的Kea（主要修改了一些policy的细节）。

```bash
git clone https://github.com/Flora-Alex/Kea.git
cd Kea
pip install -e .
```

输入以下命令获取PPOLLM

```bash
git clone https://github.com/Luinoa/PPOLLM.git
pip install -r requirements.txt
```

再根据 shell 文件夹下的脚本，写出时候你的环境和需求的脚本，在根目录下运行，即可开始使用，对于参数的具体意义，请根据 api_server.py 下的描述进行使用，这里不再赘述。

如：

```bash
#!/bin/bash
# This script is used to run the inference server with large model.

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=4,5,6,7


python ./api_server.py \
-t \
-p 8001 \
--policy-minibatch-size 1 \
--model Qwen/Qwen3-8B \
--load-path weights/PPO
```

特别注意：

--policy-minibatch-size 内存极端敏感，谨慎调节。

--model 建议使用支持 Accelerate 框架多卡推理的 HuggingFace 模型，根据我们的实验，8B 模型需要约 50G 显存（至少 3 张 3090）进行微调。

部署过程中要是还缺什么，可以试试“缺啥补啥”。

**快速开始**

首先需要引入langchain需要的包，需要保证运行是处于联网状态下，否则RAG的retriever功能可能无法正常使用网络资源。

```bash
#Document loading, retrieval methods and text splitting 
%pip install -qU langchain langchain_community 
# Local vector store via Chroma 
%pip install -qU langchain_chroma 

%pip install -qU "unstructured[md]" nltk
# Web Loader
%pip install -qU beautifulsoup4
```

**注意：** 目前 LLM 模块仍处于实验阶段。我们正在积极收集反馈以改进该模块的功能和稳定性。感谢您的理解和支持，同时欢迎您提出建议和意见。

### 实验数据集

训练是基于运行kea检查apk对应的property的过程通过给予reward进行的，因此实验的数据集可以近似理解为不同的app

经过测试如果properties对应的py文件不适合，对应的覆盖率结果也有较大差距。因此选择在原kea仓库下properties文件夹中对应的apk与原KeaPlusEvaluation仓库下已经插桩的apk的交集，也就是四个apk分别是AmazeFileManger与AntennaPod，Omninotes，Ankidroid作为实验数据集，作为参考我们的实验是将其中的AmazeFileManger与AntennaPod作为训练集，而Omninotes，Ankidroid作为测试集。

输入以下命令获取我们的实验仓库

```
git clone https://github.com/StarMaster10/KeaPlusEvaluation.git
```

**实验要求：**

windows10以上系统（linux系统可以参照原仓库链接自行适配，仓库链接在我们的实验仓库ReadMe中提供）

虚拟机设备版本为Android11版本，其他版本无法正常进行覆盖度文件转储

**运行实验**

在仓库的scripts目录下运行

```
python ./themis.py --avd “” --apk "YOUR_APK_DIRECTORY" -f "YOU_PROPERTIES" -p "POLICY" --time "TIME"  -o  "OUTPUT_DIRECTORY"  --offset 0  --repeat 1
```

执行完成5次后，执行以下代码，其中OUTPUT_DIRECTORY同上述代码的OUTPUT_DIRECTORY

```
python coverage_diff_tool.py -dir “OUTPUT_DIRECTORY”
```

再执行

```
 python coverage_diff_tool_average.py -dir H:\KeaPlusEvaluation-main\KeaPlusEvaluation\KeaPlusEvaluation\scripts\output\b 
```

**注意事项**

我们的脚本是针对原KeaPlusEvaluation仓库下的run_hybirdDroid.ps1基础上重新增添了参数处理了逻辑增加对windows系统的适配性，但是依然存在如下问题：

1.原版repeat参数失效，并没有做相对应处理，经过一些不同方法试验后仍然不能很好的重复执行，目前仍只能单次执行。

2.原版提取广播器名称逻辑仍存在问题，尝试修改仍无效，目前只能手动获取，在run_hybirdDroid.ps1文件对应的$RECEIVER_NAME条目，先手动执行

```
adb -s $DEVICE_SERIAL shell pm dump $PACKAGE_NAME | Select-String "jacocoInstrument.SMSInstrumentedReceiver"
```

指令然后提取字符串中的对应参数形如

```
"it.feio.android.omninotes.alpha/it.feio.android.omninotes.jacocoInstrument.SMSInstrumentedReceiver"
```



### 作者

叶旭哲，罗剑波，王滨

### 现版本Kea 参考的开源工具

- [Droidbot](https://github.com/honeynet/droidbot)
- [HMDroidbot](https://github.com/ecnusse/HMDroidbot)
- [hypothesis](https://github.com/HypothesisWorks/hypothesis)
- [hmdriver2](https://github.com/codematrixer/hmdriver2)
- [uiautomator2](https://github.com/openatx/uiautomator2)
- [langchain](https://github.com/langchain-ai/langchain)

### 相关阅读

https://dl.acm.org/doi/pdf/10.1145/263244.263267

https://xyiheng.github.io//files/Property_Based_Testing_for_Android_Apps.pdf

https://ieeexplore.ieee.org/document/10638617

https://ylimit.github.io/static/files/DroidBot_ICSE2017.pdf

Property-Based Testing for Validating User Privacy-Related Functionalities in Social Media Apps. FSE 2024. [pdf](https://dl.acm.org/doi/10.1145/3663529.3663863)

Property-Based Fuzzing for Finding Data Manipulation Errors in Android Apps. ESEC/FSE 2023. [pdf](https://dl.acm.org/doi/10.1145/3611643.3616286)

Property-Based Testing in Practice. ICSE 2024. [pdf](https://dl.acm.org/doi/10.1145/3597503.3639581)