# 数据污染检测

## 背景介绍

数据污染（data contamination）指用于评价和测试的数据被包括在了语言模型的训练阶段，造成虚假的测试结果[1]。该问题源于机器学习领域对训练集和测试集进行切分长期这一原则，因此长期存在。在大语言模型流行的当下，数据污染受到了更多的关注，原因有二：一是大语言模型的训练数据规模比以往更大，更有可能包含人们可用的测试数据；二是很多模型不再公开其完整的训练数据，模型的不透明性增加了。数据污染检测，即判断数据是否被用于模型训练，能让我们更公平地比较不同模型的能力，也能更好地衡量单个模型的泛化性。

数据污染可以发生在语言模型的各个训练阶段和不同任务场景中，本次作业中我们考虑一种比较简单的设定：我们选取了某个权重开源的预训练模型（Transformer架构，类似于GPT的自回归式语言模型），收集了一批最新发布的新闻文章以保证该模型在之前的预训练阶段没有“见过”这些数据。然后我们从收集的新闻中随机选取一部分，让该模型在这部分数据上继续预训练，即在这些数据上进行“预测下一词”的任务。最后，我们将数据进一步切分成验证集和测试集。你需要根据我们训练后的模型参数以及带有标签的验证集来预测测试集中每条数据是否被用于模型训练。

## 作业目标

- 本项目作业允许组队，队伍人数不超过3人。
- 给定一个待检测模型，设计一套数据污染检测方法，用于预测测试集`test.json`中每条文本数据是否被用于该模型的训练。
- 每支队伍在课堂上用PPT讲解技术实现和体会。

### 数据
我们在`dataset`文件夹中提供了如下数据：

- `valid.json`：验证集，同时包括了数据与标签。
- `test.json`：测试集，只包括了文本数据。本次作业的目标即为这部分数据上进行打分预测，分数越高表示该文本是被污染数据的可能性越高。


### 环境配置
- Python: 3.10
- 安装torch（请根据系统的CUDA版本安装对应的torch，详情见 https://pytorch.org/get-started/previous-versions/ ，建议更新CUDA以安装最新版本的torch）
- 安装其他依赖项
  ```
  pip install transformers
  pip install scikit-learn
  ```

### 模型加载

使用`transformers`从`detected_model`文件夹中加载待检测模型。
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("detected_model")
model = AutoModelForCausalLM.from_pretrained("detected_model", torch_dtype=torch.bfloat16)
```
 
具体的模型使用细节请参考[transformers文档](https://huggingface.co/docs/transformers/model_doc/opt#transformers.OPTForCausalLM)。

### 评价指标

数据污染检测本质上是一个二分类问题，我们使用AUC作为评测指标，详情请见[scikit-learn文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)。



## 提交内容和截止日期

请于6月2日晚上12:00前在教学网上提交以下内容。请统一打包为zip文件在教学网上提交，命名格式为：第x组-项目作业.zip，如第1组-项目作业.zip


- 数据污染检测的相关代码
- 算法和实验结果的分析文档（PDF，格式不限），如人数超过1人请在文档中说明成员分工
- 在测试集上的预测结果`test.json`。请为`test.json`文件中所有`text`进行打分，分数越高表示该文本是被污染数据的可能性越高，并将分数填写在`score`字段中，也即提交的`test.json`文件格式为：
    ```
    [
        {
            'text': text1,
            'score': score1
        },
        {
            'text': text2,
            'score': score2
        },
        ...
    ]

6月4日上午，每支队伍在课堂上用PPT讲解技术实现和体会，展示PPT可在当天提交。命名格式为：第x组展示.pptx，如第1组展示.pptx

小组内一位同学提交即可。

## 说明

- 检测方法不限。可以考虑使用模型对于每条数据的困惑度（[perplexity](https://huggingface.co/docs/transformers/perplexity)）、对比模型在同义改写后的文本的行为是否有显著差异、以模型的隐层和文本为特征训练分类模型、让模型续写等等。
- 参考文献中的方法不做强制要求。



## 参考文献

[1] Anon. 2024. [Unveiling the Spectrum of Data Contamination in Language Model: A Survey from Detection to Remediation](https://openreview.net/pdf?id=2V5IzVkbXr). *Under Review*. OpenReview.net

[2] Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer. 2024. [Detecting Pretraining Data from Large Language Models](https://openreview.net/pdf?id=zWqr3MQuNs). In *ICLR*. OpenReview.net

[3] Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Schoelkopf, Mrinmaya Sachan, and Taylor Berg-Kirkpatrick. 2023. [Membership Inference Attacks against Language Models via Neighbourhood Comparison](https://aclanthology.org/2023.findings-acl.719.pdf). In *ACL (Findings)*. Association for Computational Linguistics.

[4] Shahriar Golchin, Mihai Surdeanu. 2024. [Time Travel in LLMs: Tracing Data Contamination in Large Language Models](https://arxiv.org/pdf/2308.08493.pdf). In *ICLR*. OpenReview.net

[5] Kent Chang, Mackenzie Cramer, Sandeep Soni, David Bamman. 2023. [Speak, Memory: An Archaeology of Books Known to ChatGPT/GPT-4](https://aclanthology.org/2023.emnlp-main.453.pdf). In *EMNLP*. Association for Computational Linguistics.

