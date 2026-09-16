# QQZ-minimind

跟随 MiniMind 教程，亲手实现小型语言模型的学习仓库。逐步实现模型、数据集、训练循环和文本生成，目前从空白 Notebook 开始。

## 本机启动环境

```bash
cd /root/autodl-fs/wjq_self_mini_gpt/QQZ-minimind
source activate.sh
```

使用已经安装并验证过的独立环境 `/root/autodl-tmp/envs/minimind`：Python 3.10、PyTorch 2.6.0（CUDA 12.4）、Transformers 4.57.6、Datasets 3.6.0，以及 Notebook 依赖。这个目录和旁边的 `MiniMind` 参考目录使用同一个 MiniMind 环境。

## 开始写代码

在 VS Code 中打开本机的 `QQZ-minimind.code-workspace`（文件 → 从文件打开工作区），即可使用本目录的解释器配置。若此前手动选过其他解释器，请用“Python: Select Interpreter”选择 `/root/autodl-tmp/envs/minimind/bin/python`。工作区配置和 `activate.sh` 仅用于本机，不提交到 Git。

你可以自由创建 `.py` 文件，或在空白的 `selfminimind.ipynb` 里跟课。Notebook 内核选择 **MiniMind (Python 3.10)**。

视频入口：[第 5 集：架构图解读](https://www.bilibili.com/video/BV1T2k6BaEeC/?p=5)。

## 在新机器上创建环境

```bash
conda env create -f environment.yml
conda activate qqz-minimind
python -m ipykernel install --user --name qqz-minimind --display-name "QQZ-minimind (Python 3.10)"
```

新机器上打开 Notebook 后选择新注册的内核。

## Git 使用

默认分支为 `main`。每完成一小节，检查改动并提交：

```bash
git status
git diff
git add README.md selfminimind.ipynb
# 新建了 Python 文件时，把需要提交的文件名也加到 git add 后面。
git diff --cached
git commit -m "Implement RMSNorm"
git push
```

首次上传需配置 `origin` 并执行 `git push -u origin main`。本机待完成的身份和远程设置见 `GIT_SETUP.local.md`，该说明文件不提交。

已忽略数据、模型权重、训练输出、缓存、环境和本机编辑器配置。Notebook 输出会随文件提交，提交前检查并按需清空。

参考：[MiniMind 官方项目](https://github.com/jingyaogong/minimind)、[视频配套 MokioMind](https://github.com/Wood-Q/MokioMind)。复用上游代码时保留来源和相应许可证要求。
