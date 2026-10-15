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

## 预训练

项目使用 `model/tokenizer.json` 与 `model/tokenizer_config.json` 中的 6400 词表分词器。预训练数据是每行含有 `text` 字段的 JSONL 文件。`dataset/debug_pretrain.jsonl` 是仅用于检查训练链路的小样本；正式数据 `dataset/pretrain_t2t_mini.jsonl` 来自 [MiniMind 数据集](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/main/pretrain_t2t_mini.jsonl)，不会提交到 Git。

从项目根目录运行：

```bash
source activate.sh
python trainer/train_pretrain.py --data_path dataset/debug_pretrain.jsonl --hidden_size 64 --num_hidden_layers 2 --max_seq_len 32 --batch_size 2 --accumulation_steps 1 --num_workers 0
```

小样本跑通后，开始正式预训练：

```bash
python trainer/train_pretrain.py --max_seq_len 320 --batch_size 2 --accumulation_steps 8
```

默认从头训练，权重写入 `out/`，续训状态写入 `checkpoints/`。中断后用相同的模型参数并添加 `--from_resume 1` 继续。需要改用自己的 JSONL 数据时，传入 `--data_path`。

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
