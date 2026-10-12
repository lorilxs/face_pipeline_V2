# 批次输出整合脚本使用说明

`consolidate_batch_outputs.sh` 用于把某个批次目录下分散的 pipeline 输出，汇总成三个统一的文件夹：`audio/`、`flame/`、`videos/`。

## 一、脚本做了什么

以批次目录下的**每个子文件夹（一个视频）**为单位，做「一一对应」匹配：

| 数据 | 来源路径（相对每个视频子文件夹） |
|------|----------------------------------|
| audio npy | `stage17_multimodal_export/audio/*.npy` |
| flame npy | `stage17_multimodal_export/flame/*.npy` |
| video mp4 | `verification/<clip_id>.mp4`（音视频合并后的视频，不含 `*_original.mp4`） |

### 整合规则（缺一不可）

- 一个视频**必须同时**存在 audio npy、flame npy、mp4 三类文件，才会被整合；
- 三类中**任意一类缺失**，该视频就**整组跳过**，不整合到最终文件夹；
- 一个视频如果对应**多个** audio npy / flame npy，**全部保留**；
- 使用 `cp` **复制**，原始分散文件保持不变（是否需要删除原始文件请自行处理）。

### 输出位置

```
<批次目录>/<批次名>_new_pipeline/
├── audio/    # 所有有效视频的 audio npy
├── flame/    # 所有有效视频的 flame npy
├── videos/   # 所有有效视频的 mp4
└── consolidate.log   # 本次运行日志（包含 SKIP/OK 明细和最终汇总）
```

其中 `<批次名>` 由你传入的目录名**自动推导**，无需手动改脚本。
例如目录叫 `HDTFxxxx`，输出就是 `HDTFxxxx/HDTFxxxx_new_pipeline/`。

## 二、使用方法

### 基本用法

```bash
bash /mnt/home/lxs/face_pipeline/Face_Pipeline_V2/scripts/consolidate_batch_outputs.sh <批次目录路径>
```

### 示例

```bash
# 处理 vivihead_batch_283_videos
bash /mnt/home/lxs/face_pipeline/Face_Pipeline_V2/scripts/consolidate_batch_outputs.sh \
  /mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/vivihead_batch_283_videos

# 下一个批次换成 vivihead_batch_881，只改参数即可
bash /mnt/home/lxs/face_pipeline/Face_Pipeline_V2/scripts/consolidate_batch_outputs.sh \
  /mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/vivihead_batch_881

# 名字变成 HDTFxxxx 或任意其它目录，同样只改参数
bash /mnt/home/lxs/face_pipeline/Face_Pipeline_V2/scripts/consolidate_batch_outputs.sh \
  /mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/HDTFxxxx
```

### 查看帮助

```bash
bash /mnt/home/lxs/face_pipeline/Face_Pipeline_V2/scripts/consolidate_batch_outputs.sh --help
```

## 三、如何换下一个批次

脚本里**没有任何写死的批次名或路径**，换批次只需改命令行里的路径参数：

1. 把路径改成新批次目录的完整路径（可以是相对路径）；
2. 输出子目录名会自动变成 `<新批次名>_new_pipeline`，不用额外修改。

## 四、运行日志怎么看

运行结束后会打印汇总，并同时写入 `<输出目录>/consolidate.log`：

- `OK    <视频名>`：三类文件齐全，已整合；
- `SKIP  <视频名>  [缺audio/缺flame/缺video]`：缺哪类就跳过，并标注原因；
- 最后的「汇总」段落给出：扫描子目录数、有效整合数、各类跳过数量、复制文件数、失败数。

## 五、注意事项

1. 脚本会自动跳过隐藏目录和 `*_new_pipeline` 目录，重复运行不会把上次的输出再当作视频目录处理；
2. 重复运行时，目标文件若已存在会被跳过（不会覆盖、不会重复计数），日志中记为「已存在跳过」；
3. 文件名带空格也能正常处理；
4. 脚本只做复制，不做删除；确认结果无误后，原始分散文件是否删除由你决定。
