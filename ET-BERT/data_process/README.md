## Description of processing PCAP files to generate dataset
## PCAP 文件处理生成数据集的说明

For PCAP data, it is recommended to clean it first. Since the program processing logic is not smooth, we detail the data pre-processing for pre-training and fine-tuning as followed.
对于 PCAP 数据，建议先进行清洗。由于程序处理逻辑不够完善，我们详细说明了预训练和微调的数据预处理步骤。

### Pre-training Stage
### 预训练阶段

*Main Program*: dataset_generation.py
*主程序*: dataset_generation.py

*Functions*: pretrain_dataset_generation, get_burst_feature
*功能*: pretrain_dataset_generation, get_burst_feature

1. Initialization. 
1. 初始化
Set the variable `pcap_path` (line:616) as the directory of PCAP data to be processed. 
设置变量 `pcap_path`（第616行）为要处理的 PCAP 数据目录。
Set the variable `word_dir` (line:23) and `word_name` (line:24) as the storage directory of pre-training daraset.
设置变量 `word_dir`（第23行）和 `word_name`（第24行）为预训练数据集的存储目录。

2. Pre-process PCAP. 
2. 预处理 PCAP
Set the variable `output_split_path` (line:583) and `pcap_output_path` (line:584). 
设置变量 `output_split_path`（第583行）和 `pcap_output_path`（第584行）。
The `pcap_output_path` indicates the storage directory where the pcapng format of PCAP data is converted to pcap format. 
`pcap_output_path` 表示将 pcapng 格式的 PCAP 数据转换为 pcap 格式的存储目录。
The `output_split_path` represents the storage directory for PCAP data slicing into session format. 
`output_split_path` 表示将 PCAP 数据切分为会话格式的存储目录。

3. Gnerate Pre-training Datasets. 
3. 生成预训练数据集
Following the completion of PCAP data processing, the program generates a pre-training dataset composed of BURST.
完成 PCAP 数据处理后，程序生成由 BURST 组成的预训练数据集。

### Fine-tuning Stage
### 微调阶段

*Main Program*: main.py
*主程序*: main.py

*Functions*: data_preprocess.py, dataset_generation.py, open_dataset_deal.py, dataset_cleanning.py
*功能*: data_preprocess.py, dataset_generation.py, open_dataset_deal.py, dataset_cleanning.py

The key idea of the fine-tuning phase when processing public PCAP datasets is to first distinguish folders for different labeled data in the dataset, then perform session slicing on the data, and finally generate packet-level or flow-level datasets according to sample needs.
微调阶段处理公共 PCAP 数据集的关键思想是首先区分数据集中不同标签数据的文件夹，然后对数据进行会话切分，最后根据样本需求生成数据包级别或流级别的数据集。

**Note:** Due to the complexity of the possible existence of raw PCAP data, it is recommended that the following steps be performed to check the code execution when it reports an error.
**注意：** 由于原始 PCAP 数据可能存在复杂性，建议在代码执行报错时按以下步骤检查：

1. Initialization. 
1. 初始化
`pcap_path`, `dataset_save_path`, `samples`, `features`, `dataset_level` (line:28) are the basis variables, which represent the original data directory, the stored generated data directory, the number of samples, the feature type, and the data level. `open_dataset_not_pcap` (line:215)  represents the processing of converting PCAP data to pcap format, e.g. pcapng to pcap. 
`pcap_path`、`dataset_save_path`、`samples`、`features`、`dataset_level`（第28行）是基本变量，分别表示原始数据目录、生成数据存储目录、样本数量、特征类型和数据级别。`open_dataset_not_pcap`（第215行）表示将 PCAP 数据转换为 pcap 格式的处理，例如 pcapng 转 pcap。
And `file2dir` (line:226) represents the generation of category directories to store PCAP data when a pcap file is a category. 
`file2dir`（第226行）表示当一个 pcap 文件是一个类别时，生成类别目录来存储 PCAP 数据。

2. Pre-process. 
2. 预处理
The data pre-processing is primarily to split the PCAP data in the directory into session data. 
数据预处理主要是将目录中的 PCAP 数据切分为会话数据。
Please set the `splitcap_finish` parameter to 0 to initialize the sample number array, and the value of `sample` set at this time should not exceed the minimum number of samples. 
请将 `splitcap_finish` 参数设置为 0 以初始化样本数量数组，此时设置的 `sample` 不应超过最小样本数。
Then you can set `splitcap=True` (line:54) and run the code for splitting PCAP data. The splitted sessions will be saved in `pcap_path\\splitcap`.
然后可以设置 `splitcap=True`（第54行）并运行代码进行 PCAP 数据切分。切分后的会话将保存在 `pcap_path\\splitcap` 中。

3. Generation. 
3. 生成
After data pre-processing is completed, variables need to be changed for generating fine-tuned training data. The `pcap_path` should be the path of splitted data and set 
完成数据预处理后，需要更改变量以生成微调训练数据。`pcap_path` 应该是切分数据的路径，并设置
`splitcap=False`. Now the `sample` can be unrestricted by the minimum sample size. The `open_dataset_not_pcap` and `file2dir` should be False. Then the dataset for fine-tuning will be generated and saved in `dataset_save_path`. 
`splitcap=False`。现在 `sample` 可以不受最小样本数量的限制。`open_dataset_not_pcap` 和 `file2dir` 应该为 False。然后将生成微调数据集并保存在 `dataset_save_path` 中。
