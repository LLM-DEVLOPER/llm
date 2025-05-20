参考链接：[https://zhuanlan.zhihu.com/p/713944891](https://zhuanlan.zhihu.com/p/713944891)（<font style="color:rgb(25, 27, 31);">Ascend NPU 之 llama.cpp</font>）(文中比较理想化，还有很多坑，可以参考配置llama.cpp环境)



## 第一章-华为310黑盒子适配
1.环境配置

先安装ubuntu系统环境和python环境，在此基础上开始安装CANN

```python
# CANN依赖包
pip3 install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions
# CANN包安装
# 需要重新下载安装kernel和tookit两个包
# 下载链接：https://www.hiascend.com/developer/download/community/result?module=cann
sh Ascend-cann-toolkit_8.0.RC2.alpha002_linux-aarch64.run --install
sh Ascend-cann-kernels-910b_8.0.RC2.alpha002_linux.run --install
# CANN环境变量配置
echo "source ~/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
source ~/.bashrc
```



测试使用的**华为310**盒子，内部已经安装了CANN，但是是5.0版本，去华为晟腾官网寻找新版本的CANN，未找到适配310芯片的版本，所以暂时未尝试重装CANN（推荐8.0以上，验证过7.0也不行）

拉取llama.cpp项目到本地

```python
git clone https://github.com/ggerganov/llama.cpp.git
```

进入llamacpp项目，开始编译项目

```python
cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release
cmake --build build --config release
```

此步骤出现报错：

CMake Error at ggml/src/ggml-cann/CMakeLists.txt:73 (message):

CANN: Can't find CANN_INSTALL_DIR, did you forget to source set_var.sh?

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1746698084990-71fe7d0d-7cda-40e8-ad43-a626455a5327.png)

根据报错找到了ggml/src/ggml-cann/CMakeLists.txt：

```python
if ("cann${CANN_INSTALL_DIR}" STREQUAL "cann" AND DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(CANN_INSTALL_DIR $ENV{ASCEND_TOOLKIT_HOME})
    message(STATUS "CANN: updated CANN_INSTALL_DIR from ASCEND_TOOLKIT_HOME=$ENV{ASCEND_TOOLKIT_HOME}")
endif()

# Auto-detech Soc type and Soc version, if detect failed, will abort build
set(SOC_VERSION "")
function(detect_ascend_soc_type SOC_VERSION)
    execute_process(
        COMMAND bash -c "npu-smi info|awk -F' ' 'NF > 0 && NR==7 {print $3}'"
        OUTPUT_VARIABLE npu_info
        RESULT_VARIABLE npu_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if("${npu_info}" STREQUAL "" OR ${npu_result})
        message(FATAL_ERROR "Auto-detech ascend soc type failed, please specify manually or check ascend device working normally.")
    endif()
    set(${SOC_VERSION} "Ascend${npu_info}" PARENT_SCOPE)
endfunction()

if(NOT SOC_TYPE)
    detect_ascend_soc_type(SOC_VERSION)
    set(SOC_TYPE "${SOC_VERSION}")
    message(STATUS "CANN: SOC_VERSION auto-detected is:${SOC_VERSION}")
endif()

string(TOLOWER ${SOC_TYPE} SOC_VERSION) # SOC_VERSION need lower

# Construct Soc specify compile option: ASCEND_#Soc_Major_SN. Such as ASCEND_910B, ASCEND_310P.
string(REGEX MATCH "[0-9]+[a-zA-Z]" SOC_TYPE_MAJOR_SN "${SOC_VERSION}")
set(SOC_TYPE_COMPILE_OPTION "ASCEND_${SOC_TYPE_MAJOR_SN}")
string(TOUPPER ${SOC_TYPE_COMPILE_OPTION} SOC_TYPE_COMPILE_OPTION)

if (CANN_INSTALL_DIR)
    # Only Support Linux.
    if (NOT UNIX)
        message(FATAL_ERROR "CANN: CANN toolkit supports unix but not ${CMAKE_SYSTEM_NAME}")
    endif()

    # Supported platforms: x86-64, arm64
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
    else()
        message(FATAL_ERROR "CANN: CANN toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    # Set header and libs
    set(CANN_INCLUDE_DIRS
        ${CANN_INSTALL_DIR}/include
        ${CANN_INSTALL_DIR}/include/aclnn
        ${CANN_INSTALL_DIR}/acllib/include
    )

    list(APPEND CANN_LIBRARIES
        ascendcl
        nnopbase
        opapi
        acl_op_compiler
    )

    file(GLOB GGML_SOURCES_CANN "*.cpp")

    ggml_add_backend_library(ggml-cann ${GGML_SOURCES_CANN})
    target_link_libraries(ggml-cann PRIVATE ${CANN_LIBRARIES})
    target_include_directories(ggml-cann PRIVATE ${CANN_INCLUDE_DIRS})
    target_link_directories(ggml-cann PRIVATE ${CANN_INSTALL_DIR}/lib64)

    target_compile_definitions(ggml-cann PRIVATE "-D${SOC_TYPE_COMPILE_OPTION}")

    message(STATUS "CANN: CANN_INCLUDE_DIRS =  ${CANN_INCLUDE_DIRS}")
    message(STATUS "CANN: CANN_LIBRARIES =  ${CANN_LIBRARIES}")
else()
    message(FATAL_ERROR "CANN: Can't find CANN_INSTALL_DIR, did you forget to source set_var.sh?")
endif()

```

根据代码来看，是未找到CANN_INSTALL_DIR，这个变量是由ASCEND_TOOLKIT_HOME决定的，所以尝试手动设置环境变量

```python
export ASCEND_TOOLKIT_HOME=/home/HwHiAiUser/Ascend/ascend-toolkit
```

然后再次尝试build环境，再次报错：

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1746698313044-5f26fe86-6950-4f26-a2d0-dfa50506b82a.png)

根据提示发现5.0的CANN中没有aclnn工具，无法完成编译。

后续可能尝试在310上安装新版本CANN框架。



如**编译过程中遇报错：**

 -- Could NOT find CURL (missing: CURL_LIBRARY CURL_INCLUDE_DIR) CMake Error at common/CMakeLists.txt:92 (message): Could NOT find CURL. Hint: to disable this feature, set -DLLAMA_CURL=OFF  

没有找到curl工具，建议直接卸载掉curl重新安装

```python
sudo apt remove --purge libcurl4 libcurl4-openssl-dev
sudo apt update
sudo apt install libcurl4-openssl-dev
```

但执行此操作可能造成系统无法找到cmake工具，因此也需要重新安装

```python
wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-aarch64.tar.gz
tar -xzf cmake-3.27.9-linux-aarch64.tar.gz
cd cmake-3.27.9-linux-aarch64
./bin/cmake --version
export PATH=$PWD/bin:$PATH

```



**注意：CANN 7.0不支持int4类型，编译llama.cpp报错，把这行代码注释掉也不行，建议直接升级8.0**：

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1747116333260-9cebc5ff-a5f1-40c0-803c-fa74a2ce554c.png)

## 第二章-华为310b4香橙派盒子适配
这个盒子安装的CANN是8.0版本，按照上述教程下载llamacpp并编译。

安装pytorch环境（npu上配置torch环境）：

[https://gitee.com/ascend/pytorch/tree/v2.4.0/](https://gitee.com/ascend/pytorch/tree/v2.4.0/)

编译完成后，通过modelscope下载模型文件

```python
pip install modelscope
```

```python
python
from modelscope.models import Model
#开始下载0.5b的gguf模型，内包含多个量化版本
model = Model.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct-GGUF')
```



问题：Ascend 310B型号芯片不支持RMSNorm，也就不能跑量化版本的Qwen2.5

参考：[https://www.hiascend.com/forum/thread-02102160455229791049-1-1.html](https://www.hiascend.com/forum/thread-02102160455229791049-1-1.html)



查阅资料发现qwen-1.5-0.5B模型可能没有使用到rmsnorm，随下载这个模型进行试验

下载的原始模型

```python
>>> from modelscope.models import Model
>>> model = Model.from_pretrained('Qwen/Qwen1.5-0.5B-Chat')
```

利用llama-cpp对原始模型进行量化操作：

-先转换模型格式

```python
python convert_hf_to_gguf.py /root/.cache/modelscope/hub/models/Qwen/Qwen1.5-0.5B-Chat/ --outfile my_model/qwen1.5-0.5b-chat-fp16.gguf
```

执行成功

再进行量化操作：

```python
./build/bin/llama-quantize ./my_model/qwen1.5-0.5b-chat-fp16.gguf ./my_model/qwen1.5-0.5b-chat-fp16-
q4_0.gguf q4_0
```

执行成功

然后尝试离线推理：

```python
./build/bin/llama-cli -m /home/HwHiAiUser/llama.cpp/my_model/qwen1.5-0.5b-chat-fp16-q4_0.gguf -p "Building a website can be done in 10 simple times:" -ngl 80
```

报错：

```python
build: 5307 (814f795e) with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for aarch64-linux-gnu
main: llama backend init
main: load the model and apply lora adapter, if any
llama_model_load_from_file_impl: using device CANN0 (Ascend310B4) - 1551 MiB free
llama_model_loader: loaded meta data with 30 key-value pairs and 291 tensors from /home/HwHiAiUser/llama.cpp/my_model/qwen1.5-0.5b-chat-fp16-q4_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen1.5 0.5B Chat
llama_model_loader: - kv   3:                           general.finetune str              = Chat
llama_model_loader: - kv   4:                           general.basename str              = Qwen1.5
llama_model_loader: - kv   5:                         general.size_label str              = 0.5B
llama_model_loader: - kv   6:                            general.license str              = other
llama_model_loader: - kv   7:                       general.license.name str              = tongyi-qianwen-research
llama_model_loader: - kv   8:                       general.license.link str              = https://huggingface.co/Qwen/Qwen1.5-0...
llama_model_loader: - kv   9:                               general.tags arr[str,2]       = ["chat", "text-generation"]
llama_model_loader: - kv  10:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv  11:                          qwen2.block_count u32              = 24
llama_model_loader: - kv  12:                       qwen2.context_length u32              = 32768
llama_model_loader: - kv  13:                     qwen2.embedding_length u32              = 1024
llama_model_loader: - kv  14:                  qwen2.feed_forward_length u32              = 2816
llama_model_loader: - kv  15:                 qwen2.attention.head_count u32              = 16
llama_model_loader: - kv  16:              qwen2.attention.head_count_kv u32              = 16
llama_model_loader: - kv  17:                       qwen2.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  18:     qwen2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  19:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  20:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  21:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  22:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  23:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  24:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  25:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  26:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  27:                    tokenizer.chat_template str              = {% for message in messages %}{% if lo...
llama_model_loader: - kv  28:               general.quantization_version u32              = 2
llama_model_loader: - kv  29:                          general.file_type u32              = 2
llama_model_loader: - type  f32:  121 tensors
llama_model_loader: - type q4_0:  169 tensors
llama_model_loader: - type q6_K:    1 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_0
print_info: file size   = 371.02 MiB (5.02 BPW) 
load: special tokens cache size = 3
load: token to piece cache size = 0.9308 MB
print_info: arch             = qwen2
print_info: vocab_only       = 0
print_info: n_ctx_train      = 32768
print_info: n_embd           = 1024
print_info: n_layer          = 24
print_info: n_head           = 16
print_info: n_head_kv        = 16
print_info: n_rot            = 64
print_info: n_swa            = 0
print_info: n_swa_pattern    = 1
print_info: n_embd_head_k    = 64
print_info: n_embd_head_v    = 64
print_info: n_gqa            = 1
print_info: n_embd_k_gqa     = 1024
print_info: n_embd_v_gqa     = 1024
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 2816
print_info: n_expert         = 0
print_info: n_expert_used    = 0
print_info: causal attn      = 1
print_info: pooling type     = -1
print_info: rope type        = 2
print_info: rope scaling     = linear
print_info: freq_base_train  = 1000000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 32768
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 0.5B
print_info: model params     = 619.57 M
print_info: general.name     = Qwen1.5 0.5B Chat
print_info: vocab type       = BPE
print_info: n_vocab          = 151936
print_info: n_merges         = 151387
print_info: BOS token        = 151643 '<|endoftext|>'
print_info: EOS token        = 151645 '<|im_end|>'
print_info: EOT token        = 151645 '<|im_end|>'
print_info: PAD token        = 151643 '<|endoftext|>'
print_info: LF token         = 198 'Ċ'
print_info: EOG token        = 151643 '<|endoftext|>'
print_info: EOG token        = 151645 '<|im_end|>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: offloading 24 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 25/25 layers to GPU
load_tensors:        CANN0 model buffer size =   165.85 MiB
load_tensors:   CPU_Mapped model buffer size =   205.18 MiB
..............................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 4096
llama_context: n_ctx_per_seq = 4096
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 0
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (4096) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
ggml_backend_cann_context: device 0 async operator submission is OFF
llama_context:  CANN_Host  output buffer size =     0.58 MiB
llama_kv_cache_unified: kv_size = 4096, type_k = 'f16', type_v = 'f16', n_layer = 24, can_shift = 1, padding = 32
llama_kv_cache_unified:      CANN0 KV buffer size =   384.00 MiB
llama_kv_cache_unified: KV self size  =  384.00 MiB, K (f16):  192.00 MiB, V (f16):  192.00 MiB
/home/HwHiAiUser/llama.cpp/ggml/src/ggml-backend.cpp:2002: GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned") failed
```

这个错误我使用直接下载的量化模型也报过，暂时未能解决。

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1746778508097-67a7a376-1771-466f-abd3-fd5bb50316d3.png)

chatgpt解决方案：

```python
已思考 1m 34s


下面先简要总结：您在使用 `llama-server` 调用 Ascend 310B4（CANN 后端）加载 Qwen 1.5 0.5B Chat 模型时，遇到

```
GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned") failed
```

的断言失败。这源自 CANN 后端在分配“Host 缓冲区”时，没有使用对齐（pinned）内存而退回到普通 malloc，导致指针未满足 NPU 所需的 `TENSOR_ALIGNMENT` 对齐要求，从而触发断言。
针对这一问题，主要有以下几种思路可以尝试：

## 问题原因分析

### 1. CANN 后端默认未使用 HostBufferType

* 在早期版本的 `llama.cpp` 中，Ascend NPU（CANN）后端并不支持“pin memory”（Host buffer type），因此会退回到使用普通的 CPU 缓冲区来存放中间结果 ([GitHub][1])。
* 普通的 `malloc` 或 `new` 返回的内存通常只保证 16 或 32 字节对齐，但 CANN 后端要求的 `TENSOR_ALIGNMENT`（通常为 64 或更高）并未满足，导致在构造张量时触发：

  ```c
  GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned") failed
  ```

  ([GitHub][2])。

### 2. 断言触发流程

* `llama-server` 加载模型后，会为 KV 缓存和输出缓冲分配 Host 内存；
* 若该分配未使用对齐 alloc，则指针地址 `% TENSOR_ALIGNMENT != 0`，在 `ggml-backend.cpp` 第 2002 行触发上述断言。

## 解决方案

### 1. 升级至包含 HostBufferType 支持的版本

* 社区已在 PR #9406 中为 CANN 后端加入了对 “Host buffer type” 的支持，内部改为调用 Ascend CANN 提供的 `aclrtMallocHost` 或等效接口，从而保证分配的 Host 内存满足 NPU 对齐要求 ([SemanticDiff][3])。
* 操作步骤：

  1. `git pull` 或将本地代码切换到包含该 PR 的提交；
  2. 重新 `make clean && make`；
  3. 再次运行 `llama-server`。

### 2. 临时禁用 CANN，加回 CPU 后端

若短期内无法升级，可在构建时关闭 CANN 后端，让所有计算走纯 CPU 路径（无需对齐），方法是在 CMake 时添加：

```bash
cmake -B build -S . -DGGML_USE_CANN=OFF …  
make -C build -j
```

这样可绕过该断言，但会牺牲 NPU 加速。

### 3. 手动实现内存对齐分配

* 在 `ggml-backend-cann.cpp`（或相关 CANN 后端实现）中，用 `posix_memalign(&ptr, TENSOR_ALIGNMENT, size)` 替换默认的 `malloc` ([Stack Overflow][4], [Stack Overflow][5])；
* 或直接调用 CANN SDK 的 `aclrtMallocHost(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST)` 等接口，以获得已对齐且 pin 的内存。

### 4. 其他可选的 Workaround

* **减小上下文长度**：通过 `-c` 参数缩小 `n_ctx`，减少 KV 缓存大小，或许能避开一次性大块分配不对齐的情况 ([GitHub][6])（SYCL 后端类似思路）；
* **使用新版 GGUF 模型格式**：新 GGUF 格式在 mmap 加载时强制 32 字节对齐，可避免文件映射相关对齐问题 ([LibreTranslate Community][7])；
* **重新量化**：用最新的 `convert_hf_to_gguf.py` 重新生成模型，确保内部 tensor 排布满足对齐 ([GitHub][8])。

---

通过上述方式，可从底层分配、后端构建选项或模型格式三方面入手，修复或绕过 “buffer pointer must be aligned” 的断言，恢复 Ascend 310B4 的正常推理加速。若后续仍有问题，建议在更新到最新 `llama.cpp` 后，结合 CANN SDK 文档调整对齐策略，或在社区 Issue 中进一步讨论。

[1]: https://github.com/ggerganov/llama.cpp/issues/9304?utm_source=chatgpt.com "Add Host buffer type for Ascend NPU (CANN backend ... - GitHub"
[2]: https://github.com/ggerganov/llama.cpp/issues/9786?utm_source=chatgpt.com "[CANN] llama-server support Ascend CANN ？ · Issue #9786 - GitHub"
[3]: https://app.semanticdiff.com/gh/ggml-org/llama.cpp/commit/e6b7801bd189d102d901d3e72035611a25456ef1?utm_source=chatgpt.com "cann: Add host buffer type for Ascend NPU (#9406) - SemanticDiff"
[4]: https://stackoverflow.com/questions/1649398/how-do-i-ensure-buffer-memory-is-aligned?utm_source=chatgpt.com "How do I ensure buffer memory is aligned? - Stack Overflow"
[5]: https://stackoverflow.com/questions/61970675/c-alignment-when-casting-byte-buffer-to-another-type?utm_source=chatgpt.com "C++: Alignment when casting byte buffer to another type"
[6]: https://github.com/ggerganov/llama.cpp/issues/6861?utm_source=chatgpt.com "crash on llama_new_context_with_model: failed assertion `Buffer ..."
[7]: https://community.libretranslate.com/t/llama-cpp-is-now-100x-faster/559?utm_source=chatgpt.com "Llama.cpp is now 100x faster - General - LibreTranslate Community"
[8]: https://github.com/ggerganov/llama.cpp/issues/8660?utm_source=chatgpt.com "Bug: [SYCL] GGML_ASSERT Error with Llama-3.1 SYCL Backend ..."

```

我采纳了其中减小max_length的方式，尝试设置为很小的2，发现真的可以运行起来：

启动命令：

```python
./build/bin/llama-server -m /home/HwHiAiUser/llama.cpp/my_model/qwen1.5-0.5b-chat-fp16-q4_0.gguf --host 0.0.0.0 --port 8080 -c 32

```

运行结果：

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1746781334269-485c3bbb-d447-437c-8ccc-49fb4b720b2b.png)

在Ascend 310B4单块8G内存背景下，尝试了最大可以设置为32

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1747038633076-82f3a100-6c6b-40b2-ad7a-be0f692a8bf1.png)



**后续计划：**

**1.仍然使用llama.cpp的部署方式（优点：资源占用小，可以cpu+npu同步推理，迁移性好）**

**（！310原始芯片暂不支持，起码CANN5.0是不支持llama.cpp框架，仅在310b4+CANN8.0上跑通）**

—如果希望在npu上增强推理精度，可以考虑不直接使用暴力的直接量化过程，使用llama.cpp本身的量化并没有考虑到激活量化，造成的精度损失很大。因此，可以考虑使用精度损失更小的AWQ or GPTQ量化，使用一定量的校准数据集纠正量化过程的精度损失

—还要尝试一下，设置不同cpu和npu分配比例下，模型的性能问题

**2.直接部署onnx格式模型（优点：参考资料较多）**

目前网上搜到的关于npu部署大模型的资料，基本都是这样直接部署的方式，这种方式应该也可以适配310芯片



llama.cpp启动服务用到的一些命令：

```python
./build/bin/llama-cli -m /home/HwHiAiUser/llama.cpp/my_model/qwen1.5-0.5b-chat-fp16-q4_0.gguf -p "Building a website can be done in 10 simple times:" -ngl 80

python convert_hf_to_gguf.py /root/.cache/modelscope/hub/models/Qwen/Qwen1.5-0.5B-Chat/ --outfile my_model/qwen1.5-0.5b-chat-fp16.gguf

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ./build/bin/llama-server -m /home/HwHiAiUser/llama.cpp/my_model/qwen1.5-0.5b-chat-fp16-q4_0.gguf --host 0.0.0.0 --port 8080

```

