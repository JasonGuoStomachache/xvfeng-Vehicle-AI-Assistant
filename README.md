# 基于商用LLM和MCP的车载智能AI助手（demo版）

<p align="center"> <img src="asset\flow.png" width="80%"> </p>

- 工作内容：基于商用大模型API、微调小型语义识别模型和MCP技术，构建了拥有400+技能的车载智能AI助手（demo）。通过研发无关语义拒识技术和双层意图识别技术，实现车载场景准确高效的对话和控制。
- 优化重点：通过双层意图识别，实现动态的Function Calling，首先利用一级微调模型召回Top-k(5)的备选意图，然后通过商用LLM进行第二级精准意图的匹配和参数槽位的抽取，避免大模型在海量工具库中盲目寻找导致的性能下降和消耗增加。
- 任务产出：在基于3层Bert-tiny微调的拒识模型上，无效语义的拒识率达90%+和400+的QPS；在基于Bert-large微调的一级意图召回模型上，Top1准确率达到85%，Top5准确率达到98%，并在结合商用大模型的二级意图召回中，准确率达到87%。并引入了导航、音乐等第三方MCP服务。

## 环境安装与变量导入
```bash
conda create -n agent_nlu python=3.12
conda activate agent_nlu
pip install -r requirements.txt
```

修改预设环境变量 (config/config.ini)：

```
# pass
export API_KEY="Bearer xxxx" # 豆包的api key
export BASE_URL="https://ark.cn-beijing.volces.com/api/v3/chat/completions"
export BOT_URL="https://ark.cn-beijing.volces.com/api/v3/bots/chat/completions"
export AMAP_MAPS_API_KEY="xxxx" #高德的api key

# 微服务
export REJECT_URL="http://127.0.0.1:8007/reject-server/v1"
export INTENT_URL="http://127.0.0.1:8008/intent-server/v1"
export NLU_URL="http://127.0.0.1:8009/chatnlu-server/v1"
export ENTRY_URL="http://127.0.0.1:8080/request_nlu"
```

加载预设环境变量
```
source config/config.ini
```

## 模型训练或模型

### 直接使用训练好的模型
下载[训练好的模型]()
bert_tiny.ckpt 为6层bert的拒识模型，放入saved\reject
bert.ckpt 为意图识别模型，放入saved\intent

### 自行训练
训练拒识/意图识别模型（train\run.py）
```
python train\run.py --model bert_tiny --data reject
```


## 启动服务
主程序默认服务端口**http://127.0.0.1:8080**
拒识模型默认服务端口**http://0.0.0.0:8007**
意图识别默认服务端口**http://0.0.0.0:8008**
nlu默认服务端口**http://0.0.0.0:8009**
redis默认 **standalone** 模式 port=6379
```
bash server.sh
```


### 本地对话系统

```
python dialog.py

>> connected to server
>> enter query:
>> 大 开左边的窗户
>> Resposnse:{ 'query': '打开左边的车窗', 'tarce_id': '9n4k316g7', 'intent': '打开车窗'， 'intent_id': '35', 'function': 'Open_Window', 'slots': {'位置': '左侧'}, 'cost': 0.9744772911071777 }
```


## 模块准确率和压力测试

### 模块准确率测试
单独测试拒识模型的准确率
```
python test\reject_client.py

>> 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1332/1332 [00:09<00:00, 136.24it/s]
>> test avg acc: 0.9114114114114115
```

单独测试意图识别模型的准确率（Top1）
```
python test\intent_client.py

>> 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7400/7400 [01:05<00:00, 112.54it/s]
>> test avg acc@1: 0.8572972972972973

```

单独测试nlu模块的准确率
```
python test\nlu_client.py

>> 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2212/2212 [36:42<00:00,  1.00it/s]
>> test intent acc: 0.8652802893309223 slots acc: 0.9159132007233273
```

### 模块压力测试
三个模块具体的压力测试结果 拒识模型QPS 400左右，意图识别模型QPS 200左右，nlu模块由于需要调用商用API QPS在1~2
（使用一块4090D测试得到）

压测复现代码 test\xxx_benchmark.py
具体日志详见 log\xxx.log 

### 全流程测试
test\data\multi_test.txt
```
python test.py
```

## 代码理解
### 主推理函数 ``start.py:inference``

```python
@socketio.on("request_nlu")
def inference(req):
    """
    处理客户端发送的自然语言请求的核心函数
    负责协调多个NLP服务（NLU、仲裁、拒识、闲聊等），并根据业务规则返回处理结果
    触发时机：客户端通过WebSocket发送"request_nlu"事件时
    
    参数:
        req: 客户端发送的JSON字符串，包含用户查询、用户标识等信息
    """
    # 记录请求处理开始时间，用于计算总耗时
    begin = time.time()
    # 解析客户端发送的JSON数据为字典
    json_info = json.loads(req)
    # 提取用户输入的查询文本（核心处理对象）
    query = json_info.get("query")
    # 提取是否启用对话管理的标识（用于上下文跟踪）
    enable_dm = json_info.get("enable_dm")
    # 提取用户唯一标识（默认值为"test"，用于区分不同用户）
    sender_id = json_info.get("sender_id", "test")
    # 提取追踪ID（默认值为"123"，用于日志串联和问题排查）
    trace_id = json_info.get("trace_id", "123")

    # 初始化NLU结果模板，统一返回格式
    nlu_template = {
        "query": query,               # 用户查询文本
        "tarce_id": trace_id,         # 追踪ID
        "intent": "",                 # 意图（后续根据处理结果填充）
        "intent_id": "",              # 意图ID（后续根据处理结果填充）
        "function": "",               # 功能类型（后续根据处理结果填充）
        "slots": {},                  # 槽位信息（后续根据处理结果填充）
        "cost": time.time() - begin,  # 处理耗时（实时更新）
    }
    try:
        # 保存原始查询文本（避免后续改写操作覆盖原始输入）
        ori_query = query
        # 将trace_id绑定到日志上下文，确保同一请求的日志可串联追踪
        logger.session.trace_id = trace_id
        # 打印请求参数日志，便于问题排查
        logger.info("Request Params: {}".format(json_info))

        # 从Redis获取用户上次交互的历史信息（用于上下文关联）
        # Redis键格式：voice:last_service:{sender_id}
        last_info = redis_client.get(REDIS_KEY.format(sender_id))
        # 初始化历史信息变量
        last_domain, last_query, last_reject, last_answer = "", "", "", ""
        if last_info:
            # 拆分历史信息（格式：领域#上次查询#上次拒识结果#上次回答）
            last_domain, last_query, last_reject, last_answer = last_info.split("#")

        # 调用查询改写服务（基于历史回答优化用户输入，如补全省略的上下文）
        query = request_rewrite(query, last_answer, sender_id)

        # 通过线程池并发提交多个NLP任务（并行处理提升效率）
        # 1. 调用NLU服务（解析语义：意图识别、槽位提取等）
        handler_nlu = thread_pool.submit(request_nlu, query, trace_id, enable_dm)
        # 2. 调用仲裁服务（决定处理路径：任务型/闲聊型等）
        handler_arbitration = thread_pool.submit(
            request_arbitration, ori_query, sender_id
        )
        # 3. 调用拒识模型（判断查询是否有效，如无意义字符串应被拒识）
        handler_reject = thread_pool.submit(request_reject, query, trace_id)
        # 4. 调用相关性模型（判断当前查询与历史查询是否相关）
        handler_correlation = thread_pool.submit(
            request_correlation, ori_query, sender_id
        )
        # 5. 调用闲聊/百科服务（作为兜底回复方案）
        handler_bot = thread_pool.submit(request_chat, ori_query, sender_id)

        # 获取仲裁服务结果（核心决策依据）
        arbitration_result = handler_arbitration.result()

        # 打印仲裁结果日志，包含耗时信息
        logger.info(
            f"TraceID:{trace_id}, query:{query}, arbitration result: {arbitration_result}, cost time: {time.time() - begin}"
        )

        # 分支1：仲裁结果为"task"（任务型请求，需调用具体技能处理）
        if arbitration_result == "task":
            # 获取NLU服务的解析结果
            nlu_result = handler_nlu.result()
            # 检查NLU是否识别到有效功能（非Unknown）
            if nlu_result.get("function", "") not in ["Unknown"]:
                # 将当前交互信息存入Redis（领域=SKILL，过期时间TTL=40秒）
                redis_client.set(
                    REDIS_KEY.format(sender_id), f"SKILL#{query}#1#", ex=TTL
                )
                # 向客户端发送NLU解析结果（包含意图、槽位等信息）
                emit(
                    "request_nlu",
                    json.dumps(nlu_result, ensure_ascii=False),
                    broadcast=False,
                )
            else:
                # NLU未识别到有效功能，发送拒识消息
                send_msg(
                    nlu_result,
                    "REJECT",
                    prompts.DEFAULT_NLG,  # 默认拒识回复内容
                    1,                   # 序列号
                    time.time() - begin, # 耗时
                    status=-1            # 状态：-1表示拒识
                )
                logger.info(f"Query {query} has been rejected.")
        
        # 分支2：仲裁结果非"task"（非任务型请求，如闲聊）
        else:
            # 获取拒识模型结果（0=应拒识，1=不应拒识）
            reject_result = handler_reject.result()
            # 若拒识模型判定为应拒识，但相关性模型判定与历史相关，则强制改为不应拒识
            if reject_result == 0:
                correlation_result = handler_correlation.result()
                if correlation_result == "是":
                    reject_result = 1
            
            # 子分支2.1：最终判定为应拒识
            if reject_result == 0:
                send_msg(
                    nlu_template,
                    "REJECT",
                    "",                  # 无具体内容
                    1,                   # 序列号
                    time.time() - begin, # 耗时
                    status=-1            # 状态：-1表示拒识
                )
                logger.info(f"Query {query} has been rejected.")
            
            # 子分支2.2：最终判定为不应拒识，调用闲聊服务处理
            else:
                # 处理闲聊逻辑（流式返回结果）
                is_hit_chat, full_answer = handle_chat(
                    handler_bot, nlu_template, ori_query, sender_id, begin
                )
                # 若闲聊服务成功返回结果，缓存到Redis
                if is_hit_chat:
                    redis_client.set(
                        REDIS_KEY.format(sender_id),
                        f"CHAT#{query}#{reject_result}#{full_answer}",  # 缓存格式：领域#查询#拒识结果#完整回答
                        ex=TTL  # 过期时间
                    )

    # 异常处理：捕获所有可能的错误，确保服务稳定
    except Exception as e:
        # 记录错误日志，包含trace_id便于追踪
        logger.error("TraceID:{}, Internal Server Error!".format(trace_id))
        logger.error("{}".format(e))
        # 打印堆栈跟踪，详细定位错误位置
        traceback.print_exc()
        # 发生异常时向客户端发送拒识消息，避免客户端长期等待
        send_msg(nlu_template, "REJECT", "", 1, time.time() - begin, status=-1)

```

### nlu 处理过程

<p align="center"> <img src="asset\nlu.png" width="80%"> </p>

`function_call\chatnlu_infer.py:inference,predict`
```python
@app.post("/chatnlu-server/v1")  # 定义POST接口路径
async def inference(request: Request):  # 异步接口函数，接收Request对象
    json_info = await request.json()  # 解析请求体中的JSON数据
    
    begin = time.time()  # 记录接口处理开始时间
    query = json_info.get("query")  # 获取用户查询文本
    enable_dm = json_info.get("enable_dm", True)  # 获取是否启用对话管理的标志（默认True）
    trace_id = json_info.get("trace_id", "1")  # 获取追踪ID（默认"1"）
    
    # 调用predict函数，抽取意图和槽位
    nlu = predict(query, trace_id)
    
    # 解析NLU结果（格式如"意图-槽位1:值1,槽位2:值2"）
    nlu_items = nlu.split("-")  # 分割为意图和槽位两部分
    intent = nlu_items[0]  # 提取意图（如"天气查询"）
    # 处理槽位部分（若槽位包含"-"符号，需特殊处理）
    if len(nlu_items) > 2:
        slots_str = "-".join(nlu_items[1:])  # 合并含"-"的槽位字符串
    else:
        slots_str = nlu_items[1]  # 普通槽位字符串
    
    # 解析槽位为字典格式
    if slots_str != "无":  # 若存在槽位
        slots = {}
        for item in slots_str.split(","):  # 按逗号分割多个槽位
            if ":" in item:  # 校验槽位格式（必须包含":"）
                if len(item.split(":")) != 2:  # 跳过格式错误的槽位
                    continue
                k, v = item.split(":")  # 分割槽位名和值
                slots[k] = v  # 添加到槽位字典
    else:
        slots = {}  # 无槽位时返回空字典
    
    # 获取意图ID和对应的函数名
    intent_id = name2id.get(intent)  # 根据意图名称查ID
    func_name = id2func.get(intent_id)  # 根据意图ID查函数名
    
    # 构造基础响应体
    response = {
        "query": query,  # 用户原始查询
        "tarce_id": trace_id,  # 追踪ID（注：此处有拼写错误，应为"trace_id"）
        "intent": intent,  # 意图名称
        "intent_id": intent_id,  # 意图ID
        "function": func_name,  # 对应的处理函数名
        "slots": slots,  # 槽位字典
    }
    
    # 若启用对话管理，调用对应领域的对话管理器
    if enable_dm:
        # 遍历需要处理的领域（天气、音乐、地图）
        for name in ["weather", "music", "maps"]:
            # 调用领域对应的对话管理器处理
            dm_result = await DMFactory.get(name)(func_name, query, slots)
            if dm_result:  # 若处理成功
                tool_response, nlg = dm_result  # 解包工具响应和自然语言生成结果
                response["tool"] = tool_response  # 添加工具响应到结果
                response["nlg"] = nlg  # 添加自然语言回复到结果
    
    # 计算总耗时并添加到响应
    cost = time.time() - begin
    response["cost"] = cost
    
    return response  # 返回最终处理结果
```

### 槽位抽取
function_call\slot_process.py
```python
def value_process(key, value):
    # 定义位置映射表：将中文位置描述转换为标准化英文标识
    position_map = {
        "主驾": "MAIN",
        "副驾": "VICE",
        "左侧": "LEFT",
        "右侧": "RIGHT",
        "前排": "FRONT",
        "后排": "REAR",
        # 更多位置映射...
    }
    
    # 处理数字/比例类型的槽位（如百分比、数值）
    if key in ["NUMBER", "RATIO"]:
        if "%" in value:  # 若包含百分号，转换为小数（如"50%" → 0.5）
            value = float(eval(value.replace("%", "")) / 100)
        else:  # 纯数字，直接转换为浮点数（如"25" → 25.0）
            value = float(eval(value))
    
    # 处理位置类型的槽位：统一转换为英文标识
    elif key in ["POSITION"]:
        value = position_map.get(value, value)  # 若没有匹配，保留原始值
    
    # 处理"对话时长"槽位：移除"秒"单位（如"30秒" → "30"）
    elif key == "对话时长":
        value = value.replace("秒", "")
    
    # 处理"Extreme"（极值）槽位：统一为"最大"或"最小"
    elif key == "Extreme":
        if value in ["最大", "最高", "最强", "最亮", "最热"]:
            value = "最大"  # 统一正向极值描述
        if value in ["最小", "最低", "最弱", "最暗", "最冷"]:
            value = "最小"  # 统一反向极值描述
    
    return value  # 返回转换后的标准化值

def intent_slot(function, map_intent, slot_map):
    try:
        # 1. 提取模型预测的函数名（原始意图标识）
        predict_e = function[0].get("function", {}).get("name", "NULL")
        
        # 2. 将函数名映射为用户可读的意图名称（如"set_temp" → "设置温度"）
        predict_z = map_intent.get(predict_e, predict_e)  # 若映射失败，保留原始值
        
        # 3. 提取槽位参数并解析为字典（模型返回的是JSON字符串）
        slots_predict = function[0].get("function", {}).get("arguments", "{}")
        slots_predict = json.loads(slots_predict)  # 转换为Python字典
        
        # 4. 构建结果字符串（格式："意图-槽位1:值1,槽位2:值2"）
        result = predict_z + "-"  # 先拼接意图名称和分隔符
        
        # 5. 根据槽位映射表（slot_map）转换槽位名，并标准化槽位值
        dict_slot = slot_map.get(predict_e)  # 获取当前意图对应的槽位映射规则
        if slots_predict:  # 若存在槽位参数
            for key, value in slots_predict.items():
                # 过滤无效值（空值、"不限"，或无映射规则的情况）
                if value and isinstance(dict_slot, dict) and value != "不限":
                    # 转换槽位名（如"pos" → "POSITION"）
                    key = dict_slot.get(key, key)
                    # 调用value_process标准化槽位值
                    value = value_process(key, value)
                    # 拼接槽位信息到结果字符串（如"POSITION:MAIN,"）
                    result = result + f"{key}:{str(value)}" + ","
                else:
                    continue  # 跳过无效槽位
            
            # 移除最后一个多余的逗号（如"意图-槽1:值1,槽2:值2," → "意图-槽1:值1,槽2:值2"）
            result = result.rsplit(",", 1)[0]
        
        # 6. 若没有有效槽位，补充"无"（如"意图-无"）
        if ":" not in result:
            result = result + "无"
    
    # 任何异常发生时，返回默认的"未知-无"
    except Exception as e:
        return "未知-无"
    
    return result  # 返回最终的"意图-槽位"字符串
```

### 拒识模型、意图识别模型服务启动
train\intent_infer.py or train\reject_infer.py
```python
def predict(query):
    with torch.no_grad():  # 禁用梯度计算，节省内存并加速推理
        # 1. 对输入文本进行分词处理
        token = config.tokenizer.tokenize(query)  # 将查询文本分割为词元（如"打开空调"→["打开", "空调"]）
        
        # 2. 添加上下文标记并处理序列长度
        token = [CLS] + token  # 在句首添加CLS标记（BERT模型要求的起始标记）
        seq_len = len(token)  # 记录当前序列长度
        
        # 3. 处理掩码（mask）和词元ID
        mask = []  # 用于标记有效词元（1）和填充词元（0）
        token_ids = config.tokenizer.convert_tokens_to_ids(token)  # 将词元转换为模型可识别的ID
        
        # 4. 序列填充或截断至固定长度（模型要求的输入长度）
        if len(token) < config.pad_size:  # 若序列长度小于指定长度，进行填充
            mask = [1] * len(token_ids) + [0] * (config.pad_size - len(token))  # 有效词元部分为1，填充部分为0
            token_ids += [0] * (config.pad_size - len(token))  # 用0填充词元ID
        else:  # 若序列长度超过指定长度，进行截断
            mask = [1] * config.pad_size  # 截断后所有位置均为有效词元
            token_ids = token_ids[: config.pad_size]  # 截断词元ID至指定长度
            seq_len = config.pad_size  # 更新序列长度为指定长度
        
        # 5. 转换为PyTorch张量并移动到指定设备（CPU/GPU）
        x = torch.LongTensor([token_ids]).to(config.device)  # 词元ID张量
        seq_len = torch.LongTensor([seq_len]).to(config.device)  # 序列长度张量
        mask = torch.LongTensor([mask]).to(config.device)  # 掩码张量
        
        # 6. 模型推理
        texts = (x, seq_len, mask)  # 打包输入数据
        output = model(texts)  # 模型输出（未经过softmax的原始logits）
        
        # 7. 计算概率并获取TopK结果
        prob = F.softmax(output, dim=-1).cpu().numpy()[0]  # 对输出应用softmax得到概率分布，并转移到CPU转为numpy数组
        index = np.argsort(-prob)[:TOPK]  # 按概率降序排序，取前TOPK个意图的索引
        
        return index, prob[index]  # 返回TopK意图的索引和对应的概率

@app.post("/intent-server/v1")  # 定义POST请求的接口路径
async def inference(request: Request):  # 异步接口函数，接收Request对象
    json_info = await request.json()  # 解析请求体中的JSON数据
    query = json_info.get("query")  # 获取用户输入的查询文本
    trace_id = json_info.get("trace_id")  # 获取用于追踪的唯一ID
    
    result = {}  # 初始化返回结果字典
    try:
        # 调用predict函数获取TopK意图的索引和概率
        response, score = predict(query)
    except:
        # 若推理过程出错，返回默认结果（意图3，概率1.0，共TOPK个）
        response, score = [3] * TOPK, [1.0] * TOPK
    
    # 格式化结果：将索引和概率转换为逗号分隔的字符串
    result["data"] = ",".join([str(k) for k in response])  # 意图索引字符串（如"0,1,2,3,4"）
    result["score"] = ",".join([str(k) for k in score])  # 概率字符串（如"0.9,0.05,0.03,0.01,0.01"）
    
    # 记录日志：包含追踪ID、查询文本、返回结果和置信度
    logger.info(
        "Trace ID: {}, Request: {}, response: {}, confidence: {}".format(
            trace_id, query, result["data"], result["score"]
        )
    )
    
    return result  # 返回处理结果
```

## Todo
- 一次对话识别多个意图，并关联执行（关上左边的窗户，并提高音量）
- 在百科闲聊流程中加入RAG，内置用户个人数据库、专业数据库（也有可能通过商用api实现）
- ask转换系统（目前只能手动输入query）
- 在更多算力设备上的压力测试

