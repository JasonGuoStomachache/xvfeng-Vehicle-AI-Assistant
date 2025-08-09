# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©2025丁师兄大模型
# 生成时间: 2025-05
# --------------------------------------------


import json
import asyncio
from datetime import datetime
from client.nlg import request_nlg
from mcp_core.music_server import search_music


async def process(func_name, query, slots):

    if func_name not in ["Search_Music"]:
        return

    # 调用MCP接口
    if not slots:
        slots["genre"] = "流行"

    formated_slots = {
        "keyword": " ".join(list(slots.values()))
    }

    tool_response = await search_music(formated_slots["keyword"])
    nlg = request_nlg(query, tool_response)

    return (tool_response, nlg)




