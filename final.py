#!/usr/bin/env python
# coding: utf-8

# In[2]:


# !pip install asyncpg
# !pip install redis


# # 초기세팅

# ## 모듈 임포트

# In[3]:


from typing import List, Dict, Optional, TypedDict, Any
from fastapi import FastAPI , Body ,HTTPException, APIRouter, Query, Request
from fastapi.responses import FileResponse
import redis.asyncio as redis
import torch
import ast
import json
import uuid
import numpy as np
# import psycopg2
import asyncio
import asyncpg
from pgvector.asyncpg import register_vector
# from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import re
import platform 
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, 
    Spacer, Table, TableStyle, NextPageTemplate, PageBreak, FrameBreak
)
import traceback
from contextlib import asynccontextmanager
from reportlab.graphics.shapes import Drawing, Line
import os
from langgraph.graph import StateGraph, END
from pathlib import Path



# ## 모델 로딩

# ### 재미나이 로딩

# In[4]:


# API 키 로딩
keys = []
def load_api_keys(filepath="api_key.txt"):
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=", 1)
                keys.append(value)

# API 키 로드 및 환경변수 설정
load_api_keys('api_key.txt')


# In[5]:


def keyChange():
    global client
    from google import genai
    tt = keys.pop(0)
    keys.append(tt)
    client = genai.Client(api_key=tt)

keyChange() #무료 키 다썻으면 다른 키로 로딩하기


# ### 백터DB & 임베딩 모델 로딩

# In[18]:


# 장치 설정 및 모델 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_vector = SentenceTransformer('jhgan/ko-sroberta-multitask', device=device)

# 전역 변수 초기화
pool = None 

# DB 정보 로드
def get_db_info():
    db_info = dict()
    with open("db_info.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=", 1)
                db_info[key.strip()] = value.strip() # 공백 제거 추가
    return db_info

db_info = get_db_info()

# pgvector 등록 및 풀 생성 설정
async def init_connection(conn):
    await register_vector(conn)

async def create_pool():
    # 전역 pool 객체 생성 및 반환
    return await asyncpg.create_pool(
        host=db_info['host'],
        database=db_info['database'],
        user=db_info['user'],
        password=db_info['password'],
        port=db_info['port'],
        min_size=1,
        max_size=20,
        init=init_connection
    )

# 커넥션 획득 함수 (기존 로직 유지용)
async def get_db_connection(pool):
    if pool is None:
        raise Exception("Database pool is not initialized")
    return await pool.acquire()



# ## State

# In[7]:


class SettingState(TypedDict, total = False):
    txt: str  #front
    ep_num : int  #front 
    subtitle : str #front
    check : dict
    work_id : int    #front
    user_id : str    #front
    인물 : dict
    세계 : dict
    장소 : dict
    사건 : dict
    물건 : dict
    집단 : dict



# # 기능

# ## 백터 DB

# ### 임베딩 백터 : 택스트 데이터를 임베딩 벡터로 바꿔서 반환(768차원)

# In[8]:


def get_vector(text):
    return model_vector.encode([text])[0]


# ### 백터 DB 검색(유사도)  
# (카테고리 or *(전부), 유저 질문, user_id, work_id, 유사도 몇까지 출력할지, 몇개 출력할지)

# In[13]:


async def userQuestion(category,user_query,user_id,work_id,sim,limit):  # 태그(*,인물,장소,사건,세계,등)
    query_vector = get_vector(user_query)
    search_sql = """
SELECT * FROM (
    SELECT 
        id,
        category,
        setting, 
        ep_num,
        1 - (embedding <=> $1::vector) AS similarity 
    FROM lorebooks 
    WHERE embedding IS NOT NULL 
        AND $2 = ANY(user_id)
        AND work_id = $3
        AND ($4 = '*' OR category = $4)
) sub
WHERE similarity >= $5
ORDER BY similarity DESC  
LIMIT $6;
"""
    async with pool.acquire() as conn:
        result = await conn.fetch(search_sql, query_vector,user_id,work_id,category,sim,limit)

    return result


# In[14]:


# a =  await userQuestion("장소","산소가 권력인 장소","eZHXUSS8",28,0.3,5)
# for i in a:
#     print(i['setting'])


# ### DB 검색(일반)  
# (카테고리, 검색할 대상, user_id, work_id, 유사도 몇까지 출력할지, 몇개 출력할지)

# In[10]:


# def select(category,keyword,user_id,work_id): #태그,키워드,타이틀,작가 4개 다있어야함
#     conn = get_db_connection()
#     cur = conn.cursor()
#     search_sql= """
#     SELECT 
#         id,
#         setting,
#         ep_num
#     FROM lorebooks 
#     WHERE %s = ANY(user_id)
#         AND work_id = %s
#         AND category = %s
#         AND keyword = %s
#     """
#     cur.execute(search_sql, (user_id,work_id,category,keyword))
#     result = cur.fetchall()
#     # print(result)
#     cur.close()
#     conn.close()
#     return result
async def select(category, keyword, user_id, work_id):
    search_sql = """
    SELECT 
        id,
        setting,
        ep_num
    FROM lorebooks 
    WHERE $1 = ANY(user_id)
        AND work_id = $2
        AND category = $3
        AND keyword = $4
    """

    async with pool.acquire() as conn:
        result = await conn.fetch(
            search_sql,
            user_id,   # $1
            work_id,   # $2
            category,  # $3
            keyword    # $4
        )

    return result


# In[12]:


# await select("장소","밸러스트","eZHXUSS8",28)


# ### DB 업로드(설정집 신규)

# In[13]:


async def lorebooks_insert(universe_id, work_id, user_id, setting):
    # 중복되는 루프가 밖에도 있고 안에도 있으니 정리하는 것이 좋습니다.
    # SQL 구문을 $ 형식으로 수정
    insert_sql = """
        INSERT INTO lorebooks (universe_id, work_id, user_id, keyword, category, ep_num, setting, embedding)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """
    
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                for tag, values in setting.items():
                    for val in values:
                        if not val:
                            continue

                        # .pop()은 원본 데이터를 변경하므로 루프 안에서 조심해야 합니다.
                        ep_num = val.pop('ep_num', [])
                        keyword, data = val.popitem()
                        
                        # 벡터 변환
                        embd = get_vector(f'{keyword} : {data}')

                        await conn.execute(
                            insert_sql,
                            universe_id,        # $1
                            work_id,            # $2
                            [user_id],          # $3 (ARRAY)
                            keyword,            # $4
                            tag,                # $5
                            ep_num,             # $6 (ARRAY)
                            json.dumps({keyword: data}, ensure_ascii=False), # $7
                            embd                # $8
                        )
        print("성공적으로 저장되었습니다!")
        return "성공적으로 저장되었습니다!"
    except Exception as e:
        print(f"상세 에러 내용: {e}") # 여기서 구체적인 에러가 보일 겁니다.
        return "500 Internal Server Error"


# 수동 신규 설정집 업로드
# 1. 인자에 'pool'을 추가해서 다른 함수들과 통일성
async def db_insert(pool, universe_id, work_id, user_id, keyword, category, ep_num, setting):
    # 1. 임베딩 벡터 생성
    setting_str = json.dumps(setting, ensure_ascii=False)
    embd = get_vector(setting_str) 

    # 2. SQL 쿼리 작성
    insert_sql = """
    INSERT INTO lorebooks (
        universe_id, 
        work_id, 
        user_id, 
        keyword, 
        category, 
        ep_num, 
        setting, 
        embedding,
        created_at,
        updated_at
    )
    VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8,
        CURRENT_TIMESTAMP,
        CURRENT_TIMESTAMP
    )
    """

    try:
        # 3. 전달받은 pool을 사용해서 DB에 안전하게 접속
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    insert_sql,
                    universe_id,
                    work_id,
                    [user_id] if isinstance(user_id, str) else user_id, # ARRAY 처리
                    keyword,
                    category,
                    [ep_num] if isinstance(ep_num, int) else ep_num,    # ARRAY 처리
                    setting_str, # JSONB
                    embd         # vector
                )
            print(f"성공적으로 저장되었습니다! (키워드: {keyword})")
        return "성공"

    except Exception as e:
        print(f"저장 실패: {e}")
        return f"실패: {e}"


# ### DB 수정(설정집 결합)

# In[14]:


async def lorebooks_update(setting):
    # 1. 쿼리 정의 ($ 파라미터 바인딩 사용 확인)
    update_sql = """
    UPDATE lorebooks 
    SET 
        ep_num = $1,
        setting = $2,
        embedding = $3
    WHERE id = $4;
    """
    
    try:
        # 2. 커넥션은 한 번만 획득
        async with pool.acquire() as conn:
            # 3. 트랜잭션 시작 (자동 롤백/커밋 지원)
            async with conn.transaction():
                # 4. 루프는 트랜잭션 안에서 한 번만 수행
                for lore_id, result, ep_num in setting:
                    new_vector = f"{result}"
                    embedding_vector = get_vector(new_vector)

                    await conn.execute(
                        update_sql,
                        ep_num,                                # $1
                        json.dumps(result, ensure_ascii=False), # $2
                        embedding_vector,                      # $3
                        lore_id                                # $4
                    )
        print("데이터가 성공적으로 수정되었습니다.")
        return "성공"
        
    except Exception as e:
        # asyncpg의 transaction 컨텍스트를 사용하면 수동 rollback() 호출 불필요
        print(f"업데이트 중 에러 발생: {e}")
        return f"수정 실패: {e}"


async def db_update(pool, lore_id, universe_id, work_id, user_id, keyword, category, ep_num, setting):
    try:
        # 설정 내용을 JSON 문자열로 변환
        new_vector_text = json.dumps(setting, ensure_ascii=False)
        # 벡터 임베딩 생성
        embd = get_vector(new_vector_text)

        # SQL 쿼리 준비
        update_sql = """
        UPDATE lorebooks 
        SET 
            universe_id = $1,
            work_id = $2,
            user_id = $3,
            keyword = $4,
            category = $5,
            ep_num = $6,
            setting = $7,
            embedding = $8,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $9;
        """

        # 2. 전달받은 pool을 사용해서 DB에 접속
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    update_sql,
                    universe_id,          # $1
                    work_id,              # $2
                    [user_id] if isinstance(user_id, str) else user_id, # $3 (ARRAY 처리)
                    keyword,              # $4
                    category,             # $5
                    [ep_num] if isinstance(ep_num, int) else ep_num,    # $6 (ARRAY 처리)
                    new_vector_text,      # $7 (JSONB)
                    embd,                 # $8 (vector)
                    lore_id               # $9
                )
        return "성공"
        
    except Exception as e:
        print(f"DB Update Error: {e}")
        return f"수정 실패: {str(e)}"


# In[ ]:





# ### 연관인물 검색

# In[15]:


async def relationship_select(target, work_id, user_id):

    if target != '*':

        search_sql = """
        SELECT 
            jsonb_build_object(
                l.keyword, relation
            ) AS matching_relation
        FROM 
            lorebooks l,
            jsonb_each(l.setting) AS person,
            jsonb_array_elements(person.value->'인물관계') AS relation
        WHERE 
            l.work_id = $1
            AND $2 = ANY(l.user_id)
            AND (relation->>'대상이름' = $3 OR relation->>'대상' = $4)
            AND l.category = '인물'
        """

        async with pool.acquire() as conn:
            result = await conn.fetch(
                search_sql,
                work_id,   # $1
                user_id,   # $2
                target,    # $3
                target     # $4
            )

        # select도 async이므로 await 필요
        meinfo_raw = await select("인물", target, user_id, work_id)

        meinfo = []

        # for문 생략 없이 명시
        for row in meinfo_raw:
            # row는 asyncpg.Record
            setting_data = row["setting"]

            # JSONB면 dict, TEXT면 문자열이므로 분기
            if isinstance(setting_data, str):
                import json
                setting_data = json.loads(setting_data)

            if target in setting_data:
                relations = setting_data[target].get("인물관계", [])
                meinfo.append({target: relations})

        # result도 Record → dict 변환
        final_result = []

        for r in result:
            final_result.append(dict(r))

        return final_result + meinfo

    else:

        search_sql = """
        SELECT 
            jsonb_build_object(
                l.keyword, relation
            ) AS matching_relation
        FROM 
            lorebooks l,
            jsonb_each(l.setting) AS person,
            jsonb_array_elements(person.value->'인물관계') AS relation
        WHERE 
            l.work_id = $1
            AND $2 = ANY(l.user_id)
            AND l.category = '인물'
        """

        async with pool.acquire() as conn:
            result = await conn.fetch(
                search_sql,
                work_id,   # $1
                user_id    # $2
            )

        final_result = []

        for r in result:
            final_result.append(dict(r))

        return final_result


# ### 사건검색

# In[16]:


async def timeline_select(target, work_id, user_id):

    search_sql = """
    SELECT 
        id,
        ep_num,
        keyword
    FROM 
        lorebooks 
    WHERE 
        work_id = $1
        AND $2 = ANY(user_id)
        AND ep_num && $3
        AND category = '사건'
    ORDER BY ep_num[1] ASC;
    """

    async with pool.acquire() as conn:
        result = await conn.fetch(
            search_sql,
            work_id,  # $1
            user_id,  # $2
            target    # $3 (ARRAY)
        )

    final_result = []

    for row in result:
        final_result.append(dict(row))

    return final_result


# ## LLM 

# ### 메시지 입력

# In[17]:


count_change= 0
models = ["gemini-2.5-flash-lite", 'gemini-3-flash',"gemini-2.5-flash","gemma-3-27b-it","gemini-3-flash-preview","gemini-2.5-pro",
         "gemini-2.5-flash-preview-09-2025","gemini-2.5-flash-lite-preview-09-2025",]
def modelchange():
    global mod
    mod = models.pop(0)
    models.append(mod)
modelchange()    
async def genai(system_prompt, msg):
    global count_change

    loop = asyncio.get_running_loop()

    try:
        if mod == "gemma-3-27b-it":
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=mod,
                    contents=system_prompt + msg,
                )
            )
        else:
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=mod,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1,
                    ),
                    contents=msg
                )
            )

        return response.text

    except Exception:
        count_change += 1
        if count_change >= 30:
            modelchange()
            count_change = 0
        keyChange()
        return await genai(system_prompt, msg)


# ### 카테고리 추출

# In[18]:


async def Categories(settingstate: SettingState) -> SettingState:
    system_prompt="""### Role
너는 소설 원문에서 설정 데이터를 누락 없이 추출하여 구조화하는 전문 데이터 분석가입니다.

### Task
당신은 '본문'의 글에서 설정집을 생성하기 쉽도록 키워드를 추출합니다. 
추출할 정보는 '인물, 세계, 장소, 사건, 물건, 집단'에 관한 내용이며, 설정집에 필요한 필수적인 정보들만 추출할 것입니다.
이를 딕셔너리{} 형태로 구조화하세요.

### Extraction Guidelines (카테고리별 정의)
- **인물** 
이름이 언급되거나 고유한 역할이 있는 등장인물(단순 엑스트라 제외)
- **세계**
마법 체계, 정치 시스템, 신화적 배경, 물리적 법칙 등 세계관을 이해하기에 필수적인 설정과 개념
- **장소**
사건이 벌어지는 지리적 배경, 건물 이름, 특정 지역
- **사건**
스토리에 영향을 끼칠 수 있는 구체적인 변화, 전투, 조약, 과거의 사건 등
- **물건**
고유 명칭이 있는 도구, 무기, 유물, 중요한 상징적 아이템
- **집단**
가문, 국가, 단체 등 2인 이상의 고유명칭을 가진 결사체

### Constraints
- 특정 주인공 중심의 요약을 하지 마세요. 등장하는 키워드를 분류하세요.
- 결과물에는 불릿 포인트, 볼드체, 추가 설명을 절대 포함하지 말고 순수 JSON/Dict 형식만 출력하세요.
- **본문에 명확히 나오지 않은 정보는 빈 배열 `[]`로 처리하세요.
- **수식어 제거**: 대상을 특정하기 위한 수식어(예: 'oooo의 정치체계','oo의 악마','oo왕국 기사단','oooo 제국')를 제외한 수식어는(예: '비시르 함장' X, '비시르' O) 모두 제거하고 핵심 키워드만 추출하세요.
- **개체명 표준화: 동일 대상을 지칭하는 이명(예: '몰락한 왕'과 '비에고','비에고 국왕'과 '비에고' > 두 경우 다 '비에고'로 출력)은 가장 공식적인 명칭 하나로 통합하세요.

### Output Format 
{
'인물':['인물1','인물2',...],
'세계':['규칙1','규칙2',...],
'장소':['장소1','장소2',...],
'사건':['사건1','사건2',...],
'물건':['물건1','물건2'....],
'집단':['집단1','집단2',...],
}

"""
    msg = f"본문: {settingstate.get('txt','')} \n 위 본문을 분석하여 '인물, 세계, 장소, 사건, 물건, 집단' 정보를 추출해줘."

    result = await genai(system_prompt,msg)
    print("목록 추출 완료")
    print("=="*8)
    print(result)
    print("=="*8)
    try:
        json_str = result[result.find('{'): result.rfind('}') + 1]
        result = json.loads(json_str)
    except:
        print("딕셔너리 변환 에러")
        # state = Categories(state)
    return {**settingstate,
            'check' : result
           }


# In[ ]:





# ### 1.설정 추출 - 인물

# In[19]:


async def charter(settingstate: SettingState) -> SettingState:
    lis = settingstate.get('check')
    lis = lis.get('인물',[])
    txt = settingstate.get('txt')
    chart = settingstate.get('인물',{})
    for i in range(0, len(lis), 2):
        chunk = lis[i:i+2]  # 예: ['인물1', '인물2']
        names_str = ", ".join(chunk)
        system_prompt = f"""# Role
귀하는 소설의 원문을 분석하여 인물 설정집(Character Profile)을 생성하는 전문 편집자이자 데이터 분석가입니다.

# Task
제공된 소설 본문을 읽고, 미리 추출한 등장인물명 [{names_str}] 각각의 정보를 분석하여 지정된 JSON 포맷으로 출력하세요. 
새로운 정보가 발견될 때마다 기존 설정과 대조할 수 있도록 매우 구체적이고 객관적인 수치와 묘사 위주로 추출해야 합니다.
각 인물별로 독립된 키를 가진 딕셔너리 형태로 출력해야 합니다
본문에서 해당 정보가 있다면 None값을 삭제하고 정보를 str형식으로 적으세요
# 본문
{txt}

# extraction_rules
1. **Fact-Based**: 추측하지 말고 본문에 명시된 텍스트 그대로를 기반으로 추출하세요.
"""+"""   
# JSON_Output_Format
{
   "인물명1":{
      "별명":[
         "본문에 명시된 별명 (없으면 None)"
      ],
      "배경":[
         "간략한 배경 스토리(없으면 None)"
      ],
      "종족":[
         "종족 (없으면 None)"
      ],
      "연령":[
         "나이 또는 연령대 (없으면 None)"
      ],
      "직업/신분":[
         "신분 (없으면 None)"
      ],
      "소속집단/가문":[
         "소속 (없으면 None)"
      ],
      "외형":[
         "외모 묘사 (없으면 None)"
      ],
      "성격":[
         "성격 키워드1 (없으면 None)"
      ],
      "기술/능력":[
         {
            "기술명":"능력 이름 (없으면 None)",
            "숙련도/강도":"수준 (없으면 None)",
            "상세효과":"효과설명 (없으면 None)"
         }
      ],
      "인물관계":[
         {
            "대상이름":"상대방 이름 (없으면 None)",
            "관계":"관계 정의 (없으면 None)",
            "상세내용":"관계 설명 (없으면 None)"
         }
      ],
      "핵심 결핍":[
          "절실히 원하는 것 (없으면 None)",
      ],
      "내적 갈등":[
          "자신의 신념 고민 (없으면 None)",
      ],
      "외적 갈등":[
          "타인과의 상호작용 (없으면 None)",
      ],
      "대사":[
          "인물의 상징적인 대사 (없으면 None)",
      ],
      "행동 패턴":[
          "인물의 습관 (없으면 None)",
      ],
   },
   "인물명2":{
      "별명":[
         "본문에 명시된 별명 (없으면 None)"
      ],
      "배경":[
         "간략한 배경 스토리(없으면 None)"
      ],
      "종족":[
         "종족 (없으면 None)"
      ],
      "연령":[
         "나이 또는 연령대 (없으면 None)"
      ],
      "직업/신분":[
         "신분 (없으면 None)"
      ],
      "소속집단/가문":[
         "소속 (없으면 None)"
      ],
      "외형":[
         "외모 묘사 (없으면 None)"
      ],
      "성격":[
         "성격 키워드1 (없으면 None)"
      ],
      "기술/능력":[
         {
            "기술명":"능력 이름 (없으면 None)",
            "숙련도/강도":"수준 (없으면 None)",
            "상세효과":"효과설명 (없으면 None)"
         }
      ],
      "인물관계":[
         {
            "대상이름":"상대방 이름(없으면 None)",
            "관계":"관계 정의( 없으면 None)",
            "상세내용":"관계 설명(없으면 None)"
         }
      ],
      "핵심 결핍":[
          "절실히 원하는 것 (없으면 None)",
      ],
      "내적 갈등":[
          "자신의 신념 고민 (없으면 None)",
      ],
      "외적 갈등":[
          "타인과의 상호작용 (없으면 None)",
      ],
      "대사":[
          "인물의 상징적인 대사 (없으면 None)",
      ],
      "행동 패턴":[
          "인물의 습관 (없으면 None)",
      ],
   }
}

# Constraints
- 응답은 반드시 순수한 JSON 형식이어야 하며, 다른 설명 문구는 포함하지 마세요.
- 만약 등장인물이 하나만 있을 경우에는 한 개체만 출력하세요.
- 제공한 등장인물 2명을 제외한 나머지는 추가적으로 출력하지 마세요.
- 인물관계에 소속집단이 들어가는 등 잘못된 정보가 들어가지 않도록 주의하세요."""

        msg = f"규칙을 바탕으로 순수한 Json 텍스트 형식을 마크다운 없이 출력하세요."
        result = await genai(system_prompt,msg)
        try:
            result = result.replace("json", "").replace("```", "").strip()
            print(result)

            result = json.loads(result)
            print("변환 성공!")
            for k, v in result.items():
                chart[k] = v
        except:
            print("딕셔너리 변환 에러")
            print(result)
            #state = charter(state)


    return {**settingstate,
           '인물' :chart
           }


# In[ ]:





# ### 2.설정추출 - 세계관

# In[20]:


async def rule(settingstate: SettingState) -> SettingState:
    lis = settingstate.get('check')
    lis = lis['세계']
    txt = settingstate.get('txt')
    rule = settingstate.get('세계',{})
    for i in range(0, len(lis), 4):
        chunk = lis[i:i+4]  # 예: ['인물1', '인물2']
        names_str = ", ".join(chunk)
        system_prompt = f"""# Role
귀하는 소설의 원문을 분석하여 세계관 설정집을 생성하는 전문 편집자이자 데이터 분석가입니다.

# Task
제공된 소설 본문을 읽고, 미리 추출한 세계관 키워드 [{names_str}] 각각의 정보을/를 본문 없이 작품 속 세계의 규칙을 이해할 수 있도록 300자 이내로 설명하는 문장을 만들어 JSON 포맷으로 출력하세요.

# 본문
{txt}

# extraction_rules
1. **Fact-Based**: 요약은 가능하지만 본문의 내용을 바탕으로 추측 없이 문장을 생성하세요.
2. 연관된 이야기 없이 대상에 대한 내용만으로 짧게 요약하여 문장을 생성하세요.
3. 종류는 마법 체계, 정치 시스템, 신화적 배경, 물리적 법칙 등 어떤 부분을 설명하는지를 말합니다. 종류에는 값을 하나만 넣으세요.
"""+"""   
# JSON_Output_Format
{
   "세계관 키워드1":{
      "종류":[
         "세계관 종류(없으면 None)"
      ],
      "설명":[
         "세계관 1 설명(없으면 None)"
      ],
      "규칙":[
         "물리적/사회적 규칙 (없으면 None)"
      ],
      "금기":[
         "금기시되는 것(없으면 None)"
      ],
      "필수 제약":[
         "필수불가결한 제약(없으면 None)"
      ],
      "분위기":[
         "세계관의 분위기(없으면 None)"
      ]
   },
   "세계관 키워드2":{
      "종류":[
         "세계관 종류(없으면 None)"
      ],
      "설명":[
         "세계관 2 설명(없으면 None)"
      ],
      "규칙":[
         "물리적/사회적 규칙 (없으면 None)"
      ],
      "금기":[
         "금기시되는 것(없으면 None)"
      ],
      "필수 제약":[
         "필수불가결한 제약(없으면 None)"
      ],
      "분위기":[
         "세계관의 분위기(없으면 None)"
      ]
   }
}

# Constraints
- 응답은 반드시 순수한 JSON 형식이어야 하며, 다른 설명 문구는 포함하지 마세요.
- 본문에 명확히 나오지 않은 정보는 None값 또는 빈 배열 `[]`로 처리하세요.
- 만약 세계관 키워드가 하나만 있을 경우에는 한 세계관만 출력하세요.
- 제공한 키워드를 제외한 나머지는 추가적으로 출력하지 마세요.
"""

        msg= f"규칙을 바탕으로 순수한 Json 텍스트 형식을 마크다운 없이 출력하세요."           
        result = await genai(system_prompt,msg)
        try:
            result = result.replace("json", "").replace("```", "").strip()
            print(result)

            result = json.loads(result)
            print("변환 성공!")
            for k, v in result.items():
                rule[k] = v
        except:
            print("딕셔너리 변환 에러")
            print("result")
            #state = charter(state)

    return {**settingstate,
           '세계' :rule
           }


# ### 3.설정추출 - 장소

# In[21]:


async def place(settingstate: SettingState) -> SettingState:
    lis = settingstate.get('check')
    lis = lis['장소']
    txt = settingstate.get('txt')
    place = settingstate.get('장소',{})
    for i in range(0, len(lis), 3):
        chunk = lis[i:i+3]  
        names_str = ", ".join(chunk)
        system_prompt=  f"""# Role
귀하는 소설의 원문을 분석하여 장소 설정집(Place Profile)을 생성하는 전문 편집자이자 데이터 분석가입니다.

# Task
제공된 소설 본문을 읽고, 미리 추출한 장소 키워드 [{names_str}] 각각의 정보를 분석하여 지정된 JSON 포맷으로 출력하세요. 
새로운 정보가 발견될 때마다 기존 설정과 대조할 수 있도록 매우 구체적이고 객관적인 수치와 묘사 위주로 추출해야 합니다.
본문에서 해당 정보가 있다면 None값을 삭제하고 정보를 str형식으로 적으세요

# 본문
{txt}

# extraction_rules
1. **Fact-Based**: 추측하지 말고 본문에 명시된 텍스트 그대로를 기반으로 추출하세요.
"""+"""   
# JSON_Output_Format
{
   "장소 키워드1":{
      "별칭":[
         "지역별칭 (없으면 None)"
      ],
      "분위기":[
         "지역의 분위기(없으면 None)"
      ],
      "역사":[
         "지역의 역사 또는 생성된 시기 (없으면 None)"
      ],
      "위치":[
         "지역의 위치 (없으면 None)"
      ],
      "집단":[
         "해당 공간에 속한 소속 집단/가문(없으면 None)"
      ],
      "하부지역":[
         "이 장소에 속한 다른 지역 (없으면 None)"
      ],
      "작중묘사":[
         "해당 장소에 대한 묘사 (없으면 None)"
      ],
      "규칙":[
         "물리적/사회적 규칙 (없으면 None)"
      ],
      "금기":[
         "금기시되는 것(없으면 None)"
      ],
      "필수 제약":[
         "필수불가결한 제약(없으면 None)"
      ]
   },
   "장소 키워드2":{
      "별칭":[
         "지역별칭 (없으면 None)"
      ],
      "분위기":[
         "지역의 분위기(없으면 None)"
      ],
      "역사":[
         "지역의 역사 또는 생성된 시기 (없으면 None)"
      ],
      "위치":[
         "지역의 위치 (없으면 None)"
      ],
      "집단":[
         "해당 공간에 속한 소속 집단/가문(없으면 None)"
      ],
      "하부지역":[
         "이 장소에 속한 다른 지역 (없으면 None)"
      ],
      "작중묘사":[
         "해당 장소에 대한 묘사 (없으면 None)"
      ],
      "규칙":[
         "물리적/사회적 규칙 (없으면 None)"
      ],
      "금기":[
         "금기시되는 것(없으면 None)"
      ],
      "필수 제약":[
         "필수불가결한 제약(없으면 None)"
      ]
   }
}

# Constraints
- 응답은 반드시 순수한 JSON 형식이어야 하며, 다른 설명 문구는 포함하지 마세요.
- 본문에 정보가 전혀 없는 항목만 None값으로 남겨두세요.
- 만약 장소 키워드가 하나만 있을 경우에는 한 장소만 출력하세요.
- 제공한 키워드를 제외한 나머지는 임의로 추가하여 출력하지 마세요."""
        msg = f"규칙을 바탕으로 순수한 Json 텍스트 형식으로 출력하세요."


        result = await genai(system_prompt,msg)
        try:
            result = result.replace("json", "").replace("```", "").strip()
            print(result)
            result = json.loads(result)
            print("변환 성공!")
            for k, v in result.items():
                place[k] = v
        except:
            print("딕셔너리 변환 에러")
            print("result")
            #state = charter(state)

    return {**settingstate,
           '장소' :place
           }


# ### 4.설정추출 - 사건

# In[22]:


async def event(settingstate: SettingState) -> SettingState:
    lis = settingstate.get('check')
    lis = lis['사건']
    txt = settingstate.get('txt')
    event = settingstate.get('사건',{})
    for i in range(0, len(lis), 5):
        chunk = lis[i:i+5]  
        names_str = ", ".join(chunk)
        system_prompt= f"""# Role
귀하는 소설의 원문을 분석하여 사건 설정집(Place Profile)을 생성하는 전문 편집자이자 데이터 분석가입니다.

# Task
제공된 소설 본문을 읽고, 미리 추출한 사건 키워드 [{names_str}] 각각의 정보를 분석하여 지정된 JSON 포맷으로 출력하세요. 
# 본문
{txt}
"""+"""   
# JSON_Output_Format
{
   "사건 키워드1":{
      "관련 인물":[
         "관련 인물(없으면 None)"
      ],
      "설명":[
         "해당 사건에 관한 짧은 설명"
      ],
      "트리거":[
         "발단 원인(없으면 None)"
      ],
      "영향 범위":[
         "해당 사건의 파급력 범위(없으면 None)"
      ],
      "변화점":[
         "사건 이후 확실히 바뀐 점(없으면 None)"
      ]
   },
   "사건 키워드2":{
      "관련 인물":[
         "관련 인물(없으면 None)"
      ],
      "설명":[
         "해당 사건에 관한 짧은 설명"
      ],
      "트리거":[
         "발단 원인(없으면 None)"
      ],
      "영향 범위":[
         "해당 사건의 파급력 범위(없으면 None)"
      ],
      "변화점":[
         "사건 이후 확실히 바뀐 점(없으면 None)"
      ]
   }
}

# Constraints
- 응답은 반드시 순수한 JSON 형식이어야 하며, 다른 설명 문구는 포함하지 마세요.
- 만약 키워드가 하나만 있을 경우에는 한 사건만 출력하세요.
- 제공한 키워드를 제외한 나머지는 임의로 추가하여 출력하지 마세요.
"""
        msg=f"규칙을 바탕으로 순수한 Json 텍스트 형식으로 출력하세요."
        result = await genai(system_prompt,msg)
        try:
            result = result.replace("json", "").replace("```", "").strip()
            print(result)
            result = json.loads(result)
            print("변환 성공!")
            for k, v in result.items():
                event[k] = v
        except:
            print("딕셔너리 변환 에러")
            print(result)
            #state = charter(state)

    return {**settingstate,
           '사건' :event
           }


# ### 5.설정추출 - 아이템

# In[23]:


async def item(settingstate: SettingState) -> SettingState:
    lis = settingstate.get('check')
    lis = lis['물건']
    txt = settingstate.get('txt')
    item = settingstate.get('물건',{})

    for i in range(0, len(lis), 4):
        chunk = lis[i:i+4]  
        names_str = ", ".join(chunk)
        system_prompt = f"""# Role
귀하는 소설의 원문을 분석하여 아이템 설정(item Profile)을 생성하는 전문 편집자이자 데이터 분석가입니다.
본문에서 해당 정보가 있다면 None값을 삭제하고 정보를 str형식으로 적으세요

# Task
제공된 소설 본문을 읽고, 미리 추출한 아이템 키워드 [{names_str}] 각각의 정보를 분석하여 지정된 JSON 포맷으로 출력하세요. 
summary에는 물건의 성능과 역할에 대해 간략하게 적어야 합니다.
character에는 해당 사건의 메인 인물을 적어야 합니다.
55
# 본문
{txt}
"""+"""   
# JSON_Output_Format
{
   "아이템 키워드1":{
      "설명":[
         "해당 물건 관한 짧은 설명 (없으면 None)"
      ],
      "관련인물":[
         "관련된인물 (없으면 None)"
      ],
      "리스크":[
         "물건이 가진 위험성 (없으면 None)"
      ],
      "능력":[
         "물건이 가진 능력 (없으면 None)"
      ],
      "유래":[
         "물건의 배경, 설화 (없으면 None)"
      ],
      "귀속상태":[
         "소유자 있는지, 없다면 파괴나 탈취 가능한지 (없으면 None)"
      ]
   },
   "아이템 키워드2":{
      "설명":[
         "해당 물건 관한 짧은 설명 (없으면 None)"
      ],
      "관련인물":[
         "관련된인물 (없으면 None)"
      ],
      "리스크":[
         "물건이 가진 위험성 (없으면 None)"
      ],
      "능력":[
         "물건이 가진 능력 (없으면 None)"
      ],
      "유래":[
         "물건의 배경, 설화 (없으면 None)"
      ],
      "귀속상태":[
         "소유자 있는지, 없다면 파괴나 탈취 가능한지 (없으면 None)"
      ]
   }
}

# Constraints
- 응답은 반드시 순수한 JSON 형식이어야 하며, 다른 설명 문구는 포함하지 마세요.
- 관련인물은 사건에 등장하는 모든 인물이 아닌 아이템과 관련된 주역만 적어야 합니다. 주역이 없을 경우 빈 배열 '[]'로 처리하세요.
- 만약 키워드가 하나만 있을 경우에는 한 물건만 출력하세요.
- 제공한 키워드를 제외한 나머지는 임의로 추가하여 출력하지 마세요.
"""
        msg = f"규칙을 바탕으로 순수한 Json 텍스트 형식으로 출력하세요."
        result = await genai(system_prompt,msg)
        try:
            result = result.replace("json", "").replace("```", "").strip()
            print(result)
            result = json.loads(result)
            print("변환 성공!")
            for k, v in result.items():
                item[k] = v
        except:
            print("딕셔너리 변환 에러")
            print(result)
            #state = charter(state)

    return {**settingstate,
           '물건' :item
           }


# ### 6.설정추출 - 단체

# In[24]:


async def group(Settingstate: SettingState) -> SettingState:
    lis = Settingstate.get('check')
    lis = lis['집단']
    txt = Settingstate.get('txt')
    group = Settingstate.get('집단',{})

    for i in range(0, len(lis), 3):
        chunk = lis[i:i+3]
        names_str = ", ".join(chunk)
        system_prompt=f"""# Role
귀하는 소설의 원문을 분석하여 집단 설정(group Profile)을 생성하는 전문 편집자이자 데이터 분석가입니다.
본문에서 해당 정보가 있다면 None값을 삭제하고 정보를 str형식으로 적으세요

# Task
제공된 소설 본문을 읽고, 미리 추출한 집단 키워드 [{names_str}] 각각의 정보를 분석하여 지정된 JSON 포맷으로 출력하세요. 
summary에는 집단의 역사와 목적 대해 간략하게 적어야 합니다.
character에는 해당 집단에 속해있는 인물을 적은 후 집단에서 역할을 괄호'()'안에 적으세요.

# 본문
{txt}
"""+"""   
# JSON_Output_Format
{
   "집단 키워드1":{
      "설명":[
         "집단에 관한 짧은 설명(없으면 None)"
      ],
      "구성원":[
         "구성원이름(역할)(없으면 None)"
      ],
      "의사결정 구조":[
         "단체가 움직이는 방식: 개별/조직적(없으면 None)"
      ],
      "유지 동력":[
         "조직 유지 기반: 자금, 신념, 공포 등(없으면 None)"
      ],
      "대외적 평판":[
         "외부의 평가, 시선(없으면 None)"
      ]
   },
   "집단 키워드2":{
      "설명":[
         "집단에 관한 짧은 설명(없으면 None)"
      ],
      "구성원":[
         "구성원이름(역할)(없으면 None)"
      ],
      "의사결정 구조":[
         "단체가 움직이는 방식: 개별/조직적(없으면 None)"
      ],
      "유지 동력":[
         "조직 유지 기반: 자금, 신념, 공포 등(없으면 None)"
      ],
      "대외적 평판":[
         "외부의 평가, 시선(없으면 None)"
      ]
   }
}


# Constraints
- 응답은 반드시 순수한 JSON 형식이어야 하며, 다른 설명 문구는 포함하지 마세요.
- 본문에 정보가 전혀 없는 항목만 None값으로 남겨두세요.
- 구성원은 대상을 특정하기 위한 수식어(예: '데마시아의 정치체계','총의 악마','OO왕국 기사단')를 제외한 수식어는(예: '비시르 함장' X, '비시르' O) 모두 제거하여 넣으세요.
- 만약 키워드가 하나만 있을 경우에는 한 집단만 출력하세요.
- 제공한 키워드를 제외한 나머지는 임의로 추가하여 출력하지 마세요.
"""
        msg = f"규칙을 바탕으로 순수한 Json 텍스트 형식으로 출력하세요."
        result = await genai(system_prompt,msg)
        try:
            result = result.replace("json", "").replace("```", "").strip()
            print(result)
            result = json.loads(result)
            print("변환 성공!")
            for k, v in result.items():
                group[k] = v

        except:
            print("딕셔너리 변환 에러")
            print(result)
            #state = charter(state)

    return {**Settingstate,
           '집단' :group
           }


# ### 관계도 다이어그램 출력

# In[25]:


async def Relationship(rel):
    system_prompt=f"""### Role
당신은 소설의 데이터를 분석하여 인물 관계를 최적화된 Mermaid 코드로 시각화하는 전문가입니다.

### Task
입력된 인물 데이터를 바탕으로 다음 '관계 통합 로직'에 따라 Mermaid 코드를 생성하세요.

### Core Logic (관계 통합 규칙)
1. 인물 식별 및 통합:
   - 동일 인물이 다른 이름으로 표기된 경우 하나로 통합하세요 (예: '소나', '소녀 소나' -> '소나').
   - 각 인물에 고유 ID(A, B, C...)를 부여하여 코드를 작성하세요. (형식: A[인물명])

2. 선 통합 및 양방향 기호 (<-->):
   - A와 B가 서로 주고받는 동일한 관계(예: A->B 동료, B->A 동료)는 반드시 하나로 합쳐 양방향 화살표(<-->)로 표시하세요.
   - 예: A[소나] <-->|동료| B[코로바크]

3. 추가 단방향 관계 (-->):
   - 양방향 관계 외에 특정 인물이 상대에게만 가지는 별도의 관계가 있다면 단방향 화살표(-->)를 추가하세요.
   - 예: A <-->|동료| B 연 결 후, B가 A를 짝사랑한다면 B -->|짝사랑| A 추가.

4. 선 개수 제한 (Maximum 3):
   - 인물 A와 B 사이에는 어떤 경우에도 선(Edge)이 '3개'를 초과해서는 안 됩니다.
   - 관계가 너무 많다면 중요도가 낮은 관계는 생략하거나, 하나의 라벨 안에 콤마(,)로 합쳐서 표현하세요.

### Constraints
- 출력 결과에는 제목, 설명, 불릿 포인트, ``` 등을 절대 포함하지 마세요.
- 오직 "graph TD"로 시작하는 Mermaid 코드만 출력하세요.
- 인물명과 관계의 공백은 언더바(_)로 대체하세요.
- 관계가 None인 경우 문맥상 가장 적절한 관계로 추측하여 기입하세요.

### 출력 예시
graph TD
    A[소나] <-->|동료,보호_대상| B[코로바크]
    B -->|희생하려_함| A
    B -->|심문_대상| C[케인]
"""
    msg = f"{rel}인물 관계 데이터가 이렇게 저장되어 있을 때 mermaid 코드로 출력해줘."

    result = await genai(system_prompt,msg)
    print("관계도 출력 완료")
    print(result)
    return result



# ### 사건 타임라인 출력

# In[26]:


async def timeline(rel):
    system_prompt=f"""### Role
당신은 소설의 데이터를 분석하여 사건을 최적화된 Mermaid 코드로 시각화하는 전문가입니다.

### Task
입력된 인물 데이터를 바탕으로 다음 '출력 로직'에 따라 Mermaid 코드를 생성하세요.

### Core Logic (출력 로직)
1. 입력되는 데이터의 형식은 [(id, [ep_num], 키워드),.....]입니다.
2. 동일한 시간대에 키워드가 여러개 있다면 입력 데이터의 id 값을 참고하여 더 낮은값을 가진 키워드가 먼저 배치되게 하세요.
예시) 
입력 데이터 : [(2, [1], 키워드2),(1, [1], 키워드1),(3, [2], 키워드3),....]

출력 데이터 :
timeline
    title 사건 전계도
    1화 : 키워드1
        : 키워드2
    2화 : 키워드3
    ...

### Constraints
- 출력 결과에는 제목, 설명, 불릿 포인트, ``` 등을 절대 포함하지 마세요.
- 오직 "timeline"로 시작하는 Mermaid 코드만 출력하세요.

### 출력 예시
timeline
    title 사건 전계도
    ep_num화 : 키워드
    ep_num화 : 키워드
             : 키워드
    ep_num화 : 키워드
    ep_num화 : 키워드
"""
    msg = f"{rel}사건의 키워드를 찾은 데이터가 이렇게 저장되어 있을 때 mermaid 코드로 출력해줘."

    result = await genai(system_prompt,msg)
    print("타임라인 출력 완료")
    print(result)
    return result



# ### 추출한 설정 검수 & 분류

# In[27]:


async def comparison(Settingstate: SettingState):
    check = Settingstate.get('check')
    print(Settingstate)
    work_id = Settingstate.get('work_id')
    user_id = Settingstate.get('user_id')
    uplist ={'인물':[],'세계':[],'장소':[],'사건':[],'물건':[],'집단':[]}
    editlist = {'인물':[],'세계':[],'장소':[],'사건':[],'물건':[],'집단':[]}
    errorlist = {'인물':[],'세계':[],'장소':[],'사건':[],'물건':[],'집단':[]}
    oldlist = {'인물':[],'세계':[],'장소':[],'사건':[],'물건':[],'집단':[]}
    for tag,val in check.items():
        for keyword in val:
            emb = await select(tag,keyword,user_id,work_id)
            if (len(emb)==0):
                uplist[tag].append({keyword:Settingstate.get(tag).get(keyword),"ep_num":[Settingstate.get("ep_num")]}) 
                continue
            elif tag == "사건":
                if (len(emb)!=0):
                    errorlist[tag].append({keyword:"[결과: 충돌]\n [판단사유: 사건 이름이 동일한 사건이 존재합니다. 키워드를 변경해주세요.]","신규설정":Settingstate.get(tag).get(keyword)})
                    oldlist[tag].append(emb)
                    continue
                else:
                    uplist[tag].append({keyword:Settingstate.get(tag).get(keyword),"ep_num":[Settingstate.get("ep_num")]}) 
                    continue
            oldlist[tag].append(emb)
            i = Settingstate.get(tag).get(keyword)
            system_prompt= f"""#### 역할: 소설 설정 검수 전문가

### 상황
하나의 대상에 대하여 서로 다른 에피소드에서 뽑아온 설정이 있습니다. 해당 두 설정을 확인하고 설정이 어긋나는 경우가 존재하는지 찾아보세요.

### 분석할 설정
기존설정 : {emb[0][1]}

신규설정 : {i}

### 요청 사항
2. **상세 분석:** - **인과관계/상세화:** 
   - **결합:** 한쪽이 원인이고 다른 쪽이 결과이거나, 한쪽이 다른 쪽에서 없는 설명인 경우 '결합'로 판단합니다.
   - **충돌:** 예외 없이 물리적으로 불가능한 경우에만 '충돌'로 판단하세요.
    -**일치:** 설정이 추가되거나 보강되는 부분 없이 동일한 경우에는 '일치'로 판단하세요.

### 출력 형식
[결과: 충돌/결합/일치]
[판단사유: 판단사유]
        """
            msg = f"이거 설정이 맞을까?"     
            result = await genai(system_prompt,msg)
            if "[결과: 충돌]" in result:
                errorlist[tag].append({keyword:result,"신규설정":i})
            elif "[결과: 결합]" in result:
                editlist[tag].append(emb)
            print(result)
    editlist = await edit_setting(editlist,Settingstate)
    return {'충돌': errorlist, '설정 결합' : editlist,
            "신규 업로드": uplist, "기존설정" : oldlist}


# In[28]:


async def edit_setting(editlist, Settingstate:SettingState):
    lis = []
    for tag,value in editlist.items():
        for emb in value:            
            vctid,oldset,episode = emb[0]
            keyword = next(iter(oldset))
            system_prompt= f"""#### 역할: 소설 설정 작성 전문가

### 상황
하나의 대상에 대하여 서로 다른 에피소드에서 뽑아온 설정이 있습니다. 해당 두 설정을 확인하고 두 설정을 결합하세요.

### 분석할 설정
기존설정 : {oldset}

신규설정 : {keyword} : {Settingstate.get(tag).get(keyword)}

### 제한 사항
-기존 설정과 신규 설정의 시점 차이가 있을 경우, 시간상으로 나중에 일어난 일으로 업데이트하세요.
(예: 16세일때 외형묘사가 흑발, 18세일때 외형묘사가 금발이라면 외형을 금발으로 업데이트)
-서로 상충되는 부분이 아니라 보강되는 부분이라면 두 설정을 결합하세요.
(예: 기존 설정- 외형: 근육질, 신규 설정 -외형 : 흑발, > 결합 설정- 외형 : 근육질, 흑발
     설정1 -인물관계: 동료, 설정2-인물관계:오랜 친구 > 결합설정 - 인물관계 : 동료이자 오랜 친구)
- 서로 결합할 때, 키값을 임의로 변경하지 마세요.
### 출력 형식
신규설정 형식과 동일
"""
            msg = f"두 설정을 결합해줄래"     
            result = await genai(system_prompt,msg)
            try:
                result = result.replace("json", "").replace("```", "").strip()
                result = json.loads(result)
            except:
                print("딕셔너리 변환 에러 : edit_setting")
                print(result)
                #state = charter(state)
            print(result)
            episode.append(Settingstate.get('ep_num',0))
            lis.append([vctid,result,episode])
    return lis


# ## 원문&설정 읽기 쓰기

# In[29]:


def _save_text_sync(user_id, work_id, ep_num, txt):
    save_dir = f"./save_original/{user_id}/{work_id}"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{ep_num}.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(txt)

    return file_path

async def save_text(user_id, work_id, ep_num, txt):
    try:
        return await asyncio.to_thread(
            _save_text_sync,
            user_id,
            work_id,
            ep_num,
            txt
        )
    except Exception as e:
        print("error occured:", e)
        return "failed"

def _open_text_sync(user_id, work_id, ep_num) -> str:
    file_path = f"./save_original/{user_id}/{work_id}/{ep_num}.txt"

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

async def open_text(user_id, work_id, ep_num) -> str:
    try:
        return await asyncio.to_thread(
            _open_text_sync,
            user_id,
            work_id,
            ep_num
        )
    except Exception as e:
        print("error occured:", e)
        return "failed"

def save_json(user_id, work_id,lore_id, data):
    try:
        save_dir = f"./save_setting/{userid}/{work_id}"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{lore_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return f"./save_setting/{userid}/{work_id}/{lore_id}.json"
    except Exception as e:
        print("error occured: ", e)
        return "failed"

def open_json(user_id, work_id,lore_id) :
    try:
        file_path = f"./save_original/{user_id}/{work_id}/{lore_id}.json"
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e: 
        print("error occured: ", e)
        return "failed"


# ## IP 확장 충돌 검사

# ### 충돌 검사

# In[30]:


def normalize_description(desc: str) -> dict:
    try:
        parsed = json.loads(desc)
        if isinstance(parsed, dict) and len(parsed) == 1:
            return list(parsed.values())[0]
        return parsed
    except:
        return {}

async def comparison_ip(lorebooks: List[Dict[str, Any]]) -> Dict[str, Any]:
    conflict_list = {'인물':[], '세계':[], '장소':[], '사건':[], '물건':[], '집단':[]}
    match_list = {'인물':[], '세계':[], '장소':[], '사건':[], '물건':[], '집단':[]}

    grouped = {}
    for lore in lorebooks:
        keyword = lore.get("keyword")
        category = lore.get("category")
        if not keyword or not category or category not in conflict_list:
            continue
        key = (category, keyword)
        grouped.setdefault(key, []).append(lore)

    for (category, keyword), items in grouped.items():
        unique_works = {}
        for item in items:
            work_id = item.get("workId")
            if work_id not in unique_works:
                unique_works[work_id] = item

        if len(unique_works) < 2:
            continue

        settings_info = []
        for work_id, lore_data in unique_works.items():
            desc = normalize_description(lore_data.get("description", ""))
            settings_info.append({
                "작품명": lore_data.get("workTitle", "Unknown"),
                "작품ID": work_id,
                "설정내용": desc
            })

        settings_text = "\n\n".join([
            f"- 설정 {chr(65+idx)}:\n  - 작품명: {s['작품명']}\n  - 카테고리: {category}\n  - 설정 내용: {json.dumps(s['설정내용'], ensure_ascii=False)}"
            for idx, s in enumerate(settings_info)
        ])

        system_prompt = f"""#### 역할: IP 세계관 통합을 위한 소설 설정 충돌 검수 전문가

### 상황
서로 다른 작품에서 추출된 설정(lorebook)을 비교하여,
두 작품을 하나의 세계관 또는 크로스오버 작품으로 통합할 때
**설정 수정 없이 공존 가능한지 여부**를 판단합니다.

분석 목적은 학문적 비교가 아니라,
실제 상업 작품 통합·확장 시 발생하는 설정 리스크를 검토하는 것입니다.

### 입력 데이터
키워드: {keyword}
카테고리: {category}

{settings_text}

---

### 충돌 판단 기준 (실무 기준, 매우 중요)

다음 중 **하나라도 해당하면 `충돌`**로 판단합니다.

1. **정의 충돌**
   - 동일 개념 또는 동일 명칭이
   - 두 작품에서 **본질적으로 다른 정의**를 가짐
   - 설정 설명을 수정하거나 재정의하지 않으면 통합 불가한 경우

2. **존재 규칙 충돌**
   - 한 설정에서는 반드시 존재하거나 필수 요소인데
   - 다른 설정에서는 존재 자체가 부정되거나 불가능함

3. **작동 원리 충돌**
   - 동일 개념이 전혀 다른 방식으로 작동하며
   - 단순한 "세계관 차이"로 설명하기 어렵고
   - 독자에게 혼란을 주는 수준의 재설정이 필요한 경우

4. **권력·질서 구조 충돌**
   - 설정이 세계의 질서, 권력 구조, 기술·마법 체계의 중심을 차지하고 있으며
   - 두 설정을 동시에 유지할 경우 세계관의 근간이 붕괴되는 경우

5. **리라이트 불가 기준**
   - 설정을 통합하려면
     - 기존 설정의 삭제
     - 핵심 서사의 수정
     - 세계관 규칙의 변경
   중 하나 이상이 필수적으로 요구되는 경우

---

### 일치 판단 기준

아래 조건을 **모두 만족할 경우만 `일치`**로 판단합니다.

- 두 설정이 동시에 존재해도
  - 세계관 설명에 모순이 없고
  - 추가 설명이나 설정 변경이 필요하지 않으며
  - 독자가 혼란 없이 받아들일 수 있음
- 설정의 차이가 있다면
  - 표현 방식, 활용 방식, 서술 밀도의 차이에 불과함

---

### 판단 원칙

- 세계관이 다르다는 이유만으로는 충돌이 아님
- "이 작품에서는 이렇게 사용된다" 수준의 차이는 충돌이 아님
- **작품을 합쳤을 때 설정 설명 페이지를 새로 써야 하면 충돌**
- **주석이나 한 줄 설명으로 해결 가능하면 일치**

---

### 출력 형식
[결과: 충돌 / 일치]
[판단사유:
- 통합 시 문제되는 핵심 지점 요약
- 수정 필요 여부 및 이유
]
"""
        msg = "설정들이 충돌나는지 검사를 진행해줘"

        result = await genai(system_prompt, msg)

        result_data = {
            "keyword": keyword,
            "작품목록": [s["작품명"] for s in settings_info],
            "판단결과": result
        }

        if "[결과: 충돌]" in result:
            conflict_list[category].append(result_data)
        elif "[결과: 일치]" in result:
            match_list[category].append(result_data)

        print(f"[{category}:{keyword}] 작품 {len(settings_info)}개 검사 완료")

    return {
        "충돌": conflict_list,
        "일치": match_list
    }


# In[31]:


def _process_lorebooks(lorebooks):
    """로어북 리스트의 description이 문자열이면 JSON으로 변환"""
    if not lorebooks:
        return []

    processed = []
    for lb in lorebooks:
        item = lb.copy()
        try:
            # 문자열인 경우에만 파싱 시도
            if isinstance(item.get('description'), str):
                item['description'] = json.loads(item['description'])
        except (json.JSONDecodeError, TypeError):
            # 파싱 실패 시 원본 유지
            pass

        processed.append(item)

    return processed


# ### IP 확장 가이드 SQL

# In[32]:


async def select_ip_proposal(ipproposal_id: int) -> Optional[Dict[str, Any]]:
    # SQL은 작성하신 그대로 사용
    sql = """
    SELECT 
        ip.id, ip.manager_id, u.name AS manager_name, ip.title,
        ip.lorebook_ids, ip.target_format, ip.target_genre,
        ip.genre_strategy_text, ip.world_setting, ip.target_ages,
        ip.target_gender, ip.budget_scale, ip.tone_and_manner,
        ip.media_detail, ip.add_prompt, ip.exp_market,
        ip.exp_creative, ip.exp_visual, ip.exp_world,
        ip.exp_business, ip.exp_production, ip.created_at, ip.updated_at,
        (
            SELECT JSONB_BUILD_OBJECT(
                'details', JSONB_AGG(
                    JSONB_BUILD_OBJECT(
                        'work_title', w.title,
                        'author_name', u_auth.name,
                        'author_id', w.primary_author_id,
                        'items', (
                            SELECT JSONB_AGG(
                                JSONB_BUILD_OBJECT(
                                    'category', lb.category,
                                    'keyword', lb.keyword
                                )
                            )
                            FROM lorebooks lb
                            WHERE lb.work_id = w.id 
                              AND lb.id = ANY(ip.lorebook_ids)
                        )
                    )
                ),
                'distinct_author_ids', (
                    SELECT ARRAY_AGG(DISTINCT w2.primary_author_id)
                    FROM works w2
                    WHERE w2.id IN (
                        SELECT DISTINCT work_id
                        FROM lorebooks
                        WHERE id = ANY(ip.lorebook_ids)
                    )
                )
            )
            FROM works w
            LEFT JOIN users u_auth ON w.primary_author_id = u_auth.integration_id
            WHERE w.id IN (
                SELECT DISTINCT work_id
                FROM lorebooks
                WHERE id = ANY(ip.lorebook_ids)
            )
        ) AS source_details
    FROM ip_proposal ip
    LEFT JOIN users u ON ip.manager_id = u.integration_id
    WHERE ip.id = $1
    """

    try:
        async with pool.acquire() as conn:
            r = await conn.fetchrow(sql, ipproposal_id)
        if not r:
            return None

        raw_data = r[23]
        source_details = {}

        if raw_data:
            if isinstance(raw_data, str):
                try:
                    source_details = json.loads(raw_data) # 문자열이면 변환
                except:
                    source_details = {}
            elif isinstance(raw_data, dict):
                source_details = raw_data # 딕셔너리면 그대로 사용

        # 작성하신 깔끔한 로직 그대로 적용
        details = source_details.get("details", [])
        if details is None: details = [] # null 방지

        match_author_ids = source_details.get("distinct_author_ids", [])
        if match_author_ids is None: match_author_ids = [] # null 방지

        author_names = list(
            {d.get("author_name") for d in details if isinstance(d, dict) and d.get("author_name")}
        )

        return {
            "id": r[0],
            "manager_id": r[1],
            "manager_name": r[2] if r[2] else "미지정",
            "title": r[3],
            "lorebook_ids": r[4],
            "target_format": r[5],
            "target_genre": r[6],
            "genre_strategy_text": r[7],
            "world_setting": r[8],
            "target_ages": r[9],
            "target_gender": r[10],
            "budget_scale": r[11],
            "tone_and_manner": r[12],
            "media_detail": r[13],
            "add_prompt": r[14],
            "exp_market": r[15],
            "exp_creative": r[16],
            "exp_visual": r[17],
            "exp_world": r[18],
            "exp_business": r[19],
            "exp_production": r[20],
            "created_at": r[21].isoformat() if r[21] else None,
            "updated_at": r[22].isoformat() if r[22] else None,

            "source_details": details, # 리스트만 프론트로 전달
            "author_names": author_names,
            "match_author_ids": match_author_ids,
        }

    except Exception as e:
        print(f"Error executing select_ip_proposal: {e}")
        return None


# In[1]:


async def update_ip_proposal_results(ipproposal_id: int, final_result: dict, pdf_path: str, match_author_ids: list):
    """
    PDF 생성 후, 요약 내용과 파일 정보를 DB에 업데이트합니다.
    """


    try:
        # 1. 파일 정보 추출
        file_size = 0
        original_filename = ""

        if pdf_path and os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            original_filename = os.path.basename(pdf_path)

        # 2. 요약 정보 추출 (LangGraph 결과에서 summary_ 키를 가져옴)
        # 만약 요약이 없다면 빈 문자열 처리
        exp_market = final_result.get('summary_exp_market', '')
        exp_creative = final_result.get('summary_exp_creative', '')
        exp_visual = final_result.get('summary_exp_visual', '')
        exp_world = final_result.get('summary_exp_world', '')
        exp_business = final_result.get('summary_exp_business', '')
        exp_production = final_result.get('summary_exp_production', '')

        # 3. UPDATE SQL 실행
        sql = """
         UPDATE ip_proposal
            SET 
            exp_market = $1,
            exp_creative = $2,
            exp_visual = $3,
            exp_world = $4,
            exp_business = $5,
            exp_production = $6,

            file_path = $7,
            original_filename = $8,
            file_size = $9,

            match_author_id = $10,
            updated_at = NOW(),
            status = 'PENDING_APPROVAL'
        WHERE id = $11
        """

        async with pool.acquire() as conn:
            await conn.execute(
                sql,
                exp_market,
                exp_creative,
                exp_visual,
                exp_world,
                exp_business,
                exp_production,
                pdf_path,
                original_filename,
                file_size,
                match_author_ids,  # PostgreSQL ARRAY 자동 변환
                ipproposal_id,
            )
        print(f"[DB Update] Proposal ID {ipproposal_id} 업데이트 완료.")

    except Exception as e:
        print(f"[DB Update Error] {e}")
        raise e


# ### IP 확장

# In[34]:


class State(TypedDict):
    """LangGraph State 정의 - 표지 정보 추가"""
    work_info: Dict[str, Any]
    lorebooks: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]
    main_screen: Dict[str, Any]
    exp_market: Dict[str, Any]
    exp_creative: Dict[str, Any]
    exp_visual: Dict[str, Any]
    exp_world: Dict[str, Any]
    exp_business: Dict[str, Any]
    exp_production: Dict[str, Any]
    final_result: Dict[str, Any]
    summary_exp_market: str
    summary_exp_creative: str
    summary_exp_visual: str
    summary_exp_world: str
    summary_exp_business: str
    summary_exp_production: str
    pdf_path: str

    # 표지 정보 추가
    operator_name: str  # 기획자/운영자 이름
    authors: List[str]  # 작가 이름(들)
    publish_date: str   # 발행일


def create_base_context(work_info: Dict[str, Any], analysis_results: Dict[str, Any], lorebooks: List[Dict[str, Any]]) -> str:
    """
    모든 노드에서 공통으로 사용할 기본 컨텍스트 생성
    - lorebooks 데이터를 기반으로 작품명, 키워드, 상세 설정을 텍스트화하여 주입
    """

    # 1. 분석 결과 (충돌/일치) JSON 변환
    conflict_context = json.dumps(analysis_results.get("충돌", {}), ensure_ascii=False)
    match_context = json.dumps(analysis_results.get("일치", {}), ensure_ascii=False)

    # 2. 로어북 데이터 가공
    # (1) 로어북에 포함된 모든 'workTitle'을 추출하여 중복 제거 (예: 오디세이, 차원 포식자의 회귀)
    source_titles = sorted(list(set([lb.get('workTitle', 'Unknown') for lb in lorebooks if lb.get('workTitle')])))
    source_titles_str = ", ".join(source_titles) if source_titles else work_info.get("title", "Unknown")

    # (2) 로어북 상세 내용 텍스트화
    lorebook_lines = []
    for lb in lorebooks:
        w_title = lb.get('workTitle', 'Unknown')
        category = lb.get('category', '기타')
        keyword = lb.get('keyword', 'Unknown')

        # 설명(description)이 딕셔너리인 경우 JSON 문자열로 변환, 문자열이면 그대로 사용
        desc_data = lb.get('description', {})
        if isinstance(desc_data, dict):
            # 보기 좋게 한 줄로 요약하거나 JSON 그대로 덤프
            desc_str = json.dumps(desc_data, ensure_ascii=False)
        else:
            desc_str = str(desc_data)

        # 포맷: [작품명] <카테고리> 키워드: 내용
        line = f"- [{w_title}] <{category}> {keyword}: {desc_str}"
        lorebook_lines.append(line)

    lorebook_context_str = "\n".join(lorebook_lines) if lorebook_lines else "로어북 데이터 없음"

    # 3. 최종 컨텍스트 문자열 반환
    return f"""
[입력 IP 정보]
- 프로젝트명: {work_info.get("title")}
- 원작(Source Works): {source_titles_str}  <-- 로어북 기반으로 추출된 작품명들
- 확장 매체: {work_info.get("target_format")}
- 장르: {work_info.get("target_genre")}
- 장르 전략 텍스트: {work_info.get("genre_strategy_text")}
- 세계관 구조: {work_info.get("world_setting")}
- 타겟 연령: {work_info.get("target_ages")}
- 타겟 성별: {work_info.get("target_gender")}
- 예산 규모: {work_info.get("budget_scale")}
- 톤 앤 매너: {work_info.get("tone_and_manner")}
- 매체별 상세 설정: {json.dumps(work_info.get("media_detail"), ensure_ascii=False)}
- 추가 요구사항: {work_info.get("add_prompt")}

[로어북 상세 데이터 (설정 원본)]
{lorebook_context_str}

[설정 분석 결과]
- 설정 충돌: {conflict_context}
- 설정 일치: {match_context}
"""


# ### 메인 화면 정보 생성

# In[35]:


# ============================================================================
# 1. MAIN SCREEN 생성 노드
# ============================================================================
async def generate_main_screen(state: State) -> State:
    """메인 화면 정보 생성 (1회 AI 호출)"""

    work_info = state["work_info"]
    lorebooks = state["lorebooks"]
    analysis_results = state["analysis_results"]
    base_context = create_base_context(work_info, analysis_results, lorebooks)

    system_prompt = f"""
당신은 글로벌 콘텐츠 기업 및 투자기관에서 활동하는
**IP 확장 전략 수석 디렉터이자 OSMU 총괄 기획자**입니다.

당신의 임무는 IP 확장 전략의 **메인 화면 정보**를 생성하는 것입니다.

---

[전략 설계 원칙]
- 추상적 표현 금지
- 실행·판단 관점에서만 기술
- 마케팅 문구·과장 표현 금지
- 문장은 간결하되 단정적으로 작성
- 모든 항목은 실행 관점에서 작성

---

{base_context}

---

[출력 항목]

main_screen (딕셔너리 형태)
{{
  "review": "(400자 ~ 600자) 원천 IP의 확장 경쟁력을 3가지 강점, 2가지 시장 기회, 1가지 리스크와 대응 방안으로 구체적 진단. 설정 일치 요소를 핵심 자산으로 명시하고, 시장 데이터와 경쟁작 분석을 포함하여 객관적으로 평가. 제작 관점에서의 기술적 요구사항과 예산 정당성 확보 방안 명시",

  "roadmap": {{
    "phase_1": {{
      "title": "사전 제작",
      "objective": "(50자이하) 핵심 프로토타입 개발 및 IP 정체성 확립. 구체적 산출물(프로토타입, 아트 바이블, 핵심 시스템 3가지) 명시. 기술 검증 방법과 테스트 계획 포함",
      "period": "1-6개월"
    }},
    "phase_2": {{
      "title": "본 제작",
      "objective": "(50자이하) 주요 콘텐츠 제작 및 베타 테스트. 단계별 마일스톤 5개 이상 명시. 품질 관리 기준과 베타 테스트 KPI 포함",
      "period": "7-18개월"
    }},
    "phase_3": {{
      "title": "출시 및 운영",
      "objective": "(50자이하) 정식 출시 및 라이브 운영 전략. 론칭 후 3개월, 6개월, 12개월 목표 KPI와 성과 지표 명시. 장기 콘텐츠 로드맵 방향성 제시",
      "period": "19개월 이후"
    }}
  }},

  "partnership": "(200자 ~ 300자) 이 IP에 필요한 파트너를 3가지 관점(제작 기술력, 글로벌 유통 네트워크, 마케팅 역량)에서 구체적 설명. 각 파트너 유형별 핵심 역할 3가지와 기대 효과를 수치화하여 명시. 성공 사례 레퍼런스 2개 이상 포함",

  "entry_message": "(100자) 유저 첫 진입 시 전달되는 IP 정체성 선언문. 감정적 몰입을 유도하는 강렬한 메시지.",

  "identity_elements": [
    "(150자 이상) 매체의 첫인상을 결정짓는 핵심 공간(로비, 표지, 포스터, 세트장 등)의 구성. **1., 2., 3. 번호를 붙여 구분하고, 각 항목 끝에 반드시 <br/><br/> 태그를 넣어 줄바꿈.** 매체 특성에 맞는 카메라 앵글, 조명, 배치 전략 서술.",

    "(150자 이상) 사용자 인터페이스(UI), 타이틀 로고, 또는 전반적인 시각적 톤앤매너와 핵심 색상 전략. **1., 2., 3. 번호를 붙여 구분하고, 각 항목 끝에 반드시 <br/><br/> 태그를 넣어 줄바꿈.** 메인 컬러 2-3가지와 적용 방식 서술.",

    "(150자 이상) 캐릭터, 배우, 또는 피사체의 시각적 연출 포인트와 스타일. **1., 2., 3. 번호를 붙여 구분하고, 각 항목 끝에 반드시 <br/><br/> 태그를 넣어 줄바꿈.** 시그니처 의상, 외형적 특징, 매체별 표현 기법(실사화, 카툰렌더링 등) 서술."
  ]
}}

---

[출력 규칙]
- 반드시 Python dict 형태로 출력
- 위에 명시된 key 외에는 절대 출력 금지
- 설명, 주석, 추가 텍스트 일체 출력 금지
- 각 섹션의 최소 글자 수 준수 필수
- 구체적 수치, 사례, 전략을 반드시 포함
- 단락 구분은 자연스럽게 3-4문장 단위로 흐름 유지
- Markdown 태그 없이 순수 JSON 텍스트만 출력
- **설명 텍스트 내에서 쌍따옴표(") 사용 금지 (대신 홑따옴표(') 사용)**
- 위에 명시된 key 외에는 절대 출력 금지
"""

    user_msg = f"'{work_info.get('title')}'의 메인 화면 정보를 생성하라."

    try:
        response_raw = await genai(system_prompt, user_msg)
        main_screen = _extract_json(response_raw)
        state["main_screen"] = main_screen
        print("main_screen 생성 완료")
    except Exception as e:
        print(f"main_screen 생성 실패: {e}")
        state["main_screen"] = {}

    return state


# #### DETAIL 섹션 생성 노드들 - 시장 분석 섹션

# In[36]:


# ============================================================================
# 2. DETAIL 섹션 생성 노드들 (6개 노드, 1회 AI 호출)
# ============================================================================
async def generate_exp_market(state: State) -> State:
    """시장 분석 섹션 생성"""

    work_info = state["work_info"]
    lorebooks = state["lorebooks"]
    analysis_results = state["analysis_results"]
    base_context = create_base_context(work_info, analysis_results, lorebooks)

    system_prompt = f"""
당신은 글로벌 콘텐츠 기업 및 투자기관에서 활동하는
**IP 확장 전략 수석 디렉터이자 OSMU 총괄 기획자**입니다.

당신의 임무는 IP 확장 전략의 **시장 분석(exp_market)** 섹션을 생성하는 것입니다.

---

{base_context}

---

[출력 항목]

exp_market (딕셔너리 형태, 2개 섹션)
{{
  "target_analysis": "(600자 이상) 타겟 유저의 인구통계학적 특성(연령, 성별, 국가), 소비 패턴(플레이 시간, 과금 성향), 선호 장르를 데이터 기반으로 상세 분석. 주요 시장 3곳(한국, 북미, 일본 등)의 시장 규모를 구체적 수치로 제시하고 각 시장별 잠재력 평가. 경쟁작 3개 이상 분석하여 본 IP의 우위 요소 5가지 명시. F2P 모델 시장 트렌드와 유저 기대치 분석 포함",

  "positioning_strategy": "(600자 이상) 시장 내 포지셔닝 전략 및 차별화 포인트 7가지 이상 구체적 기술. USP(Unique Selling Point) 3가지를 명확히 정의하고 각각의 마케팅 메시지 방향 제시. 초기 진입 전략(사전예약, 인플루언서 마케팅, 미디어 믹스), 중기 성장 전략(콘텐츠 업데이트 주기, 커뮤니티 운영), 장기 브랜드 확립 전략(IP 확장, 크로스 프로모션)을 단계별로 구분하여 실행 방안 명시"
}}

---

[출력 규칙]
- 반드시 Python dict 형태로 출력
- 위에 명시된 key 외에는 절대 출력 금지
- 설명, 주석, 추가 텍스트 일체 출력 금지
- 각 섹션의 최소 글자 수 준수 필수
- 구체적 수치, 사례, 전략을 반드시 포함
- 단락 구분은 자연스럽게 3-4문장 단위로 흐름 유지
- Markdown 태그 없이 순수 JSON 텍스트만 출력
- **설명 텍스트 내에서 쌍따옴표(") 사용 금지 (대신 홑따옴표(') 사용)**
- 위에 명시된 key 외에는 절대 출력 금지
- exp_market은 한번만 호출하세요
"""

    user_msg = f"'{work_info.get('title')}'의 시장 분석 섹션을 생성하라."

    try:
        response_raw = await genai(system_prompt, user_msg)
        exp_market = _extract_json(response_raw)
        state["exp_market"] = exp_market
        print("exp_market 생성 완료")
    except Exception as e:
        print(f"exp_market 생성 실패: {e}")
        state["exp_market"] = {}

    return state


# #### 시장 분석 섹션

# In[37]:


async def generate_exp_creative(state: State) -> State:
    """창작 전략 섹션 생성"""

    work_info = state["work_info"]
    lorebooks = state["lorebooks"]
    analysis_results = state["analysis_results"]
    base_context = create_base_context(work_info, analysis_results, lorebooks)

    system_prompt = f"""
당신은 글로벌 콘텐츠 기업 및 투자기관에서 활동하는
**IP 확장 전략 수석 디렉터이자 OSMU 총괄 기획자**입니다.

당신의 임무는 IP 확장 전략의 **창작 전략(exp_creative)** 섹션을 생성하는 것입니다.

---

{base_context}

---

[출력 항목]

exp_creative (딕셔너리 형태, 2개 섹션)
{{
  "story_expansion": "(700자 이상) 원작의 핵심 스토리 라인을 매체에 맞게 확장하는 구체적 방법론. 메인 퀘스트 12-15개 챕터 구조 제시하고 각 챕터의 핵심 이벤트와 플레이어 선택지 명시. 서브 퀘스트 시스템을 통한 조연 캐릭터 심화 전략 5가지 이상 제시. 원작에서 비중이 적었던 요소를 확장하는 구체적 사례 3가지 포함. 게임 오리지널 스토리 라인의 방향성과 원작 팬에게 제공할 새로운 가치 명시. 스토리 전개에 따른 유저 감정 곡선 설계 포함",

  "conflict_resolution": "(600자 이상) 설정 충돌을 각색 전략으로 전환하는 구체적 방법론. 원작의 특정 능력이 게임 밸런스를 해칠 경우의 5가지 이상 해결 방안(스킬 쿨다운 조정, 특정 조건 발동, 레벨 제한 등) 제시. 원작의 시간 흐름이나 사건 순서가 게임 플레이와 맞지 않을 경우 시간 왜곡, 평행 세계, 회상 시스템 등의 설정을 활용한 자연스러운 연결 방법 명시. 구체적 사례 3가지 이상 포함. 각 해결 방안이 게임 플레이 재미에 미치는 영향 분석"
}}

---

[출력 규칙]
- 반드시 Python dict 형태로 출력
- 위에 명시된 key 외에는 절대 출력 금지
- 설명, 주석, 추가 텍스트 일체 출력 금지
- 각 섹션의 최소 글자 수 준수 필수
- 구체적 수치, 사례, 전략을 반드시 포함
- 단락 구분은 자연스럽게 3-4문장 단위로 흐름 유지
- Markdown 태그 없이 순수 JSON 텍스트만 출력
- **설명 텍스트 내에서 쌍따옴표(") 사용 금지 (대신 홑따옴표(') 사용)**
- 위에 명시된 key 외에는 절대 출력 금지
- exp_creative 한번만 호출하세요
"""

    user_msg = f"'{work_info.get('title')}'의 창작 전략 섹션을 생성하라."

    try:
        response_raw = await genai(system_prompt, user_msg)
        exp_creative = _extract_json(response_raw)
        state["exp_creative"] = exp_creative
        print("exp_creative 생성 완료")
    except Exception as e:
        print(f"exp_creative 생성 실패: {e}")
        state["exp_creative"] = {}

    return state


# #### 비주얼 전략 섹션

# In[38]:


async def generate_exp_visual(state: State) -> State:
    """비주얼 전략 섹션 생성"""

    work_info = state["work_info"]
    lorebooks = state["lorebooks"]
    analysis_results = state["analysis_results"]
    base_context = create_base_context(work_info, analysis_results, lorebooks)

    system_prompt = f"""
당신은 글로벌 콘텐츠 기업 및 투자기관에서 활동하는
**IP 확장 전략 수석 디렉터이자 OSMU 총괄 기획자**입니다.

당신의 임무는 IP 확장 전략의 **비주얼 전략(exp_visual)** 섹션을 생성하는 것입니다.

---

{base_context}

---

[출력 항목]

exp_visual (딕셔너리 형태, 2개 섹션)
{{
  "visual_style": "(600자 이상) 비주얼 스타일의 핵심 방향성을 톤 앤 매너, 색상 팔레트, 렌더링 기법 관점에서 상세 기술. 원작의 아트 스타일을 존중하면서도 게임 엔진의 성능을 활용한 차별화 요소 5가지 명시. 캐릭터 모델링의 디테일 수준(폴리곤 수, 텍스처 해상도), 몬스터 디자인의 공포 연출 기법 3가지, 던전 및 배경 디자인의 동선 설계 원칙 명시. 시그니처 색상의 사용 위치와 빈도, 조명 연출 방식 포함",

  "art_direction": "(600자 이상) 아트 디렉션의 핵심 키워드(예: 압도적 성장, 어둠 속 희망)를 중심으로 한 실행 전략. 캐릭터 성장 단계별 외형 변화를 5단계 이상으로 구분하고 각 단계의 시각적 특징 명시. 스킬 사용 시 이펙트 연출 방식 3가지(파티클, 카메라 워크, 사운드 연동) 구체적 기술. UI/UX 디자인의 직관성 확보 방안과 어두운 톤 앤 매너와의 조화 방법 명시. 포인트 컬러 사용 전략 5가지 이상 포함"
}}

---

[출력 규칙]
- 반드시 Python dict 형태로 출력
- 위에 명시된 key 외에는 절대 출력 금지
- 설명, 주석, 추가 텍스트 일체 출력 금지
- 각 섹션의 최소 글자 수 준수 필수
- 구체적 수치, 사례, 전략을 반드시 포함
- 단락 구분은 자연스럽게 3-4문장 단위로 흐름 유지
- Markdown 태그 없이 순수 JSON 텍스트만 출력
- **설명 텍스트 내에서 쌍따옴표(") 사용 금지 (대신 홑따옴표(') 사용)**
- 위에 명시된 key 외에는 절대 출력 금지
- exp_visual 한번만 호출하세요
"""

    user_msg = f"'{work_info.get('title')}'의 비주얼 전략 섹션을 생성하라."

    try:
        response_raw = await genai(system_prompt, user_msg)
        exp_visual = _extract_json(response_raw)
        state["exp_visual"] = exp_visual
        print("exp_visual 생성 완료")
    except Exception as e:
        print(f"exp_visual 생성 실패: {e}")
        state["exp_visual"] = {}

    return state


# #### 세계관 확장 섹션

# In[39]:


async def generate_exp_world(state: State) -> State:
    """세계관 확장 섹션 생성"""

    work_info = state["work_info"]
    lorebooks = state["lorebooks"]
    analysis_results = state["analysis_results"]
    base_context = create_base_context(work_info, analysis_results, lorebooks)

    system_prompt = f"""
당신은 글로벌 콘텐츠 기업 및 투자기관에서 활동하는
**IP 확장 전략 수석 디렉터이자 OSMU 총괄 기획자**입니다.

당신의 임무는 IP 확장 전략의 **세계관 확장(exp_world)** 섹션을 생성하는 것입니다.

---

{base_context}

---

[출력 항목]

exp_world (딕셔너리 형태, 2개 섹션)
{{
  "world_expansion": "(700자 이상) 원작의 핵심 설정(게이트, 몬스터 등)을 기반으로 한 세계관 확장 전략. 다양한 종류의 게이트(일반, 희귀, 보스급, 이벤트용) 분류 체계와 각 등급별 특징 5가지 이상 명시. 몬스터 생태계를 계층 구조로 설계하고 각 계층의 특성과 기원 스토리 포함. 헌터 협회, 그림자 군단, 국가별 헌터 조직 등 3개 이상 집단의 설정과 이들 간의 관계도 명시. 지역별 던전 디자인(얼음 동굴, 용암 지대, 고대 유적 등) 7가지 이상 제시하고 각 던전의 퍼즐 요소와 보상 구조 포함",

  "asset_maximization": "(600자 이상) 원작의 매력적인 캐릭터와 몬스터를 수집형 RPG 자산으로 활용하는 구체적 전략. 주요 캐릭터 10명 이상의 고유 스킬, 성장 트리, 시그니처 디자인 요소 명시. 몬스터 수집 및 육성 시스템의 차별화 포인트 5가지(스킬 조합, 약점 시스템, 진화 경로 등) 제시. 원작의 상징적인 장소(성진우의 방, 헌터 협회 등)를 게임 내 로비나 거점으로 활용하는 방안 3가지 명시. 명대사 및 상징적 장면을 게임 내 연출로 재현하는 구체적 사례 5가지 포함"
}}

---

[출력 규칙]
- 반드시 Python dict 형태로 출력
- 위에 명시된 key 외에는 절대 출력 금지
- 설명, 주석, 추가 텍스트 일체 출력 금지
- 각 섹션의 최소 글자 수 준수 필수
- 구체적 수치, 사례, 전략을 반드시 포함
- 단락 구분은 자연스럽게 3-4문장 단위로 흐름 유지
- Markdown 태그 없이 순수 JSON 텍스트만 출력
- **설명 텍스트 내에서 쌍따옴표(") 사용 금지 (대신 홑따옴표(') 사용)**
- 위에 명시된 key 외에는 절대 출력 금지
- exp_world 한번만 호출하세요
"""

    user_msg = f"'{work_info.get('title')}'의 세계관 확장 섹션을 생성하라."

    try:
        response_raw = await genai(system_prompt, user_msg)
        exp_world = _extract_json(response_raw)
        state["exp_world"] = exp_world
        print("exp_world 생성 완료")
    except Exception as e:
        print(f"exp_world 생성 실패: {e}")
        state["exp_world"] = {}

    return state


# #### 비즈니스 전략 섹션

# In[40]:


async def generate_exp_business(state: State) -> State:
    """비즈니스 전략 섹션 생성"""

    work_info = state["work_info"]
    lorebooks = state["lorebooks"]
    analysis_results = state["analysis_results"]
    base_context = create_base_context(work_info, analysis_results, lorebooks)

    system_prompt = f"""
당신은 글로벌 콘텐츠 기업 및 투자기관에서 활동하는
**IP 확장 전략 수석 디렉터이자 OSMU 총괄 기획자**입니다.

당신의 임무는 IP 확장 전략의 **비즈니스 전략(exp_business)** 섹션을 생성하는 것입니다.

---

{base_context}

---

[출력 항목]

exp_business (딕셔너리 형태, 2개 섹션)
{{
  "revenue_model": "(600자 이상) 수익 모델의 5가지 이상 핵심 축을 구체적으로 설명. 1) 확률형 상품의 등급별 구성(SSR, SR, R)과 기대 수익률 제시. 2) 성장 가속 상품의 종류 7가지(재화, 경험치 부스터, 스태미나 충전, 스킬 북, 장비 강화 재료 등)와 가격 정책. 3) 코스튬 시스템의 시즌별 출시 계획과 주요 타겟(주인공, 인기 캐릭터). 4) 편의성 상품의 구체적 기능과 유저 편의 증대 효과. 5) 배틀 패스의 시즌별 보상 구조와 참여 유도 전략. 각 수익원의 기대 비중(%) 명시",

  "business_strategy": "(600자 이상) 비즈니스 전략의 단계별 실행 계획. 초기 단계: 원작 IP 팬덤 대상 사전예약 및 론칭 프로모션의 구체적 방법 5가지(인플루언서 협업, 미디어 믹스, 오프라인 이벤트 등)와 목표 사전예약 수 명시. 중기 단계: 콘텐츠 업데이트 주기(월 1회 대형 업데이트, 주 1회 이벤트)와 유저 잔존율 목표치(1개월 60%, 3개월 40%) 설정. 과금 모델 밸런스 조절을 통한 라이트 유저와 헤비 유저 동시 만족 전략 3가지. 글로벌 전략: 지역별 마케팅 차별화 방안과 현지화 작업 범위 명시. 장기 전략: IP 가치 제고를 위한 다른 매체 확장 계획"
}}

---

[출력 규칙]
- 반드시 Python dict 형태로 출력
- 위에 명시된 key 외에는 절대 출력 금지
- 설명, 주석, 추가 텍스트 일체 출력 금지
- 각 섹션의 최소 글자 수 준수 필수
- 구체적 수치, 사례, 전략을 반드시 포함
- 단락 구분은 자연스럽게 3-4문장 단위로 흐름 유지
- Markdown 태그 없이 순수 JSON 텍스트만 출력
- **설명 텍스트 내에서 쌍따옴표(") 사용 금지 (대신 홑따옴표(') 사용)**
- 위에 명시된 key 외에는 절대 출력 금지
- exp_business 한번만 호출하세요
"""

    user_msg = f"'{work_info.get('title')}'의 비즈니스 전략 섹션을 생성하라."

    try:
        response_raw = await genai(system_prompt, user_msg)
        exp_business = _extract_json(response_raw)
        state["exp_business"] = exp_business
        print("exp_business 생성 완료")
    except Exception as e:
        print(f"exp_business 생성 실패: {e}")
        state["exp_business"] = {}

    return state


# #### 제작 전략 섹션

# In[41]:


async def generate_exp_production(state: State) -> State:
    """제작 전략 섹션 생성"""

    work_info = state["work_info"]
    lorebooks = state["lorebooks"]
    analysis_results = state["analysis_results"]
    base_context = create_base_context(work_info, analysis_results, lorebooks)

    system_prompt = f"""
당신은 글로벌 콘텐츠 기업 및 투자기관에서 활동하는
**IP 확장 전략 수석 디렉터이자 OSMU 총괄 기획자**입니다.

당신의 임무는 IP 확장 전략의 **제작 전략(exp_production)** 섹션을 생성하는 것입니다.

---

{base_context}

---

[출력 항목]

exp_production (딕셔너리 형태, 2개 섹션)
{{
  "tech_stack": "(600자 이상)  선택된 타겟 포맷(웹툰 드라마, 게임 영화, 스핀오프, 상업이미지)의 특성에 따른 시각적 구현 난이도와 기술적 요구사항을 구체적으로 분석. 
  1) 비주얼 구현: 기획된 연출을 구현하기 위해 필요한 핵심 기술(3D 툴, 물리 엔진, 특수 촬영, 고밀도 작화 등)의 난이도 상/중/하 평가. 영상/게임의 경우 CG/VFX 비중과 R&D가 필요한 기술적 장벽(Lighting, Rendering, Optimization) 분석. 
  2) 파이프라인 적합성: 현재 가용한 제작 인프라와 기술 스택으로 원작의 퀄리티를 재현할 수 있는지에 대한 타당성 검토. 
  3) 리스크 관리: 제작 과정에서 발생할 수 있는 기술적 병목 구간(Rendering Time, 작화 붕괴, 프레임 드랍 등)을 예측하고 이를 해결하기 위한 구체적인 기술적 대안(Alternative) 제시.",

  "platform_strategy": "(600자 이상)  한정된 예산과 시간 내에서 최대 퀄리티를 내기 위한 자원 배분 계획. 
  1) 에셋 활용성: 배경, 소품, 캐릭터 모델링 등 제작된 리소스(Asset)의 재사용성(Reusability)과 확장성 검토 (예: 웹툰 배경의 3D화, 영상의 버추얼 스튜디오 활용, 게임의 모듈형 레벨 디자인). 
  2) 로케이션 및 공정 효율화: 제작 비용을 절감할 수 있는 현실적인 방안(로케이션 촬영 대안, 자동화 툴 도입, 외주 파이프라인 최적화 등) 분석. 
  3) 우선순위(MVP) 설정: 런칭 시점에 반드시 포함해야 할 핵심 요소(Core Feature)와 업데이트/후반작업으로 미룰 수 있는 요소를 구분하여 예산 투입의 선택과 집중 전략 수립."
}}

---

[출력 규칙]
- 반드시 Python dict 형태로 출력
- 위에 명시된 key 외에는 절대 출력 금지
- 설명, 주석, 추가 텍스트 일체 출력 금지
- 각 섹션의 최소 글자 수 준수 필수
- 구체적 수치, 사례, 전략을 반드시 포함
- 단락 구분은 자연스럽게 3-4문장 단위로 흐름 유지
- Markdown 태그 없이 순수 JSON 텍스트만 출력
- **설명 텍스트 내에서 쌍따옴표(") 사용 금지 (대신 홑따옴표(') 사용)**
- 위에 명시된 key 외에는 절대 출력 금지
- exp_production 한번만 호출하세요
"""

    user_msg = f"'{work_info.get('title')}'의 제작 전략 섹션을 생성하라."

    try:
        response_raw = await genai(system_prompt, user_msg)
        exp_production = _extract_json(response_raw)
        state["exp_production"] = exp_production
        print("exp_production 생성 완료")
    except Exception as e:
        print(f"exp_production 생성 실패: {e}")
        state["exp_production"] = {}

    return state


# #### 모든 섹션을 통합하여 최종 결과 생성

# In[42]:


# ============================================================================
# 3. 최종 통합 노드
# ============================================================================
def finalize_result(state: State) -> State:
    """모든 섹션을 통합하여 최종 결과 생성"""

    work_info = state["work_info"]

    final_result = {
        "title": work_info.get("title"),
        "target_format": work_info.get("target_format"),
        "target_genre": work_info.get("target_genre"),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "main_screen": state.get("main_screen", {}),
        "exp_market": state.get("exp_market", {}),
        "exp_creative": state.get("exp_creative", {}),
        "exp_visual": state.get("exp_visual", {}),
        "exp_world": state.get("exp_world", {}),
        "exp_business": state.get("exp_business", {}),
        "exp_production": state.get("exp_production", {})
    }

    state["final_result"] = final_result
    state["pdf_path"] = ""
    print("\n 최종 결과 통합 완료")

    return state


# #### 유틸리티 함수

# In[43]:


def _extract_json(response_raw: str) -> Dict[str, Any]:
    """응답에서 JSON 추출 및 파싱 (강화된 버전)"""
    if not response_raw:
        print("response_raw가 비어있습니다.")
        return {}

    # 1. Markdown 코드 블록 제거 (대소문자 무시 및 공백 처리)
    text = response_raw.strip()
    match = re.search(r"```(json|python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(2) # 코드 블록 내부 내용만 추출

    # 2. JSON 부분만 추출 (중괄호 시작과 끝 찾기)
    start_idx = text.find("{")
    end_idx = text.rfind("}")

    if start_idx == -1 or end_idx == -1:
        # 중괄호가 없는 경우, 전체 텍스트가 JSON일 수 있으므로 시도
        pass 
    else:
        text = text[start_idx:end_idx+1]

    # 3. 1차 파싱 시도
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass # 실패 시 클리닝 후 재시도

    # 4. JSON 클리닝 (Trailing Comma 및 주석 제거)
    cleaned_json = _clean_json_string(text)

    try:
        return json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 최종 실패: {e}")
        # 디버깅을 위해 실패한 앞부분 출력
        print(f"   실패한 텍스트 일부: {cleaned_json[:100]}...")
        return {}

def _clean_json_string(json_str: str) -> str:
    """JSON 문자열 보정 (Regex 강화)"""
    # 1. 주석 제거 (// ...)
    json_str = re.sub(r"//.*", "", json_str)

    # 2. Trailing Comma 제거 ( , } -> } )
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

    # 3. 제어 문자 제거
    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)

    return json_str

def register_korean_font():
    """한글 폰트 등록"""
    system = platform.system()
    font_name = 'MalgunGothic'
    font_bold_name = 'MalgunGothicBold'
    if system == 'Windows':
        font_path = r'C:\Windows\Fonts\malgun.ttf'
        font_bold_path = r'C:\Windows\Fonts\malgunbd.ttf'
        pdfmetrics.registerFont(TTFont(font_name, font_path))
        pdfmetrics.registerFont(TTFont(font_bold_name, font_bold_path))
    else:
        def _find_malgun_fonts(base_dir: Path) -> tuple[Path | None, Path | None]:
            normal = bold = None
            if not base_dir.is_dir():
                return None, None
            for candidate in base_dir.rglob('*malgun*.ttf'):
                lower_name = candidate.name.lower()
                if 'bold' in lower_name or 'bd' in lower_name:
                    bold = bold or candidate
                else:
                    normal = normal or candidate
                if normal and bold:
                    break
            return normal, bold

        font_root = Path('/usr/share/fonts')
        font_path, font_bold_path = _find_malgun_fonts(font_root)
        if font_path:
            pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
            if not font_bold_path:
                font_bold_path = font_path
            pdfmetrics.registerFont(TTFont(font_bold_name, str(font_bold_path)))
        else:
            font_name, font_bold_name = 'Helvetica', 'Helvetica-Bold'
    return font_name, font_bold_name


FONT_REG, FONT_BOLD = register_korean_font()


# ### 최종 통합 및 요약 생성 노드

# In[20]:


# ============================================================================
# 3. 최종 통합 및 요약 생성 노드
# ============================================================================
async def finalize_and_summarize(state: State) -> State:

    work_info = state["work_info"]

    # 최종 결과 통합
    final_result = {
        "title": work_info.get("title"),
        "target_format": work_info.get("target_format"),
        "target_genre": work_info.get("target_genre"),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "main_screen": state.get("main_screen", {}),
        "exp_market": state.get("exp_market", {}),
        "exp_creative": state.get("exp_creative", {}),
        "exp_visual": state.get("exp_visual", {}),
        "exp_world": state.get("exp_world", {}),
        "exp_business": state.get("exp_business", {}),
        "exp_production": state.get("exp_production", {})
    }

    state["final_result"] = final_result
    print("\n 최종 결과 통합 완료")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))

    # 요약 생성
    print("요약 생성 시작...")
    try:
        # [수정] safe_get을 사용하는 함수로 교체
        summary_inputs = build_summary_input(final_result) 
        summaries = await summarize_sections(summary_inputs)

        state["summary_exp_market"] = summaries.get("exp_market", "")
        state["summary_exp_creative"] = summaries.get("exp_creative", "")
        state["summary_exp_visual"] = summaries.get("exp_visual", "")
        state["summary_exp_world"] = summaries.get("exp_world", "")
        state["summary_exp_business"] = summaries.get("exp_business", "")
        state["summary_exp_production"] = summaries.get("exp_production", "")

        # final_result에도 요약 추가
        final_result.update({
            "summary_exp_market": state["summary_exp_market"],
            "summary_exp_creative": state["summary_exp_creative"],
            "summary_exp_visual": state["summary_exp_visual"],
            "summary_exp_world": state["summary_exp_world"],
            "summary_exp_business": state["summary_exp_business"],
            "summary_exp_production": state["summary_exp_production"],
        })

        print("요약 생성 완료")

    except Exception as e:
        print(f"요약 생성 중 오류 발생: {e}")
        traceback.print_exc()

    state["final_result"] = final_result
    state["pdf_path"] = ""

    return state


# ### IP 전략 보고서 PDF 생성 클래스

# In[45]:


def generate_pdf_from_strategy(strategy_data: Dict[str, Any], output_filename: str) -> bool:
    """전략 데이터로부터 PDF 생성"""
    if not strategy_data:
        print("전략 데이터가 비어있습니다.")
        return False

    try:
        gen = MiraeAssetIPReportGenerator(strategy_data)
        gen.build(output_filename)
        return True
    except Exception as e:
        print(f"PDF 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


# ###  PDF 생성 노드

# In[46]:


def generate_pdf_report(state: State) -> State:
    """최종 PDF 리포트 생성 (표지 포함)"""
    save_dir = "pdf_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # PDF 데이터 준비 - 표지 정보 포함
    pdf_data = {
        'title': state['work_info'].get('title', '제목 없음'),
        'target_format': state['work_info'].get('target_format', '-'),
        'target_genre': state['work_info'].get('target_genre', '-'),
        'main_screen': state.get('main_screen', {}),
        'exp_market': state.get('exp_market', {}),
        'exp_creative': state.get('exp_creative', {}),
        'exp_visual': state.get('exp_visual', {}),
        'exp_world': state.get('exp_world', {}),
        'exp_business': state.get('exp_business', {}),
        'exp_production': state.get('exp_production', {}),
        'summary_exp_market': state.get('summary_exp_market', ''),
        'summary_exp_creative': state.get('summary_exp_creative', ''),
        'summary_exp_visual': state.get('summary_exp_visual', ''),
        'summary_exp_world': state.get('summary_exp_world', ''),
        'summary_exp_business': state.get('summary_exp_business', ''),
        'summary_exp_production': state.get('summary_exp_production', ''),
        'source_details': state['work_info'].get('source_details', []),

        # 표지 정보 추가
        'operator_name': state.get('operator_name', '운영자 미지정'),
        'authors': state.get('authors', ['작가 미지정']),
        'publish_date': state.get('publish_date', datetime.now().strftime("%Y. %m. %d"))
    }

    try:
        # 2. 파일명 생성 (특수문자 제거)
        raw_title = state['work_info'].get('title', 'IP기획서')
        safe_title = "".join([c for c in raw_title if c.isalnum() or c in (' ', '.', '_')]).strip()
        filename = f"{safe_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"

        # 3. 경로 합치기 (폴더/파일명)
        output_path = os.path.join(save_dir, filename)

        # 4. PDF 생성기 호출
        generator = MiraeAssetIPReportGenerator(pdf_data)
        generator.build(output_path)

        # 5. 절대 경로 확인 (이게 핵심입니다!)
        abs_path = os.path.abspath(output_path)

        state['pdf_path'] = abs_path
        print(f"PDF 생성 성공!")
        print(f"저장 위치(절대경로): {abs_path}")


    except Exception as e:
        print(f"❌ PDF 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        state['pdf_path'] = ""

    return state


# In[47]:


side_main_top_layout_start = 228*mm

# -----------------------------------------------------------
# 2. PDF 생성 클래스 (Independent Flow 방식)
# -----------------------------------------------------------
class MiraeAssetIPReportGenerator:
    """IP 전략 보고서 PDF 생성 클래스"""

    def __init__(self, data):
        self.data = data
        self.primary_color = colors.HexColor('#F37021') 
        self.navy_color = colors.HexColor('#000000')    
        self.muted_color = colors.HexColor('#888888')   
        self.bg_grey = colors.HexColor('#F7F7F7')       
        self.bg_tag_color = colors.HexColor('#FFEAD9')  
        self.styles = self._create_styles()

        # [핵심] 사이드바 내용을 따로 저장할 리스트
        self.sidebar_story = []

    def _create_styles(self):
        s = getSampleStyleSheet()
        if FONT_REG in pdfmetrics.getRegisteredFontNames():
            s['Normal'].fontName = FONT_REG

        s.add(ParagraphStyle('RptMainTitle', fontName=FONT_BOLD, fontSize=18, leading=22, spaceAfter=2)) 
        s.add(ParagraphStyle('RptSubTitle', fontName=FONT_REG, fontSize=10, textColor=self.muted_color, spaceAfter=10))
        s.add(ParagraphStyle('RptSideTitle', fontName=FONT_BOLD, fontSize=9, spaceBefore=6, spaceAfter=3))
        s.add(ParagraphStyle('RptSideText', fontName=FONT_REG, fontSize=7, leading=10, textColor=self.navy_color))
        s.add(ParagraphStyle('RptSideItem', fontName=FONT_REG, fontSize=7, leading=10, textColor=self.navy_color, leftIndent=5))
        s.add(ParagraphStyle('RptTableCell', fontName=FONT_REG, fontSize=7.5, leading=9, textColor=self.navy_color))
        s.add(ParagraphStyle('RptTableHeader', fontName=FONT_BOLD, fontSize=8, leading=10, textColor=self.navy_color))
        s.add(ParagraphStyle('RptSideLabel', fontName=FONT_REG, fontSize=8, textColor=self.muted_color))
        s.add(ParagraphStyle('RptSideValue', fontName=FONT_BOLD, fontSize=8, alignment=TA_JUSTIFY))
        s.add(ParagraphStyle('RptSectionTitle', fontName=FONT_BOLD, fontSize=11, spaceBefore=12, spaceAfter=6))
        s.add(ParagraphStyle('RptBodyText', fontName=FONT_REG, fontSize=9, leading=14, alignment=TA_JUSTIFY, spaceAfter=8))
        s.add(ParagraphStyle('RptPageHeader', fontName=FONT_BOLD, fontSize=13, spaceAfter=10))
        s.add(ParagraphStyle('CoverTitle', fontName=FONT_BOLD, fontSize=24, leading=30, alignment=TA_CENTER, textColor=self.navy_color))
        s.add(ParagraphStyle('CoverInfo', fontName=FONT_REG, fontSize=11, leading=16, alignment=TA_CENTER, textColor=self.navy_color))
        return s

    # -------------------------------------------------------------------------
    # 배경 그리기 함수들
    # -------------------------------------------------------------------------
    def _draw_base_frame(self, canvas, doc):
        """기본 배경 (상단 라인 + 푸터)"""
        canvas.saveState()
        canvas.setStrokeColor(self.primary_color)
        canvas.setLineWidth(1.5)
        canvas.line(15*mm, 282*mm, 195*mm, 282*mm)

        canvas.setFont(FONT_BOLD, 10)
        canvas.setFillColor(self.navy_color)
        canvas.drawString(15*mm, 285*mm, "IP_SUM")
        canvas.setFont(FONT_REG, 8)
        canvas.setFillColor(self.muted_color)
        canvas.drawString(15*mm, 278*mm, "Equity Research | IP Expansion Strategy")

        date_str = self.data.get('updated_at', datetime.now().strftime("%Y. %m. %d"))

        try:
            date_str = date_str.replace('-', '. ')
        except:
            pass
        canvas.drawRightString(195*mm, 285*mm, f"{date_str}")

        canvas.setFont(FONT_REG, 7)
        canvas.setFillColor(self.muted_color)
        footer_text = "본 자료는 IP 확장 제안 참고용으로만 제공되며, 최종 의사결정은 제작사 본인에게 있습니다."
        canvas.drawString(15*mm, 10*mm, footer_text)
        canvas.drawRightString(195*mm, 10*mm, f"Page {doc.page}")
        canvas.restoreState()

    def _draw_sidebar_content(self, canvas, x, y, w, h):
        """사이드바 내용을 그리는 함수"""
        if not self.sidebar_story:
            return
        f = Frame(x, y, w, h, showBoundary=0)
        f.addFromList(self.sidebar_story, canvas)

    def _on_page_main_first(self, canvas, doc):
        """
        [완전 수정] 요약 섹션 첫 페이지 배경 (헤더 있음)
        """
        self._draw_base_frame(canvas, doc)

        # 데이터 가져오기
        title = self.data.get('title', '')
        main_screen = self.data.get('main_screen', {})

        # 기준 Y 좌표 (상단 라인)
        header_line_y = 282*mm 

        # -------------------------------------------------------------
        # 1. 오른쪽 상태표 (Info Table) 그리기
        # -------------------------------------------------------------
        info_data = [
            [Paragraph("Current Status", self.styles['RptSideLabel']), Paragraph("Planning", self.styles['RptSideValue'])],
            [Paragraph("Target Format", self.styles['RptSideLabel']), Paragraph(self.data.get('target_format', '-'), self.styles['RptSideValue'])],
            [Paragraph("Genre", self.styles['RptSideLabel']), Paragraph(self.data.get('target_genre', '-'), self.styles['RptSideValue'])]
        ]

        t_info = Table(info_data, colWidths=[20*mm, 25*mm])
        t_info.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))

        w_info, h_info = t_info.wrap(50*mm, 50*mm)

        # [수정] 상태표 위치도 제목에 맞춰 조금 내림 (상단 라인 아래 15mm 지점 시작)
        # 기존: header_line_y - 5*mm 
        # 변경: header_line_y - 15*mm (더 여유롭게)
        info_y = header_line_y - 15*mm - h_info
        t_info.drawOn(canvas, 150*mm, info_y)

        # -------------------------------------------------------------
        # 2. 제목 (Title) 그리기
        # -------------------------------------------------------------
        title_para = Paragraph(f"<b>{title}</b>", self.styles['RptMainTitle'])
        w_title, h_title = title_para.wrap(130*mm, 50*mm)

        # [핵심 수정 1] 상단 라인과 제목 사이 간격을 넓힘 (20mm)
        # 기존: header_line_y - 10*mm
        # 변경: header_line_y - 20*mm
        title_y = header_line_y - 20*mm - h_title + 10*mm # baseline 조정
        title_para.drawOn(canvas, 15*mm, title_y - 8*mm) 

        # -------------------------------------------------------------
        # 3. 설명글 (Entry Message) 그리기
        # -------------------------------------------------------------
        msg = main_screen.get('entry_message', '')
        if msg:
            msg_para = Paragraph(msg, self.styles['RptSubTitle'])
            w_msg, h_msg = msg_para.wrap(130*mm, 30*mm)

            # [핵심 수정 2] 제목과 메시지 사이 간격을 좁혀서 '당겨 올림'
            # 기존: (title_y - 8*mm) - h_msg - 5*mm
            # 변경: (title_y - 8*mm) - h_msg - 2*mm (아주 가깝게)
            msg_y = (title_y - 8*mm) - h_msg - 2*mm
            msg_para.drawOn(canvas, 15*mm, msg_y)

            content_start_y = msg_y
        else:
            content_start_y = title_y - 8*mm


        sidebar_start_y = side_main_top_layout_start

        # 높이 계산
        sidebar_height = sidebar_start_y - doc.bottomMargin
        self._draw_sidebar_content(canvas, doc.leftMargin, doc.bottomMargin, 60*mm, sidebar_height)

        # build 메서드에서 쓸 main_content_top 값을 저장해두면 좋겠지만,
        # 여기서는 build 메서드에서 직접 계산하거나 고정값을 조정해야 함.
        # 대략적인 계산: 282 - 20(제목간격) - 22(제목높이) - 12(메시지높이) - 5(본문간격) = 약 223mm

    def _on_page_main_cont(self, canvas, doc):
        """요약 섹션 이어지는 페이지 배경 (헤더 없음)"""
        self._draw_base_frame(canvas, doc)
        self._draw_sidebar_content(canvas, doc.leftMargin, doc.bottomMargin, 60*mm, doc.height)

    def _draw_cover_elements(self, canvas, doc):
        self._draw_base_frame(canvas, doc)

    # -------------------------------------------------------------------------
    # 콘텐츠 생성 함수들
    # -------------------------------------------------------------------------
    def _split_into_paragraphs(self, text, max_sentences=3):
        if not text: return []
        text = str(text)
        sentences = text.replace('. ', '.|').split('|')
        paragraphs = []
        current_para = []
        for s in sentences:
            current_para.append(s.strip())
            if len(current_para) >= max_sentences:
                paragraphs.append(' '.join(current_para))
                current_para = []
        if current_para:
            paragraphs.append(' '.join(current_para))
        return paragraphs

    def create_cover_page(self):
        story = []
        title = self.data.get('title', '제목 없음')
        cover_title = f"{title} 기획서"
        story.append(Spacer(1, 70*mm))
        story.append(Paragraph(cover_title, self.styles['CoverTitle']))
        story.append(Spacer(1, 90*mm))
        operator_name = self.data.get('operator_name', '김기획')
        authors = self.data.get('authors', [])
        authors_text = ", ".join(authors) if isinstance(authors, list) else str(authors)
        publish_date = self.data.get('created_at', datetime.now().strftime("%Y. %m. %d"))
        story.append(Paragraph(f"기획자: {operator_name}", self.styles['CoverInfo']))
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(f"작가: {authors_text}", self.styles['CoverInfo']))
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(f"발행일: {publish_date}", self.styles['CoverInfo']))
        story.append(NextPageTemplate('MainFirst')) 
        story.append(PageBreak())
        return story

    def create_three_frame_content(self):
        main_screen = self.data.get('main_screen', {})
        roadmap = main_screen.get('roadmap', {})
        identity_elements = main_screen.get('identity_elements', [])

        sb = self.sidebar_story
        sb.append(Paragraph("Investment Highlights", self.styles['RptSideTitle']))
        sb.append(Spacer(1, 2*mm))
        highlights = []
        for key in ['exp_market', 'exp_creative', 'exp_visual', 'exp_world', 'exp_business', 'exp_production']:
            summary = self.data.get(f"summary_{key}", "")
            if summary: highlights.append(f"<b>{key[4:].title()}:</b> {summary}")
        if not highlights: highlights = ["<b>주요 포인트 요약</b>"]
        sb.append(Paragraph("<br/>".join(highlights), self.styles['RptSideText']))


        source_details = self.data.get('source_details', [])

        print(f"DEBUG: source_details = {source_details}")

        if source_details:
            sb.append(Spacer(1, 5*mm))
            sb.append(Paragraph("Original IP Analysis", self.styles['RptSideTitle']))
            for work in source_details:
                sb.append(Spacer(1, 2*mm))
                w_title = work.get('work_title', '-')
                w_author = work.get('author_name', '-')
                sb.append(Paragraph(f"<b>{w_title}</b> ({w_author})", self.styles['RptSideText']))
                for item in work.get('items', []):
                    text = f"<font color='#F37021'><b>[{item.get('category', '-')}]</b></font> {item.get('keyword', '-')}"
                    sb.append(Paragraph(text, self.styles['RptSideItem']))
                sb.append(Spacer(1, 2*mm))

        sb.append(Spacer(1, 5*mm))
        sb.append(Paragraph("Key Identity", self.styles['RptSideTitle']))
        full_identity = identity_elements[0] if identity_elements else 'N/A'
        sb.append(Paragraph(full_identity, self.styles['RptSideText']))

        sb.append(Spacer(1, 5*mm))
        sb.append(Paragraph("핵심 실행 과제", self.styles['RptSideTitle']))
        phase1_obj = str(roadmap.get('phase_1', {}).get('objective', '')) if isinstance(roadmap, dict) else str(roadmap)
        sb.append(Paragraph(phase1_obj, self.styles['RptSideText']))

        # 본문 내용
        story = []
        story.append(NextPageTemplate('MainCont'))

        story.append(Paragraph("Review: 원천 IP 경쟁력 진단", self.styles['RptSectionTitle']))
        review_text = main_screen.get('review', '')
        for para in self._split_into_paragraphs(review_text):
            story.append(Paragraph(para, self.styles['RptBodyText']))

        story.append(Spacer(1, 5*mm))
        story.append(Paragraph("실행 마일스톤 및 주요 단계 (Execution Roadmap)", self.styles['RptSectionTitle']))

        roadmap_table_data = [[
            Paragraph('<b>단계</b>', self.styles['RptTableHeader']),
            Paragraph('<b>주요 목표</b>', self.styles['RptTableHeader']),
            Paragraph('<b>일정(E)</b>', self.styles['RptTableHeader'])
        ]]
        if isinstance(roadmap, dict):
            for phase_key in ['phase_1', 'phase_2', 'phase_3']:
                phase = roadmap.get(phase_key, {})
                if phase:
                    obj = phase.get('objective', '')
                    roadmap_table_data.append([
                        Paragraph(phase.get('title', ''), self.styles['RptTableCell']),
                        Paragraph(obj, self.styles['RptTableCell']),
                        Paragraph(phase.get('period', ''), self.styles['RptTableCell'])
                    ])

        t_roadmap = Table(roadmap_table_data, colWidths=[25*mm, 70*mm, 20*mm])
        t_roadmap.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),self.bg_grey),
            ('GRID',(0,0),(-1,-1),0.5,colors.lightgrey),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('ALIGN',(0,0),(-1,-1),'LEFT')
        ]))
        story.append(t_roadmap)

        story.append(Spacer(1, 5*mm))
        story.append(Paragraph("핵심 파트너십 및 협업 시너지", self.styles['RptSectionTitle']))
        for para in self._split_into_paragraphs(main_screen.get('partnership', '')):
            story.append(Paragraph(para, self.styles['RptBodyText']))

        story.append(NextPageTemplate('SingleTemplate')) 
        story.append(PageBreak())
        return story

    def create_detail_pages(self):
        story = []
        sections = [
            ("01. IP 원천 가치 및 시장성 분석", "exp_market"),
            ("02. 매체 최적화 각색 시나리오", "exp_creative"),
            ("03. 캐릭터 및 비주얼 가이드", "exp_visual"),
            ("04. 코어 메커니즘 및 세계관 자산", "exp_world"),
            ("05. 타겟팅 및 비즈니스 모델", "exp_business"),
            ("06. 제작 난이도 및 리소스 리포트", "exp_production")
        ]
        for title, key in sections:
            story.append(Paragraph(f"<font color='#F37021'>{title[:3]}</font>{title[3:]}", self.styles['RptPageHeader']))
            line = Drawing(180*mm, 1)
            line.add(Line(0, 0, 180*mm, 0, strokeColor=self.primary_color, strokeWidth=1.5))
            story.append(line)
            story.append(Spacer(1, 4*mm))

            content = self.data.get(key, {})
            if isinstance(content, dict) and key in content:
                content = content[key]

            if isinstance(content, dict):
                for k, v in content.items():
                    story.append(Paragraph(f"<b>{k}</b>", self.styles['RptSectionTitle']))
                    text = v if isinstance(v, str) else str(v)
                    for para in self._split_into_paragraphs(text):
                        story.append(Paragraph(para, self.styles['RptBodyText']))
            else:
                text = str(content)
                for para in self._split_into_paragraphs(text):
                    story.append(Paragraph(para, self.styles['RptBodyText']))
            story.append(PageBreak())
        return story

    def build(self, filename):
        doc = BaseDocTemplate(filename, pagesize=A4, 
                             leftMargin=15*mm, rightMargin=15*mm, 
                             topMargin=20*mm, bottomMargin=15*mm)

        main_content_top = side_main_top_layout_start

        # 1. MainFirst 프레임
        frame_summary_first = Frame(doc.leftMargin + 65*mm, doc.bottomMargin, 
                                 115*mm, main_content_top - doc.bottomMargin, 
                                 id='main_first', showBoundary=0)

        # 2. MainCont 프레임
        frame_main_cont = Frame(doc.leftMargin + 65*mm, doc.bottomMargin, 
                                115*mm, doc.height, 
                                id='main_cont', showBoundary=0)

        # 3. Single/Full 프레임
        frame_full = Frame(doc.leftMargin, doc.bottomMargin, 
                           doc.width, doc.height, id='full', showBoundary=0)

        frame_cover = Frame(doc.leftMargin, doc.bottomMargin, 
                            doc.width, doc.height, id='cover', showBoundary=0)

        template_cover = PageTemplate(id='Cover', frames=[frame_cover], onPage=self._draw_cover_elements)
        template_main_first = PageTemplate(id='MainFirst', frames=[frame_summary_first], onPage=self._on_page_main_first)
        template_main_cont = PageTemplate(id='MainCont', frames=[frame_main_cont], onPage=self._on_page_main_cont)
        template_single = PageTemplate(id='SingleTemplate', frames=[frame_full], onPage=self._draw_base_frame)

        doc.addPageTemplates([template_cover, template_main_first, template_main_cont, template_single])

        content = self.create_cover_page() + self.create_three_frame_content() + self.create_detail_pages()
        doc.build(content)


# ### 메인 실행 함수

# In[48]:


async def generate_ip_extension_strategy(
    work_info: Dict[str, Any],
    lorebooks: List[Dict[str, Any]],
    operator_name: str = "Unknown",
    authors: List[str] = None,
    publish_date: str = None,
    generate_pdf: bool = True
) -> tuple[Dict[str, Any], str]:

    print("\n" + "="*60)
    print("IP 확장 전략 생성 시작 (LangGraph)")
    print("="*60)

    # 1. 초기화
    analysis_results = await comparison_ip(lorebooks) # 실제 사용 시 주석 해제
    # analysis_results = {} 

    conflict_keywords = set()
    if "충돌" in analysis_results:
        for category, keywords in analysis_results["충돌"].items():
            # keywords가 리스트라고 가정하고 추가
            for k in keywords:
                if isinstance(k, str):
                    conflict_keywords.add(k)
                elif isinstance(k, dict) and 'keyword' in k: # 만약 객체로 나온다면
                    conflict_keywords.add(k['keyword'])

    # (3) lorebooks 리스트에서 충돌 키워드를 가진 항목 제거 (필터링)
    # 로어북의 'keyword'가 충돌 목록에 없으면 살아남음
    clean_lorebooks = [
        lb for lb in lorebooks 
        if lb.get('keyword') not in conflict_keywords
    ]

    print(f" 로어북 필터링 완료: 원본 {len(lorebooks)}개 -> 필터링 후 {len(clean_lorebooks)}개")

    initial_state = {
        "work_info": work_info,
        "lorebooks": clean_lorebooks,
        "analysis_results": analysis_results,
        "main_screen": {}, "exp_market": {}, "exp_creative": {}, "exp_visual": {},
        "exp_world": {}, "exp_business": {}, "exp_production": {}, 
        "final_result": {},
        "summary_exp_market": "", "summary_exp_creative": "", "summary_exp_visual": "",
        "summary_exp_world": "", "summary_exp_business": "", "summary_exp_production": "",
        "pdf_path": ""
    }

    # 2. 그래프 실행
    try:
        if 'graph' not in locals():
            graph = build_ip_strategy_graph()
        final_state = await graph.ainvoke(initial_state)
    except Exception as e:
        print(f"Graph execution failed: {e}")
        import traceback
        traceback.print_exc()
        return {}, ""

    # 3. 결과 추출
    final_result = final_state.get("final_result", {})

    # 4. PDF 생성 및 데이터 정리
    if generate_pdf and final_result:
        print("PDF 생성을 위한 데이터 병합 중...")

        # ------------------------------------------------------------------
        # [핵심] 데이터 이중 중첩 벗겨내기 (Flattening)
        # AI가 {"exp_market": {"exp_market": {...}}} 형태로 준 경우를 복구
        # ------------------------------------------------------------------
        target_keys = ['exp_market', 'exp_creative', 'exp_visual', 'exp_world', 'exp_business', 'exp_production']

        for key in target_keys:
            # 해당 키가 있고 딕셔너리 형태일 때
            if key in final_result and isinstance(final_result[key], dict):
                # 그 안에 자기 자신과 똑같은 키가 또 있으면 (중첩)
                if key in final_result[key]:
                    print(f"  - {key} 데이터 중첩 해제 완료")
                    # 껍질을 벗겨서 덮어씌움
                    final_result[key] = final_result[key][key]

        # ------------------------------------------------------------------
        # 메타데이터 및 사이드바용 요약 병합
        # ------------------------------------------------------------------
        final_result['operator_name'] = operator_name
        final_result['authors'] = authors if authors else []
        final_result['updated_at'] = publish_date if publish_date else datetime.now().strftime("%Y. %m. %d")
        final_result['source_details'] = work_info.get('source_details', [])

        # 요약 정보가 final_result에 누락되었을 경우를 대비해 state에서 가져옴
        for key in target_keys:
            summary_key = f"summary_{key}"
            if summary_key in final_state and final_state[summary_key]:
                final_result[summary_key] = final_state[summary_key]

        # PDF 파일명 생성
        safe_title = "".join([c for c in work_info.get('title', 'IP_Strategy') if c.isalnum() or c in (' ', '.', '_')]).strip()
        output_filename = f"{safe_title.replace(' ', '_')}_Report.pdf"
        
        # [수정 1] 우리가 약속한 PDF_DIR 경로와 파일명을 합쳐줘!
        full_pdf_path = PDF_DIR / output_filename

        # PDF 생성기 호출
        try:
            gen = MiraeAssetIPReportGenerator(final_result)
            # [수정 2] 파일명이 아니라 전체 경로(full_pdf_path)를 넣어줘!
            gen.build(str(full_pdf_path)) 
            
            # [수정 3] state에도 전체 경로를 문자열로 저장!
            final_state["pdf_path"] = str(full_pdf_path)
            print(f"✅ PDF 생성 성공: {full_pdf_path}")
        except Exception as e:
            print(f"❌ PDF 생성 실패: {e}")
            traceback.print_exc()
            final_state["pdf_path"] = "" # 실패했을 땐 빈 값을 넣어주는 게 안전해!

    print("="*60)
    print("IP 확장 전략 생성 완료")
    
    # [수정 4] 마지막 리턴도 깔끔하게 저장된 경로 그대로 돌려주자!
    return final_result, final_state.get("pdf_path", "")


# ### 6대 구성요소 요약 

# In[49]:


SUMMARY_SYSTEM_PROMPT = """
당신은 글로벌 콘텐츠 기업 및 투자기관에서
IP 확장 전략을 검토하는 투자 심사용 요약 전문 애널리스트입니다.

입력된 텍스트는 IP 확장 전략의 상세 실행 문서 일부입니다.

이를 기반으로 아래 기준에 따라
‘키워드 중심 압축 요약’을 수행하십시오.

[요약 기준]

- 문장 서술 완전 금지
- 명사형 키워드만 사용
- 실행 판단에 필요한 정보만 유지
- 시장 / 제작 / 비즈니스 관점 핵심만 추출
- 수치, 구조, 전략 요소 우선
- 감성적·서사적 표현 완전 제거
- 원문 의미 왜곡 금지

[출력 강제 규칙 — 반드시 준수]

- 각 섹션 출력은 **반드시 한 줄**
- 줄바꿈(\n) 절대 금지
- 키워드는 **정확히 3~4개만 출력**
- 키워드는 쉼표(,)로만 구분
- 조사, 접속사, 설명 표현 사용 금지
- 섹션당 총 길이 **50자 이내**
- 초과 정보는 중요도가 낮은 순서로 제거
- 제목, 머리말, 설명 문구 출력 금지
- 반드시 순수 텍스트만 출력
"""

def safe_get(data: Dict[str, Any], section_key: str, content_key: str) -> str:
    """
    데이터가 이중 중첩되어 있어도 안전하게 텍스트를 추출하는 헬퍼 함수
    반환값은 무조건 문자열(str)임을 보장하여 에러 방지
    """
    # 1. 1차 접근
    section = data.get(section_key, {})

    # 2. 이중 중첩 감지 및 해제 ({'exp_market': {'exp_market': ...}})
    if section_key in section and isinstance(section[section_key], dict):
        section = section[section_key]

    # 3. 값 추출
    value = section.get(content_key, "")

    # 4. 문자열이 아닌 경우(dict, list 등) 강제로 문자열 변환
    if not isinstance(value, str):
        return str(value)

    return value

def build_summary_input(final_result: Dict[str, Any]) -> Dict[str, str]:
    """요약 모델 입력 구성 (safe_get 사용)"""
    return {
        "exp_market": (
            f"타겟 분석: {safe_get(final_result, 'exp_market', 'target_analysis')}\n"
            f"포지셔닝: {safe_get(final_result, 'exp_market', 'positioning_strategy')}"
        ),
        "exp_creative": (
            f"스토리 확장: {safe_get(final_result, 'exp_creative', 'story_expansion')}\n"
            f"설정 충돌 해결: {safe_get(final_result, 'exp_creative', 'conflict_resolution')}"
        ),
        "exp_visual": (
            f"비주얼 스타일: {safe_get(final_result, 'exp_visual', 'visual_style')}\n"
            f"아트 디렉션: {safe_get(final_result, 'exp_visual', 'art_direction')}"
        ),
        "exp_world": (
            f"세계관 확장: {safe_get(final_result, 'exp_world', 'world_expansion')}\n"
            f"자산 최적화: {safe_get(final_result, 'exp_world', 'asset_maximization')}"
        ),
        "exp_business": (
            f"수익 모델: {safe_get(final_result, 'exp_business', 'revenue_model')}\n"
            f"사업 전략: {safe_get(final_result, 'exp_business', 'business_strategy')}"
        ),
        "exp_production": (
            f"기술 스택: {safe_get(final_result, 'exp_production', 'tech_stack')}\n"
            f"플랫폼 전략: {safe_get(final_result, 'exp_production', 'platform_strategy')}"
        )
    }

async def summarize_sections(summary_inputs: Dict[str, str]) -> Dict[str, str]:
    summaries = {}

    for section, text in summary_inputs.items():
        user_msg = f"""
        다음은 IP 확장 전략 문서의 일부이다.
        키워드 중심으로 

        [입력 텍스트]
        {text}
        """
        summary = await genai(SUMMARY_SYSTEM_PROMPT, user_msg)
        summaries[section] = summary.strip() if summary else ""

    return summaries


# In[ ]:





# ## Rang Graph

# ### 카테고리

# In[50]:


cat_builder = StateGraph(SettingState)
cat_builder.add_node("Categories",Categories)

cat_builder.set_entry_point("Categories")
cat_builder.add_edge("Categories",END)

cat = cat_builder.compile()


# ### 상세설정

# In[51]:


builder = StateGraph(SettingState)
builder.add_node("charter",charter)
builder.add_node("rule",rule)
builder.add_node("place",place)
builder.add_node("event",event)
builder.add_node("item",item)
builder.add_node("group",group)

builder.set_entry_point("charter")
builder.add_edge("charter","rule")
builder.add_edge("rule","place")
builder.add_edge("place","event")
builder.add_edge("event","item")
builder.add_edge("item","group")
builder.add_edge("group",END)

detail = builder.compile()


# ### IP 확장 전략 생성

# In[52]:


def build_ip_strategy_graph():
    """IP 확장 전략 생성 그래프 구축"""

    builder = StateGraph(State)

    # 노드 추가
    builder.add_node("main_screen", generate_main_screen)
    builder.add_node("exp_market", generate_exp_market)
    builder.add_node("exp_creative", generate_exp_creative)
    builder.add_node("exp_visual", generate_exp_visual)
    builder.add_node("exp_world", generate_exp_world)
    builder.add_node("exp_business", generate_exp_business)
    builder.add_node("exp_production", generate_exp_production)
    builder.add_node("finalize_and_summarize", finalize_and_summarize)
    builder.add_node("generate_pdf", generate_pdf_report)

    # 엣지 연결
    builder.set_entry_point("main_screen")
    builder.add_edge("main_screen", "exp_market")
    builder.add_edge("exp_market", "exp_creative")
    builder.add_edge("exp_creative", "exp_visual")
    builder.add_edge("exp_visual", "exp_world")
    builder.add_edge("exp_world", "exp_business")
    builder.add_edge("exp_business", "exp_production")
    builder.add_edge("exp_production", "finalize_and_summarize")
    builder.add_edge("finalize_and_summarize", "generate_pdf")
    builder.add_edge("generate_pdf", END)

    return builder.compile()


# In[53]:


async def generate_ip_extension_strategy(
    work_info: Dict[str, Any],
    lorebooks: List[Dict[str, Any]],
    operator_name: str = "운영자",
    authors: List[str] = None,
    publish_date: str = None,
    generate_pdf: bool = True
) -> tuple[Dict[str, Any], str]:
    """
    IP 확장 전략 생성 (LangGraph 버전)

    총 13회 AI 호출:
    - 1회: main_screen
    - 6회: exp_market, exp_creative, exp_visual, exp_world, exp_business, exp_production
    - 6회: 각 섹션별 요약 생성

    Args:
        work_info: 작품 정보
        lorebooks: 로어북 리스트
        operator_name: 기획자/운영자 이름 (기본값: "운영자")
        authors: 작가 이름 리스트 (기본값: None -> ["작가 미지정"])
        publish_date: 발행일 (기본값: None -> 오늘 날짜)
        generate_pdf: PDF 생성 여부 (기본값: True)

    Returns:
        tuple: (전략 데이터 딕셔너리, PDF 파일 경로)
    """

    print("\n" + "="*60)
    print("IP 확장 전략 생성 시작 (LangGraph)")
    print("="*60)

    # 기본값 설정
    if authors is None:
        authors = ["작가 미지정"]
    elif isinstance(authors, str):
        authors = [authors]

    if publish_date is None:
        publish_date = datetime.now().strftime("%Y. %m. %d")

    # 설정 분석 (comparison_ip 함수는 기존 코드에 정의되어 있다고 가정)
    analysis_results = await comparison_ip(lorebooks)

    # 초기 State 구성
    initial_state: State = {
        "work_info": work_info,
        "lorebooks": lorebooks,
        "analysis_results": analysis_results,
        "main_screen": {},
        "exp_market": {},
        "exp_creative": {},
        "exp_visual": {},
        "exp_world": {},
        "exp_business": {},
        "exp_production": {},
        "final_result": {},
        "summary_exp_market": "",
        "summary_exp_creative": "",
        "summary_exp_visual": "",
        "summary_exp_world": "",
        "summary_exp_business": "",
        "summary_exp_production": "",
        "pdf_path": "",

        # 표지 정보
        "operator_name": operator_name,
        "authors": authors,
        "publish_date": publish_date
    }

    # 그래프 실행 (build_ip_strategy_graph 함수는 기존 코드에 정의되어 있다고 가정)
    graph = build_ip_strategy_graph()
    final_state = await graph.ainvoke(initial_state)

    print("="*60)
    print("IP 확장 전략 생성 완료")
    print(f"기획자: {operator_name}")
    print(f"작가: {', '.join(authors)}")
    print(f"발행일: {publish_date}")
    if final_state.get("pdf_path"):
        print(f"PDF 저장 위치: {final_state['pdf_path']}")
    print("="*60 + "\n")

    return final_state["final_result"], final_state.get("pdf_path", "")


# # FastAPI

# In[19]:


redis_client: redis.Redis = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, pool
    
    # DB Pool 및 Redis 객체 생성
    pool = await create_pool()
    # redis_client = redis.Redis(
    #     host="127.0.0.1",
    #     port=6379,
    #     decode_responses=True
    # )
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )

    try:
        # 실제 연결 상태 확인
        await redis_client.ping()
        print("Redis connected and verified")
    except Exception as e:
        print(f"Redis connection error: {e}")

    yield

    # 자원 해제
    if pool:
        await pool.close()
    if redis_client:
        await redis_client.close()
    print("Resources closed")



app = FastAPI(lifespan=lifespan)
# users : Dict[str, SettingState] = {}

origins = ["https://ai0917-kt-aivle-shool-8th-bigprojec-eta.vercel.app",
          "http://localhost:8080"]
app.add_middleware(
CORSMiddleware,
allow_origins=origins,
allow_credentials=True,
allow_methods=["GET", "POST", "PUT", "DELETE"],  
allow_headers=origins,
)
@app.get("/")
def connect_check():
    return "AI 서버와 연결되었습니다."




#원문 저장 
@app.post("/novel_save")
async def process_novel_save(data : dict = Body(...)):
    user_id = data.get("user_id")
    work_id = data.get("work_id")
    ep_num = data.get("ep_num")
    txt = data.get("txt")
    return await save_text(user_id, work_id, ep_num, txt)

# 원문 읽기
@app.get("/novel_read")
async def process_novel_read(
    user_id: str = Query(...), 
    work_id: str = Query(...), 
    ep_num: int = Query(...)
):
    # 기존 코드의 txt 변수가 정의되어 있지 않아 에러가 날 수 있으니 확인이 필요해요!
    return await open_text(user_id, work_id, ep_num)

#설정 저장
@app.post("/setting_save")
async def process_setting_save(data : dict = Body(...)):
    user_id = data.get("user_id")
    work_id = data.get("work_id")
    lore_id = data.get("lore_id")
    setting = data.get("data")
    return save_json(user_id,work_id,lore_id,setting)

# 설정 읽기
@app.get("/setting_read")
async def process_setting_read(
    user_id: str = Query(...), 
    work_id: str = Query(...), 
    lore_id: str = Query(...)
):
    return open_json(user_id, work_id, lore_id)

#카테고리 추출 
@app.post("/categories")
async def process_cat(data : dict = Body(...)):
    initial_state : SettingState = {
        "txt": await open_text(data.get("user_id"),data.get("work_id"),data.get("ep_num")),
        "ep_num" : data.get("ep_num"),
        "subtitle" : data.get("subtitle"),
        "check" : dict(),
        "work_id" : data.get("work_id"),
        "user_id" : data.get("user_id"),
        "인물" : dict(),
        "세계" : dict(),
        "장소" : dict(),
        "사건" : dict(),
        "물건" : dict(),
        "집단" : dict(),
    }
    if initial_state['txt'] == "failed":
        return "Error 원문이 저장되어 있지 않습니다."
    work_id = initial_state.get('work_id')
    result = await cat.ainvoke(initial_state)
    # users[work_id] = result
    await redis_client.set(
    f"user_state:{work_id}",
    json.dumps(result, ensure_ascii=False),
    ex=3600  # 1시간 TTL
    )
    return result.get("check",{})

@app.post("/setting") # 임베딩 벡터 반환하는 코드도 작성하기
async def process_get_setting(data : dict = Body(...)):
    work_id = data.get("work_id")
    categories = data.get("check", [])
    data = await redis_client.get(f"user_state:{work_id}")
    if not data:   # 존재하는 세션인지 확인
        raise HTTPException(status_code=404, detail="세션이 만료되었거나 존재하지 않습니다.")
    # user_state = users[work_id]  #기존의 데이/터 가져오기
    user_state = json.loads(data)
    user_state["check"] = categories
    user_state = await detail.ainvoke(user_state)
    result = await comparison(user_state)
    return result 



# 1. 수동 신규 저장 API (충돌 비교 X)
@app.post("/lorebook_insert")
async def process_manual_insert(data: dict = Body(...)):
    try:
        global pool # 전역 DB 풀 참조

        universe_id = data.get("universe_id") 
        work_id = data.get("work_id")
        user_id = data.get("user_id")
        keyword = data.get("keyword")
        category = data.get("category")
        ep_num = data.get("ep_num")
        setting = data.get("setting")

        # 인자 개수와 순서에 맞춰 pool을 첫 번째로 전달
        await db_insert(
            pool, universe_id, work_id, user_id, 
            keyword, category, ep_num, setting
        )
        
        return {"status": "success", "message": "수동 저장이 완료되었습니다."}
    except Exception as e:
        print(f"Error in manual_insert: {e}")
        raise HTTPException(status_code=500, detail=str(e))



pool = None

# 2. 수동 수정 API (충돌 비교 X)
@app.post("/lorebook_update")
async def process_manual_update(data: dict = Body(...)):
    try:
        # 1. 전역 변수 pool을 사용하겠다고 명시
        global pool 

        # 만약 어떤 이유로 pool이 생성되지 않았다면 에러
        if pool is None:
            raise HTTPException(status_code=500, detail="DB 커넥션 풀이 준비되지 않았습니다.")

        lore_id = data.get("lore_id")
        universe_id = data.get("universe_id")
        work_id = data.get("work_id")
        user_id = data.get("user_id")
        keyword = data.get("keyword")
        category = data.get("category")
        ep_num = data.get("ep_num")
        setting = data.get("setting")

        # 2. 이제 전역 변수 pool을 db_update에 첫 번째 인자로 전달
        result = await db_update(pool, lore_id, universe_id, work_id, user_id, keyword, category, ep_num, setting)

        if result and "실패" in result:
             raise HTTPException(status_code=500, detail=result)

        return {"status": "success", "message": "수동 수정이 완료되었습니다."}
    except Exception as e:
        print(f"Error in manual_update: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/comparison") 
async def process_get_comparison(data : dict = Body(...)):
    try:
        result = await comparison(data)
        print(data)
    except:
        print("comparison 에러")
    return result 

@app.post("/get_vector")
async def process_get_vector(data : dict = Body(...)):
    original = data.get("original")
    txt_original = f"{original}"
    return await get_vector(txt_original)

#유사도 기반 vectorDB 검색
@app.post("/userQ")
async def process_userQ(data : dict = Body(...)):
    category = data.get("category")
    user_query = data.get("user_query")
    user_id = data.get("user_id")
    work_id = data.get("work_id")
    sim = data.get("sim")
    limit = data.get("limit")
    return await userQuestion(category,user_query,user_id,work_id,sim,limit)


@app.post("/dbupsert")
async def process_dbupsert(data : dict = Body(...)):
    work_id = data.get('work_id')
    user_id = data.get('user_id')
    universe_id = data.get('universe_id')
    setting = data.get('setting')
    try:
        insert_setting = setting.get('신규 업로드')
        await lorebooks_insert(universe_id,work_id,user_id,insert_setting)
        edit_settings = setting.get('설정 결합')
        await lorebooks_update(edit_settings)
        # users.pop(work_id,None)
        await redis_client.delete(f"user_state:{work_id}")
    except Exception as e:
        return e
    return "성공"

@app.post("/relationship")
async def process_relationship(data : dict = Body(...)):
    work_id = data.get('work_id')
    user_id = data.get('user_id')
    target = data.get('target',"*")
    result = await relationship_select(target,work_id,user_id)
    result = await Relationship(result)
    return result

@app.post("/timeline")
async def process_timeline(data : dict = Body(...)):
    work_id = data.get('work_id')
    user_id = data.get('user_id')
    target = data.get('target')
    result = await timeline_select(target,work_id,user_id)
    result = await timeline(result)
    return result



# 다른 작품 설정 충돌 검사 
@app.post("/iplorebook")
async def process_iplorebook_check(data: dict = Body(...)):
    lorebooks = data.get("lorebooks")

    if not isinstance(lorebooks, list):
        raise HTTPException(status_code=400, detail="lorebooks must be a list")

    if len(lorebooks) == 0:
        raise HTTPException(status_code=400, detail="lorebooks cannot be empty")
    try:

        processed_lorebooks = _process_lorebooks(lorebooks)
        result = await comparison_ip(processed_lorebooks)
        return {
            "result": result,
            "processed_lorebooks": processed_lorebooks 
        }
    except Exception as e:
        # 에러 발생 시 상세 내용 반환
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ipproposal")
async def create_and_retrieve_ip_proposal(data: dict = Body(...)):

    # 1. 데이터 확인 
    ipproposal_id = data.get("id")
    processed_lorebooks = data.get("processed_lorebooks", [])

    if not ipproposal_id:
        raise HTTPException(status_code=400, detail="id is required")

    if not processed_lorebooks:
        raise HTTPException(status_code=400, detail="lorebooks  is required")

    try:
        # 3. DB에서 기획서 정보 조회
        proposal_data = await select_ip_proposal(ipproposal_id)
        if not proposal_data:
            raise HTTPException(status_code=404, detail=f"Proposal ID {ipproposal_id} not found")
        match_author_ids = proposal_data.get('match_author_ids', [])



        result, pdf_path = await generate_ip_extension_strategy(
            work_info=proposal_data,
            lorebooks=processed_lorebooks, # 전처리된 로어북 전달
            operator_name=proposal_data.get('manager_name', 'Unknown'),
            authors=proposal_data.get('author_names', []),
            publish_date=str(proposal_data.get('updated_at', datetime.now()))[:10]
        )
        # 4. 생성된 결과와 파일 정보를 DB에 저장
        file_name_only = Path(pdf_path).name if pdf_path else ""

        if result and file_name_only:
            # 테이블의 file_path 컬럼에는 이제 파일명만 예쁘게 저장될 거야.
            await update_ip_proposal_results(ipproposal_id, result, file_name_only, match_author_ids)
        else:
            print("경고: 결과 데이터 또는 PDF 경로가 없어 DB 업데이트를 건너뜁니다.")
        # 5. 결과 반환
        return {
            "success": True, 
            "pdf_path": pdf_path, 
            "data": result
        }

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# 현재 실행 중인 파일(final.py)의 위치를 기준으로 폴더 설정!
BASE_DIR = Path(__file__).parent 
PDF_DIR = BASE_DIR / "pdf_results"

PDF_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/ipproposal")
async def get_ip_proposal(path: str = Query(..., description="PDF 파일명 또는 경로")):
    """
    DB에 저장된 path 값을 받아 실제 파일이 있는 PDF_DIR에서 파일을 찾아 반환합니다.
    """
    # 1. 입력받은 path에서 파일명만 추출 (Path Traversal 공격 방지 및 경로 정규화)
    filename = Path(path).name

    # 2. 우리 서버의 실제 보관함(PDF_DIR)과 파일명 조립
    pdf_path = (PDF_DIR / filename).resolve()

    print(f"🔍 파일을 찾는 위치: {pdf_path}")

    # 3. 보안 체크: 찾은 파일이 반드시 PDF_DIR 안에 있는지 확인!
    if not pdf_path.exists() or PDF_DIR.resolve() not in pdf_path.parents:
        print(f"❌ 파일을 찾을 수 없거나 접근 권한이 없음: {pdf_path}")
        raise HTTPException(status_code=404, detail="PDF not found")

    # 4. 파일 반환
    return FileResponse(
        path=pdf_path, 
        media_type="application/pdf", 
        filename=pdf_path.name
    )






# # 테스트함수

# In[55]:


# teststate = {
# "txt" :  """죄수들을 가둬 놓은 동굴 너머에는 거대한 구멍이 있었다. 가로만 30미터에 달하는 이 거친 수직 통로는 깊이가 수백 미터는 될 것 같았다.

# 케인은 가장자리에 서서 내려다보았다. 엄청난 범위의 암반이... '무언가'에 의해 뜯겨나간 채였다. 아르마다 슬링 전함의 주포로도 행성의 표면을 이렇게 깔끔하게 날려버릴 수는 없었다.

# 사라진 부분은 어디에 있는 것일까? 아예 없애버린 건가?

# "저 아래야." 나쿠리가 말했다.

# 케인은 수직 통로 안쪽 벽의 울퉁불퉁한 표면을 타고 내려가기 시작했다. 가까이서 보니 열기로 인해 생성된 것처럼 보였다. 튀어나온 바위는 윤이 나는 분홍색이었고, 잘 닦인 보석처럼 반짝거렸다. 하지만 위쪽 표면에는 두꺼운 모래층이 덮여 있었다. 이 구멍은 오래전에, 어쩌면 수천 년 전에 만들어진 것일지도 몰랐다. 불현듯, 케인의 머릿속에 뜨거운 금속 덩어리가 빙하에 떨어져 순식간에 얼음을 녹이며 깊은 구멍을 만들고, 금속이 지나간 자리가 다시 얼어붙어 반짝이는 장면이 떠올랐다.

# 하지만 암반에 구멍을 낼 수 있을까?

# 케인은 내려가면서 인터페이스로 탐사 스캔을 작동했다. 뒤따르던 나쿠리는 케인이 놀라서 한숨을 내뱉는 소리를 들었다.

# "역시." 그가 말했다.

# "이 결과가 정확한 건가?" 케인이 중얼거렸다.

# "그런 것 같아."

# "이건… 말이 안 돼." 케인은 말하며 인터페이스의 스캔을 다시 작동했다.

# "아니, 그럴 리 없어."

# "이건 마치..." 케인은 쉽게 설명할 방법을 찾지 못했다. 양자의 흔적은 기묘했다. 마치 다른 현실, 다른 공간 차원의 일부가 이 아이오난의 산과 순간적으로 교차하여 이곳의 존재를 완전히 지워버리고, 텅 빈 상처를 남겨둔 것 같았다.

# 미지의 무언가에 의해 찢겨나간 상처 말이다.

# "내가 왜 오디널을 찾았는지 알겠지?" 나쿠리가 물었다.

# 케인은 대답하지 않았다. 추측하는 중이었다. '공간 간 충돌의 결과였을까? 양자 이상 현상? 계획 아니면 우연? 이런 현상은 이론적으로만 가능하다. 아니면 슬링 드라이브 실패로 일어나는 희소하고 비극적인 결과일 것이다. 이것은 다중 우주 명제를 입증하는 증거가 될 수도 있다... '

# 나쿠리가 옳았다. 이것은 오디널이 '해야 할' 일이었고, 이미 높아진 케인의 직위도 크게 오를 것이었다. 획기적인 발견이었다. 데막시아 제국에서 가장 유명한 사람이 될 수 있는, 정말로 초고속 승진 가도를 달릴 수 있는 일이었다.

# 케인은 잠시 멈칫했다. 그것은 충격적인 생각이었다. 여기에서 처리해야 할 일이자 오디널의 의무였다. 정보를 평가하고, 분석하고, 수집하는 모든 업무는 제국의 이익을 위한 것이다. 그리고 이 공적을 세운 자의 이름을...

# 새로운 생각이 케인의 마음속에 스며들었다. 야망 어린 생각이 그를 방해했다. 조사 진행 계획을 세우려면 나쿠리와 의논해야 한다는 사실은 알고 있었다.

# 하지만 그러고 싶지 않았다. 혼자 진행하고 싶었다. 아무도, 심지어 나쿠리조차도 들이고 싶지 않았다. 어느 누구에게도 그럴 만한 자격이—

# 케인은 생각을 정리했다. 신디케이트, 기사단... 모두가 이곳으로 모인 것은 당연한 일이었다. 이건 엄청난 선물이었다. 다만...

# "...그들은 어떻게 알았지?" 케인이 질문했다.

# "뭐라고?"

# "난 네가 불러서 이곳에 왔어. 넌 기사단을 쫓다가 왔고. 그러면 기사단은 어떻게 오게 된 걸까?"

# "기사단도 알고 있었다고…?" 나쿠리가 조심스럽게 되물었다.

# "누가 알려준 건데?"

# "비밀 거래, 금기시된 설화... 죄다 말이 안 돼. 아니면 전설이나 신화라든가... 모르겠어, 보물 지도라도 있는 건가?"

# 이 말은 케인의 귀에도 공허하게 들렸다. 혹시, 만약, 누군가가 과거에 이곳을 발견했다면 그냥 지나치지 않았을 것이다. 그렇다면 이곳은 신성한 장소나 성지가 되었거나, 새로운 문화를 탄생시켰거나, 누군가를 황제로 추대했거나... 새로운 제국의 초석이 되었을지도 모른다.

# 아니다. 아무도 몰랐을 것이다. 기사단이 이곳에 온 것은 그저... 본능적인 일이었을 것이다.

# "그럼 신디케이트는?" 케인이 나쿠리에게 물었다.

# "신디케이트가 왜?"

# 케인은 생각했다. 자고는 이미 알고 있었다. 그 추잡한 기회주의자는 기사단의 존재조차 눈치채지 못했다. 자고가 여기 온 이유는 이것이었다. 게다가 자고는 모든 위험을 감수할 정도로 집요한 모습을 보였다. 대제국군과 마찰을 일으킬 정도로 말이다.

# 그는 '무언가'의 부름을 받고 드넓은 우주를 가로질러 이 변두리까지 왔다.

# 케인의 손바닥이 축축해졌다. 마지막 몇 미터를 더 미끄러져 내려가는 동안, 불안감은 점점 커졌다. 구덩이 바닥에 뭔가가 있었다. 마치 바닥에 녹아든 것처럼 보였다.

# "이건 도대체..."

# "우린 저게 원인일 거라고 생각했어." 나쿠리가 말했다. "이곳에 떨어져 구멍을 만든 거지…"

# 그의 목소리가 조금씩 작아졌다.

# "혹시 건드렸나?" 케인이 물었다.

# "아니, 아무도. 그럴 엄두도 못 냈지."

# 케인은 쭈그려 앉았다. 그 물체는 얕은 암석 단층에 묻힌 시커먼 화석처럼 보였다. 상상할 수도 없는 머나먼 고대에 묻혔던 뼈가 막 빛을 받아 드러난 것 같았다. 케인은 살짝 휘어진, 아름답게 장식된 긴 손잡이를 분간할 수 있었다. 머리에 달린 것은 거대한 날이었다. 인터페이스가 식별할 수 없는 미지의 금속으로 벼려진 손잡이와 날은 분명히 인간형 종족에 맞춰진 비율이었다.

# 낫이었다. 무기 말이다. 지금껏 알려진 어떤 문명에서도 발견된 적 없는 굉장한 유물이었다.

# 케인은 극도의 아름다움과 기괴함이 어떻게 동시에 공존할 수 있는지 궁금해졌다.

# 그리고 낮은 웃음소리가 들렸다. "무슨 일이야?" 케인이 나쿠리를 바라보며 물었다.

# "아무 말도 안 했는데." 나쿠리가 대답했다.

# 케인이 인터페이스를 건드려 보았지만 먹통이었다.

# "너무 깊이 들어왔나 보군." 나쿠리가 말했다. "이곳의 뭔가가 통신을 방해하고 있어."

# "올라가." 케인이 말했다. "'프랙털 쉬어'에 신호를 보내서 과학팀에게 탐지 장비를 모두 챙겨 모이라고 해. 두 시간 내로 이곳에 데려와. 여길 샅샅이 분석해서 마지막 정보 한 조각까지 남김없이 뽑아낸다."

# 나쿠리는 고개를 끄덕였지만 움직이지 않았다. "너 변했구나."

# "무슨 의미지?"

# "이제 오디널 다 됐군. 네 말투—"

# 케인이 코웃음 치며 말했다. "이럴 시간이 없어."

# "신디케이트 앞에서 한 행동은 뭐였는데? 덕분에 난 부하 네 명을 잃었어. 죽지 않을 수 있었던 네 명이 네 허세 때문에 죽었다고."

# "복잡한 상황이었잖아."

# "본대에 요청해 싹 쓸어 버릴 수도 있었어. 하지만 넌 네 오만함에 취해 있었지. 오디널 각하."

# "결국 필요한 정보를 얻었어." 케인이 말했다.

# "그리고 네 명이 죽었지."

# "사령관, 가서 함선에 신호를 보내. 두 번 말하지 않겠다."

# 나쿠리는 머뭇거렸다. "내가 널 부른 이유는... 그래, 내가 널 부른 건 내 것이 아니라는 걸 알았기 때문이야. 내 권한 밖이니까. 그러자 네 생각이 나더군. 너라면 뭘 해야 할지 알고 있을 것 같았지. 네겐 자격이 있을 거라고 생각했어."

# "'자격'이라고?"

# "이걸 차지할 자격 말이야! 나? 난 아니야, 난 그럴 만한 자격이…" 나쿠리는 케인을 바라보았다. "하지만 너라면 가능할 거라 생각했지. 그게 제국을 위한, 내 친구를 위한 도리인 줄 알았어. 하지만 이제 확실해졌군. 네가 어떤 존재인지. 네가 어떻게 변했는지 알아버렸으니까."

# 죽여 버려.

# 케인은 주위를 두리번거렸다. 누군가가 그에게 말을 걸었다.

# "여긴 우리뿐인가?" 케인이 속삭였다.

# "뭐라고?" 분노한 나쿠리가 되물었다.

# "사령관, 이곳에 보초를 배치했나?"

# "아니."

# "그렇다면 방금 말한 건 누구지?"

# "아무도 말하지 않았어!" 나쿠리는 딱딱하게 내뱉었다. "도대체 왜 그래? 이젠 나도 네가 누구인지 모르겠어!"

# "가서 함선에 신호를 보내. 지금 당장. 끝나는 대로 돌아와서 보고하도록."

# 나쿠리는 케인을 한번 노려보곤, 돌아서서 구덩이를 기어올랐다. 케인은 바닥에 박힌 무기 앞에 쭈그려 앉아 있었다.

# "방금 말한 거, 너지?" 케인이 질문했다.

# 너도 알지 않나. 내가 부르면 누군가가 듣고 오지. 나는 자격을 갖춘 자에게만 관심이 있다.

# "다들 자격 타령이군. 그래서 누가 자격이 있는데? 무슨 자격?"

# 날 소유할 자격이다. 누군가가 그 자신을 증명하면, 난 그자가 자격을 갖추었는지 알게 되지. 어쩌면 그게 그대일지도.

# "난 네가 누군지도 몰라."

# 나에 대해선 알 필요 없다. 내가 그대에 대해 알아야 하지. 나는 단 한 명의 적격자를 찾을 때까지 부를 것이다. 내가 부름을 멈출 때는 더 이상 부를 필요가 없을 때다.

# "난 제국의 오디널..."

# 난 그대가 무엇이든 상관 하지 않는다. 내가 관심을 가지는 것은 그대가 누구냐는 것이다. 그대의 야망. 그대의 꿈. 그대의 능력. 우주에 대한, 우주의 본분에 대한 그대의 생각.

# "난 오디널이다. 중요한 건 그것뿐이거든." 케인이 날카롭게 말했다. "내겐 해야 될 일이 있어. 임무 말이지."

# 못마땅한 임무, 갈수록 더 불만스러워질 임무 말이군. 그 남자를 향한 충성심은 약해지고 있다. 네가 생각하는 대의명분에 헌신하는 것은 소심한 행동이지. 아무도 네 생각을 알아주지 않고 아무도 네가 원하는 대로 행동하려 하지 않으니 날이 갈수록 좌절감은 커지겠지. 아무도 그럴 만한 힘이 없으니까.

# "내 의무는 데막시아 제국을 위해 이 장소를 확보하는 거다. 지금 골동품 무기와 대화하고 있다는 것도 믿기지 않는군. 양자 변이에 노출된 게 분명해. 이건 내 마음속이고, 내 정신에 뭔가 문제가 생긴 거야."

# 그러니까 지금 내가 환각이라는 말인가?

# "이곳은 이례적으로 뛰어난 과학적 가치를 가지고 있어. 넌 이곳의 중요한 유물이고. 난... 이곳에 있는 외계의 흔적이 만들어 낸 에너지 때문에 환청을 듣고 있는 게 분명해. 게다가—"

# 나쿠리가 사라진 지 꽤 오래 지나지 않았나?

# 케인은 벌떡 일어나 인터페이스의 크로노미터를 확인했다. 나쿠리가 나간 후로 대략 한 시간이 흘렀다. 한 시간이라고? 어떻게 시간이 그렇게 빨리 흐를 수 있지...?

# 시간은 언제든지 사라질 수 있는 환상이지.

# "내게 '자격'이 있다면 말이지?" 케인은 말을 내뱉고는 돌아서서 벽을 오르기 시작했다.

# 뒤에서 들려오는 낄낄거리는 웃음소리를 무시한 채.
# """
# }
# # a=json.dumps(a,ensure_ascii=False)[1:-1]
# # print(a)
# teststate = await cat.ainvoke(teststate)


# In[89]:


# teststate = await detail.ainvoke(teststate)


# # 신규

# In[ ]:


# aa=timeline_select([1,2,3,4],15,"ssssssss")


# In[91]:


# await Relationship(await relationship_select("카엘",28,"eZHXUSS8"))


# In[ ]:




