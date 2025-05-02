# ----- BEGINNING OF reddit_analyzer_cloud_app_v14_syntax_fix_2.py -----
# Final Cloud Version v14 - Fixed ANOTHER SyntaxError in generate_ai_response else block

import streamlit as st
import praw
import json
from datetime import datetime, time, date # Import date explicitly
import pandas as pd
import time as py_time
import pytz
import os

# Attempt to import AI libraries
try:
    import google.generativeai as genai
    google_ai_available = True
except ImportError:
    google_ai_available = False

try:
    import openai
    openai_available = True
except ImportError:
    openai_available = False

# --- 页面配置 ---
st.set_page_config(page_title="Reddit 分析器 + AI 对话 (多模型)", layout="wide", initial_sidebar_state="expanded")

# --- 添加图标 ---
st.title(":mag: Reddit 分析器 + AI 对话 (多模型选择) :robot_face:")
st.caption("抓取 Reddit 讨论，并通过多轮对话与所选 AI 模型进行分析")

# === Helper Function to Format Chat History for OpenAI API style ===
def format_history_for_openai_style(chat_history_internal):
    """Converts internal chat history (role/parts) to OpenAI messages format (role/content)."""
    openai_messages = []
    for message in chat_history_internal:
        role = message.get("role")
        content = message.get("parts", [""])[0]
        mapped_role = "assistant" if role == "model" else role
        if mapped_role in ["system", "user", "assistant"]:
             openai_messages.append({"role": mapped_role, "content": content})
    return openai_messages

# === Unified AI 关键词处理函数 ===
def get_english_keywords_for_query(api_key, user_query, selected_model_details, client_instance):
    """根据选择的模型处理关键词 (支持 Gemini 和 Qwen)"""
    provider = selected_model_details.get("provider")
    model_id = selected_model_details.get("model_id")

    if not api_key or not client_instance:
        st.warning(f"未有效配置 {provider} Key 或客户端，无法自动处理中文关键词。")
        return user_query
    if not user_query.strip():
         st.warning("输入的关键词为空，无法处理。")
         return ""
    if not model_id:
        st.error("未提供有效的模型 ID。")
        return user_query

    reddit_query = user_query # Default fallback
    processed_query = None # Initialize to track if processing happened

    try:
        if provider == "Gemini":
            if not google_ai_available: raise ImportError("Google Generative AI library not loaded.")
            model = client_instance.GenerativeModel(model_id)
            prompt = f"""
            Process the user query "{user_query}" for a Reddit keyword search.
            1. If the query is in Chinese, translate its core meaning to a concise English keyword/phrase and generate 2-3 related English terms.
            2. If the query is already English, use it as the primary term and generate 2-3 related English terms.
            3. **CRITICAL**: Return ONLY a comma-separated list of the final English terms, enclosed in parentheses. Example: (term1), (term2), (term3).
            4. **DO NOT** include any explanations, reasoning, language detection results, or any text other than the comma-separated list in parentheses.
            Example input "老年人 消费", Expected output ONLY: (elderly consumption), (senior spending), (aging population purchase), (geriatric cost)
            """
            response = model.generate_content(prompt)
            if hasattr(response, 'text') and response.text:
                 keywords_str = response.text.strip()
                 if keywords_str.startswith("(") and keywords_str.endswith(")") and "," in keywords_str:
                     keywords_list = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
                     if keywords_list:
                        processed_terms = [f"({kw.strip('()')})" for kw in keywords_list if kw.strip('()')]
                        processed_query = " OR ".join(processed_terms) if processed_terms else None
                        if processed_query: st.info(f"已使用 {model_id} (Gemini) 处理关键词: {processed_query}")
                        else: st.warning(f"{model_id} (Gemini) 解析关键词列表后为空...");
                     else: st.warning(f"{model_id} (Gemini) 未能从响应中解析出关键词列表...");
                 else: st.warning(f"{model_id} (Gemini) 返回格式不符合预期: '{keywords_str[:100]}...'");
            else: st.warning(f"{model_id} (Gemini) 未能处理关键词");

        elif provider == "Qwen":
            if not openai_available: raise ImportError("OpenAI library (for Qwen) not loaded.")
            prompt_messages = [
                {"role": "system", "content": "You are an assistant that processes user queries for a Reddit keyword search. Analyze query, detect language. If Chinese, translate to English and generate 2-3 related English terms. If English, use original and generate 2-3 related terms. Return ONLY a comma-separated list like: (term1), (term2), (term3)"},
                {"role": "user", "content": user_query}
            ]
            response = client_instance.chat.completions.create(model=model_id, messages=prompt_messages, temperature=0.5, max_tokens=150)
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                 keywords_str = response.choices[0].message.content.strip()
                 if "," in keywords_str and len(keywords_str) < 200: # Relaxed Qwen check
                     keywords_list = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
                     if keywords_list:
                        processed_terms = [f"({kw.strip('()')})" for kw in keywords_list if kw.strip('()')]
                        processed_query = " OR ".join(processed_terms) if processed_terms else None
                        if processed_query: st.info(f"已使用 {model_id} (Qwen) 处理关键词: {processed_query}")
                        else: st.warning(f"{model_id} (Qwen) 解析关键词列表后为空...");
                     else: st.warning(f"{model_id} (Qwen) 未能从响应中解析出关键词列表...");
                 else: st.warning(f"{model_id} (Qwen) 返回格式不符合预期 (缺少逗号或过长): '{keywords_str[:100]}...'");
            else: st.warning(f"{model_id} (Qwen) 未能处理关键词");
        else: st.error(f"未知的模型提供商: {provider}")

        return processed_query if processed_query else user_query

    except Exception as e:
        st.error(f"调用 {model_id} ({provider}) 处理关键词时出错: {e}")
        return user_query # Fallback


# === 核心抓取函数 ===
# (函数内容与 V12 相同)
def run_scraper(sub_name, query, limit, t_filter, get_comments, use_dates, start_dt, end_dt, reddit_client):
    """执行 Reddit 数据抓取的函数"""
    all_data = []
    status_area = st.empty()
    if not reddit_client: status_area.error("Reddit PRAW 客户端未初始化。"); return None
    try:
        status_area.info(f"正在 r/{sub_name} 中搜索 '{query}' (基本时间范围: {t_filter})..."); subreddit = reddit_client.subreddit(sub_name)
        if not query or not query.strip('()'): status_area.warning("搜索查询为空..."); return []
        search_results = list(subreddit.search(query=query, sort='new', time_filter=t_filter, limit=limit)); status_area.info(f"初步找到 {len(search_results)} 个帖子...")
        start_timestamp, end_timestamp = None, None
        if use_dates and start_dt and end_dt:
            try:
                if isinstance(start_dt, date) and isinstance(end_dt, date):
                     start_datetime_naive = datetime.combine(start_dt, time.min); end_datetime_naive = datetime.combine(end_dt, time.max)
                     utc_tz = pytz.utc; start_timestamp = utc_tz.localize(start_datetime_naive).timestamp(); end_timestamp = utc_tz.localize(end_datetime_naive).timestamp()
                     if start_timestamp > end_timestamp: st.warning("开始日期晚于结束日期..."); start_timestamp, end_timestamp = None, None
                     else: status_area.info(f"将应用日期范围: {start_dt.strftime('%Y-%m-%d')} 至 {end_dt.strftime('%Y-%m-%d')}")
                else:
                     if use_specific_dates: st.warning("日期输入无效或未完整选择..."); start_timestamp, end_timestamp = None, None
            except Exception as date_e: st.error(f"处理日期输入时出错: {date_e}"); start_timestamp, end_timestamp = None, None
        total_posts_processed_this_run, date_skipped_count = 0, 0; progress_bar = st.progress(0) if search_results else None
        for i, submission in enumerate(search_results):
            if progress_bar: progress_bar.progress((i + 1) / len(search_results))
            if start_timestamp and end_timestamp:
                if not (start_timestamp <= submission.created_utc <= end_timestamp): date_skipped_count += 1; continue
            status_area.info(f"正在处理帖子 {i+1}/{len(search_results)}: {submission.id} - {submission.title[:30]}..."); post_info = {'id': submission.id, 'title': submission.title, 'selftext': submission.selftext, 'url': submission.url, 'score': submission.score, 'created_utc': submission.created_utc, 'comments': []}
            if get_comments:
                try:
                    submission.comment_sort = 'new'; submission.comments.replace_more(limit=0); comment_limit_per_post = 20; processed_comments = 0
                    for comment in submission.comments.list():
                        if processed_comments >= comment_limit_per_post: break
                        if hasattr(comment, 'body'): post_info['comments'].append({'id': comment.id, 'author': str(comment.author), 'body': comment.body, 'score': comment.score, 'created_utc': comment.created_utc, 'parent_id': str(comment.parent_id), 'depth': comment.depth}); processed_comments += 1
                except Exception as e: st.warning(f"处理帖子 {submission.id} 评论时出错: {e}")
            all_data.append(post_info); total_posts_processed_this_run += 1; py_time.sleep(0.05)
        status_area.success(f"抓取完成！本次运行处理了 {total_posts_processed_this_run} 个帖子。跳过了 {date_skipped_count} 个不符合日期范围的帖子。"); return all_data
    except praw.exceptions.PRAWException as pe: status_area.error(f"Reddit API (PRAW) 错误: {pe}"); return None
    except Exception as e: status_area.error(f"抓取过程中发生意外错误: {e}"); return None


# === Unified AI 分析函数 (FIXED Syntax Error AGAIN) ===
def generate_ai_response(api_key, chat_history, selected_model_details, client_instance):
    """根据选择的模型调用 API 生成聊天回复 (支持 Gemini 和 Qwen)"""
    provider = selected_model_details.get("provider")
    model_id = selected_model_details.get("model_id")

    if not api_key or not client_instance:
        return f"错误：未有效配置 {provider} API Key 或客户端。"
    is_initial_qwen_call = (
            provider == "Qwen" and
            len(chat_history) == 2 and
            chat_history[0].get("role") == "system" and "content" in chat_history[0] and
            chat_history[1].get("role") == "user" and "content" in chat_history[1]
        )
    # Check normal history (role/parts) or initial Qwen call (role/content)
    if not is_initial_qwen_call and (not chat_history or not any(msg.get("parts") and msg["parts"][0] for msg in chat_history if msg.get("role") != "system")):
         return "错误：聊天历史为空或格式不正确。"

    if not model_id:
        return "错误：未提供有效的模型 ID。"

    try:
        if provider == "Gemini":
            if not google_ai_available: raise ImportError("Google Generative AI library not loaded.")
            model = client_instance.GenerativeModel(model_id)
            response = model.generate_content(chat_history)

            if hasattr(response, 'text') and response.text:
                 return response.text.strip()
            else:
                 # --- FIXED SYNTAX BLOCK ---
                 # Initialize feedback_text
                 feedback_text = "No specific feedback available."
                 # Safely try to get prompt_feedback
                 try:
                     if hasattr(response, 'prompt_feedback'):
                         feedback_text = str(response.prompt_feedback)
                 except Exception:
                     # Ignore potential errors during feedback retrieval/conversion
                     pass
                 # Now report the error on separate lines
                 st.error(f"{model_id} (Gemini) API 返回了响应，但无法提取文本。反馈: {feedback_text}")
                 return f"错误：{model_id} (Gemini) API 返回无效响应。反馈: {feedback_text}"
                 # --- END FIXED SYNTAX BLOCK ---

        elif provider == "Qwen":
            if not openai_available: raise ImportError("OpenAI library (for Qwen) not loaded.")
            # Format history correctly based on whether it's the initial call or not
            if is_initial_qwen_call:
                 formatted_history = chat_history # Use the pre-formatted system/user messages
            else:
                 formatted_history = format_history_for_openai_style(chat_history) # Convert subsequent history

            if not formatted_history: return "错误：处理后的聊天历史为空。"

            response = client_instance.chat.completions.create(model=model_id, messages=formatted_history, temperature=0.7)
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                st.error(f"{model_id} (Qwen) API 未返回有效回复。")
                print(f"Qwen API Response Error Object: {response}") # Print full response for debugging
                return f"错误：{model_id} (Qwen) API 未返回有效回复。"
        else:
             return f"错误：未知的模型提供商: {provider}"

    except Exception as e:
        st.error(f"调用 {model_id} ({provider}) API 时出错: {e}")
        # Consider logging the full traceback here for deeper debugging if needed
        # import traceback
        # traceback.print_exc()
        return f"调用 {model_id} ({provider}) API 时出错: {type(e).__name__}"


# --- 侧边栏定义 ---
# (保持 V12 的逻辑不变)
with st.sidebar:
    st.header("参数配置"); st.markdown("---")
    available_ai_models = {"Gemini 2.0 Flash (推荐)": {"provider": "Gemini", "model_id": "gemini-2.0-flash"}, "Qwen Turbo (最新)": {"provider": "Qwen", "model_id": "qwen-turbo-latest"}, "Qwen Plus (最新)": {"provider": "Qwen", "model_id": "qwen-plus-latest"}, "Gemini 2.5 Pro Exp (限制严格!)": {"provider": "Gemini", "model_id": "gemini-2.5-pro-exp-03-25"}}
    model_display_names = list(available_ai_models.keys()); selected_model_display_name = st.selectbox("选择 AI 模型:", options=model_display_names, index=0, key="model_selector", help="选择 AI 模型。注意 Gemini Pro Exp 免费层级限制严格！")
    selected_model_details = available_ai_models[selected_model_display_name]; selected_provider = selected_model_details["provider"]; selected_model_id = selected_model_details["model_id"]
    st.markdown("---"); st.markdown(f"**当前选择模型:**"); st.markdown(f"- 名称: `{selected_model_display_name}`"); st.markdown(f"- ID: `{selected_model_id}`"); st.markdown(f"- 提供商: `{selected_provider}`"); st.markdown("---")
    REDDIT_CLIENT_ID=os.environ.get("REDDIT_CLIENT_ID"); REDDIT_CLIENT_SECRET=os.environ.get("REDDIT_CLIENT_SECRET"); REDDIT_USER_AGENT=os.environ.get("REDDIT_USER_AGENT"); reddit_client_instance=None
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET or not REDDIT_USER_AGENT: st.error("错误：Reddit API 凭证未完整配置。")
    else:
        try: reddit_client_instance=praw.Reddit(client_id=REDDIT_CLIENT_ID,client_secret=REDDIT_CLIENT_SECRET,user_agent=REDDIT_USER_AGENT,read_only=True); st.success("Reddit API 凭证已加载。")
        except Exception as praw_e: st.error(f"初始化 Reddit PRAW 客户端时出错: {praw_e}"); reddit_client_instance=None
    GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY"); QWEN_API_KEY=os.environ.get("DASHSCOPE_API_KEY"); gemini_configured=False; qwen_configured=False; ai_configured_successfully=False; active_client_instance=None; active_api_key=None
    if selected_provider=="Gemini":
        if google_ai_available and GEMINI_API_KEY:
            try: genai.configure(api_key=GEMINI_API_KEY); active_client_instance=genai; active_api_key=GEMINI_API_KEY; gemini_configured=True; ai_configured_successfully=True; st.success(f"Gemini (模型: {selected_model_id}) 配置成功！")
            except Exception as gemini_config_e: st.error(f"配置 Gemini API 时出错: {gemini_config_e}")
        elif google_ai_available: st.warning("提示：选择了 Gemini，但未配置 Gemini API Key...")
        else: st.error("Gemini 库未加载。")
    elif selected_provider=="Qwen":
        if openai_available and QWEN_API_KEY:
            try: qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"; active_client_instance=openai.OpenAI(api_key=QWEN_API_KEY,base_url=qwen_base_url); active_api_key=QWEN_API_KEY; qwen_configured=True; ai_configured_successfully=True; st.success(f"Qwen (模型: {selected_model_id}) 配置成功！")
            except Exception as qwen_config_e: st.error(f"配置 Qwen (兼容模式) API 时出错: {qwen_config_e}"); active_client_instance=None
        elif openai_available: st.warning("提示：选择了 Qwen，但未配置其 API Key (名称应为 DASHSCOPE_API_KEY)...")
        else: st.error("OpenAI 库未加载，无法使用 Qwen。")
    st.markdown("---")
    subreddit_name=st.text_input("Subreddit 名称", "askSingapore", key="subreddit_input"); search_query_input=st.text_input("搜索关键词 (支持中文)", "amazon", key="query_input"); post_limit=st.number_input("最大帖子数量", min_value=1, max_value=1000, value=20, key="limit_input"); time_filter_options={"过去一小时":"hour","过去一天":"day","过去一周":"week","过去一月":"month","过去一年":"year","所有时间":"all"}; selected_time_label=st.selectbox("基本时间范围:", options=list(time_filter_options.keys()), index=5, key="time_filter_select"); time_filter=time_filter_options[selected_time_label]
    start_scrape_button=st.button("开始抓取 Reddit 数据", type="primary", icon="🔍", disabled=(not reddit_client_instance), key="scrape_button"); st.markdown("---")
    use_specific_dates=st.checkbox("使用精确日期范围过滤", key="date_checkbox"); start_date_input=st.date_input("开始日期", value=None, disabled=not use_specific_dates, key="start_date"); end_date_input=st.date_input("结束日期", value=None, disabled=not use_specific_dates, key="end_date"); st.caption("*日期筛选在获取帖子后进行。*"); st.markdown("---"); fetch_comments=st.checkbox("抓取评论 (限制每个帖子 20 条)", value=True, key="comments_checkbox"); st.markdown("---")


# --- Initialize Session State ---
# (保持 V12 的逻辑不变)
if 'scraped_data' not in st.session_state: st.session_state.scraped_data = None
if 'chat_context' not in st.session_state: st.session_state.chat_context = {"history": [], "initial_input": None, "model_details_used": None}
if 'final_query_used' not in st.session_state: st.session_state.final_query_used = None

# --- Main Logic: Handle Scrape Button Click ---
# (保持 V12 的逻辑不变)
if start_scrape_button:
    st.session_state.scraped_data=None; st.session_state.chat_context={"history": [], "initial_input": None, "model_details_used": None}; st.session_state.final_query_used=None
    st.info(f"收到抓取任务，正在处理关键词 '{search_query_input}'...")
    final_reddit_query = search_query_input
    if ai_configured_successfully and selected_model_details and active_api_key and active_client_instance:
        with st.spinner(f"正在使用 {selected_model_display_name} 处理/翻译关键词..."): final_reddit_query=get_english_keywords_for_query(active_api_key, search_query_input, selected_model_details, active_client_instance)
    elif selected_provider: st.warning(f"{selected_provider} 未成功配置，将使用原始关键词。")
    else: st.warning("未选择或未加载 AI 模型，将使用原始关键词。")
    st.session_state.final_query_used=final_reddit_query
    if st.session_state.final_query_used:
        st.info(f"最终用于 Reddit 搜索的查询语句: '{st.session_state.final_query_used}'");
        with st.spinner(f"正在 r/{subreddit_name} 中搜索并抓取数据..."): scraped_data_result=run_scraper(subreddit_name, st.session_state.final_query_used, post_limit, time_filter, fetch_comments, use_specific_dates, start_date_input, end_date_input, reddit_client_instance)
        st.session_state.scraped_data=scraped_data_result; st.session_state.chat_context["model_details_used"]=selected_model_details if ai_configured_successfully else None; st.rerun()
    else: st.error("关键词处理后为空或处理失败，无法执行搜索。"); st.stop()


# --- Display Scraped Results and AI Chat Interface ---
# (保持 V12 的逻辑不变)
if st.session_state.scraped_data is not None:
    model_details_for_this_session=st.session_state.chat_context.get("model_details_used"); model_display_name_for_this_session="AI"
    if model_details_for_this_session: model_display_name_for_this_session=next((name for name,details in available_ai_models.items() if details==model_details_for_this_session), model_details_for_this_session.get("model_id", "AI"))
    tab_preview, tab_ai_chat = st.tabs(["📊 数据预览与下载", f"💬 与 {model_display_name_for_this_session} 对话分析"])
    with tab_preview: # Preview Tab Content
        st.subheader("抓取结果预览"); data_len=len(st.session_state.scraped_data) if isinstance(st.session_state.scraped_data, list) else 0; st.write(f"共找到 {data_len} 个符合条件的帖子记录。")
        if st.session_state.final_query_used: st.write(f"(使用的最终搜索查询: `{st.session_state.final_query_used}`) ")
        if not st.session_state.scraped_data or not isinstance(st.session_state.scraped_data, list) or data_len==0: st.warning("没有有效的帖子数据可供预览。")
        else:
            try: # Preview Table
                posts_df_data=[];
                for post in st.session_state.scraped_data:
                    if isinstance(post,dict): post_preview={k: v for k,v in post.items() if k!='comments'}; post_preview['comment_count']=len(post.get('comments',[])); posts_df_data.append(post_preview)
                if posts_df_data:
                    posts_df=pd.DataFrame(posts_df_data);
                    if 'created_utc' in posts_df.columns:
                        try: posts_df['created_datetime_sgt']=pd.to_datetime(posts_df['created_utc'],unit='s',utc=True).dt.tz_convert('Asia/Singapore').dt.strftime('%Y-%m-%d %H:%M:%S SGT')
                        except Exception as tz_e: st.warning(f"无法转换时间戳: {tz_e}."); posts_df['created_datetime_sgt']=posts_df['created_utc']
                    preview_cols=['title','score','comment_count','created_datetime_sgt','url','id','selftext']; display_cols=[col for col in preview_cols if col in posts_df.columns]; st.dataframe(posts_df[display_cols], height=300)
                else: st.info("沒有帖子数据可供预览。")
            except Exception as e: st.error(f"无法将结果格式化为表格显示: {e}")
            st.subheader("下载本次抓取的数据 (JSON)") # Download Button
            try: json_string=json.dumps(st.session_state.scraped_data, indent=4, ensure_ascii=False, default=str); timestamp_str=datetime.now().strftime("%Y%m%d_%H%M%S"); safe_query="".join(c if c.isalnum() else"_" for c in(st.session_state.final_query_used or search_query_input or"query"))[:50]; download_filename=f"reddit_{subreddit_name}_search_{safe_query}_{timestamp_str}.json"; st.download_button(label="下载 JSON 数据", data=json_string, file_name=download_filename, mime="application/json", key="download_button")
            except Exception as e: st.error(f"生成 JSON 下载文件时出错: {e}")

    with tab_ai_chat: # AI Chat Tab
        st.subheader(f"与 {model_display_name_for_this_session or 'AI'} 对话分析")
        if not model_details_for_this_session: st.error("没有成功配置 AI Key 或选择模型用于本次分析，无法进行 AI 对话。")
        else:
             active_chat_client=None; active_chat_key=None; chat_provider=model_details_for_this_session["provider"]; chat_model_id=model_details_for_this_session["model_id"]
             if chat_provider=="Gemini":
                 if google_ai_available and GEMINI_API_KEY:
                     try: genai.configure(api_key=GEMINI_API_KEY); active_chat_client=genai; active_chat_key=GEMINI_API_KEY
                     except Exception as e: st.error(f"重新配置 Gemini 出错: {e}")
             elif chat_provider=="Qwen":
                  if openai_available and QWEN_API_KEY:
                     try: qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"; active_chat_client=openai.OpenAI(api_key=QWEN_API_KEY, base_url=qwen_base_url); active_chat_key=QWEN_API_KEY
                     except Exception as e: st.error(f"重新初始化 Qwen 客户端出错: {e}")
             if active_chat_client and active_chat_key: # Proceed only if client is ready
                if st.session_state.chat_context.get("initial_input"): # Show Initial Input Expander
                    with st.expander(f"查看首次分析发送给 {model_display_name_for_this_session} 的完整输入", expanded=False): st.text(st.session_state.chat_context["initial_input"])
                    st.markdown("---")
                if not st.session_state.chat_context["history"]: # Initial Analysis Section
                    query_for_prompt=st.session_state.final_query_used or search_query_input; default_initial_prompt = f"你是一个数据分析助手。请根据以下抓取的关于 '{query_for_prompt}' 的 Reddit 帖子和评论内容，总结主要的讨论观点、用户情绪、可能存在的消费趋势以及亚马逊新加坡站的选品思路，并用表格列出每个选品与Reddit内容的关联性，并按照用户需求的迫切性对选品进行排序。(使用模型: {model_display_name_for_this_session})"; st.write("**初步分析指令 (可编辑):**"); analysis_prompt_input = st.text_area(label="初步分析指令", value=default_initial_prompt, height=150, key="initial_prompt_input_area", label_visibility="collapsed")
                    if st.button(f"使用 {model_display_name_for_this_session} 进行初步 AI 分析", icon=":material/auto_awesome:", key="initial_analysis_button"):
                        with st.spinner("正在准备初步分析请求..."): # Prepare Initial Request
                            analysis_prompt=analysis_prompt_input; combined_text=""; max_chars=300000; current_chars=0; posts_included_count=0; limit_warning_shown=False
                            if isinstance(st.session_state.scraped_data, list):
                                for post in st.session_state.scraped_data: # Build combined_text
                                    if isinstance(post,dict): post_content=f"帖子 ID: {post.get('id','')}\n标题: {post.get('title','')}\n正文: {post.get('selftext','')}\n"; potential_len=current_chars+len(post_content);
                                    if potential_len>=max_chars:
                                        if not limit_warning_shown: st.warning(f"文本长度接近或达到上限 {max_chars} 字符..."); limit_warning_shown=True; break
                                    combined_text+=post_content; current_chars+=len(post_content); posts_included_count+=1; comment_count=0; comments_text=""
                                    for comment in post.get('comments',[]):
                                         if isinstance(comment,dict): comment_content=f"  评论 (ID: {comment.get('id','')}, Score: {comment.get('score','')}): {comment.get('body','')}\n"; potential_len=current_chars+len(comment_content);
                                         if potential_len>=max_chars:
                                                if not limit_warning_shown: st.warning(f"文本长度接近或达到上限 {max_chars} 字符..."); limit_warning_shown=True; break
                                         comments_text+=comment_content; current_chars+=len(comment_content); comment_count+=1
                                         if comment_count>=10: break
                                    combined_text+=comments_text+"---\n"
                            full_initial_message_for_api=f"{analysis_prompt}\n\n以下是相关数据：\n\n{combined_text}"; history_for_api_call=[]
                            if chat_provider=="Gemini": history_for_api_call=[{"role":"user","parts":[full_initial_message_for_api]}]
                            elif chat_provider=="Qwen": history_for_api_call=[{"role":"system","content":analysis_prompt},{"role":"user","content":f"以下是相关数据：\n\n{combined_text}"}]
                            st.write(f"最终准备发送给 {model_display_name_for_this_session} 的文本包含 {posts_included_count} 个帖子。总字符数约: {current_chars}"); st.session_state.chat_context["initial_input"]=full_initial_message_for_api
                        with st.spinner(f"正在调用 {model_display_name_for_this_session} 进行初步分析..."): # Call Initial Analysis
                            initial_response_text=generate_ai_response(active_chat_key, history_for_api_call, model_details_for_this_session, active_chat_client)
                            st.session_state.chat_context["history"]=[]; st.session_state.chat_context["history"].append({"role":"user","parts":[analysis_prompt]}); st.session_state.chat_context["history"].append({"role":"model","parts":[initial_response_text]}); st.rerun()
                if "history" in st.session_state.chat_context and isinstance(st.session_state.chat_context["history"], list) and st.session_state.chat_context["history"]: # Display Chat History
                    st.markdown("---"); st.write(f"**与 {model_display_name_for_this_session} 的对话分析记录:**")
                    for message in st.session_state.chat_context["history"]:
                         if isinstance(message,dict) and"role"in message and"parts"in message and isinstance(message["parts"],list) and message["parts"]: role_disp=message["role"];
                         with st.chat_message(role_disp): st.markdown(message["parts"][0])
                chat_input_disabled=not(active_chat_client and active_chat_key); # Chat Input Box
                if prompt := st.chat_input(f"向 {model_display_name_for_this_session} 继续提问...", disabled=chat_input_disabled, key="chat_input"):
                     st.session_state.chat_context["history"].append({"role":"user","parts":[prompt]});
                     with st.chat_message("user"): st.markdown(prompt)
                     with st.chat_message("model"):
                         with st.spinner(f"{model_display_name_for_this_session} 正在思考..."):
                             response=generate_ai_response(active_chat_key, st.session_state.chat_context["history"], model_details_for_this_session, active_chat_client)
                             if isinstance(response,str) and not response.startswith("错误："): st.markdown(response); st.session_state.chat_context["history"].append({"role":"model","parts":[response]})
                             else: st.error(f"未能从 {model_display_name_for_this_session} 获取有效回复。错误: {response}");
                             if st.session_state.chat_context["history"] and st.session_state.chat_context["history"][-1]["role"]=="user": st.session_state.chat_context["history"].pop()
                     st.rerun()
             else: st.error(f"无法为模型 {model_display_name_for_this_session} 初始化客户端或获取 API Key，无法进行聊天。")

# --- Fallback Message ---
elif start_scrape_button and st.session_state.scraped_data is None: st.error("未能成功抓取数据，请检查侧边栏配置、Reddit API 状态或网络连接。")
elif not start_scrape_button and st.session_state.scraped_data is None: st.info("请在左侧配置参数并点击“开始抓取 Reddit 数据”按钮。")

# --- Footer ---
st.markdown("---")
st.caption("Reddit 分析工具 v14 - 使用 Streamlit, PRAW, Google Generative AI, Qwen3 构建")
# ----- END OF reddit_analyzer_cloud_app_v14_syntax_fix_2.py -----
