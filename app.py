import streamlit as st
import streamlit_antd_components as sac
from dotenv import load_dotenv
import os
import pandas as pd
from io import StringIO 
import altair as alt # For charting

# --- AI & LANGGRAPH IMPORTS ---
from typing import TypedDict, List, Optional, Dict, Any, Annotated
import operator
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_tavily import TavilySearch

# --- LOCAL IMPORT ---
from data_manager import DataManager

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="AI AML Investigation Hub", page_icon="🤖")
load_dotenv()

# --- DATA MANAGER (cached) ---
@st.cache_resource
def get_data_manager():
    print("--- Initializing DataManager (This should only run once) ---")
    return DataManager(
        accounts_file='data/accounts.csv',
        transactions_file='data/transactions.csv',
        sanctions_file='data/sdn.csv'
    )

data_manager = get_data_manager()

# --- SESSION STATE KEYS ---
if 'current_case_id' not in st.session_state:
    st.session_state['current_case_id'] = None
if 'current_case_details' not in st.session_state:
    st.session_state['current_case_details'] = None
if 'investigation_brief' not in st.session_state:
    st.session_state['investigation_brief'] = None
if 'transaction_df' not in st.session_state:
    st.session_state['transaction_df'] = None
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = None
if 'follow_up_chat' not in st.session_state:
    st.session_state['follow_up_chat'] = []
if 'show_return_button' not in st.session_state:
    st.session_state['show_return_button'] = False


# --- LANGGRAPH STATE DEFINITION ---
class AMLState(TypedDict):
    case_id: str
    customer_details: Optional[dict]
    findings: Dict[str, Any]
    investigation_brief: str
    human_decision: str
    # FIX: This tells LangGraph to APPEND messages
    messages: Annotated[List, operator.add] 

# --- TOOLS (WITH DOCSTRINGS) ---
web_search_tool = TavilySearch(max_results=3)

@tool
def get_account_details(customer_id: str) -> dict:
    """Fetches the Know Your Customer (KYC) details for a given account ID."""
    print(f"--- TOOL: Fetching account details for {customer_id} ---")
    return data_manager.get_account_details(customer_id)

@tool
def get_transaction_cluster(customer_id: str) -> str:
    """Fetches all transactions (inbound and outbound) for a given account ID."""
    print(f"--- TOOL: Fetching transaction cluster for {customer_id} ---")
    df = data_manager.get_transaction_cluster(customer_id)
    return df.to_csv(index=False)

@tool
def check_sanctions_list(customer_name: str) -> dict:
    """Checks a customer's name against the OFAC sanctions list (sdn.csv)."""
    print(f"--- TOOL: Checking sanctions list for {customer_name} ---")
    return data_manager.check_sanctions_list(customer_name)

agent_tools = [
    get_account_details,
    get_transaction_cluster,
    check_sanctions_list,
    web_search_tool
]
tool_map = {tool.name: tool for tool in agent_tools}

# --- LLM & Agent ---
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0, streaming=False)
agent_with_tools = llm.bind_tools(agent_tools)

# --- Simplified NODES (State Fix) ---

def investigator_node(state: AMLState) -> dict:
    """Invokes the agent with the current, correctly appended message history."""
    print("--- Calling Investigator Node ---")
    response = agent_with_tools.invoke(state['messages'])
    return {"messages": [response]} 

def router_node(state: AMLState) -> str:
    """Router: decide whether to call tools or summarize."""
    print("--- Calling Router Node ---")
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "call_tools"
    else:
        return "summarize"

# --- Summarizer Node (Full Prompt) ---
def summarizer_node(state: AMLState) -> dict:
    print("--- Calling Summarizer Node ---")
    case_id = state.get('case_id', 'the account')

    summary_prompt = SystemMessage(
        content=f"""
        You are a meticulous senior AML (Anti-Money Laundering) analyst.
        Your task is to write a **full, detailed, and well-organized** "Investigation Brief" 
        for a human reviewer, based *only* on the provided conversation history and tool outputs.

        The account under investigation is: {case_id}.
        
        **CRITICAL MASTER RULE**: Your response **MUST** be in markdown and **MUST** follow the 5-section 
        template below. **DO NOT** skip any section or deviate from this format. 
        The human analyst needs the complete report for their records.

        ---
        
        ### 1. Account Details
        (List all key KYC info found: Customer ID, Name, Country, Date of Birth, and any 
         account flags like 'IS_FRAUD'. If info is missing, state that.)

        ### 2. Financial Summary (Analysis)
        (Analyze the transaction data.
         - Calculate and state the **Total Inbound Value ($)**. 
         - Calculate and state the **Total Outbound Value ($)**.
         - Calculate and state the **Net Flow ($)**.
         
         **Crucial Formatting Rule**: State *only* the final calculated numbers. 
         **Do not** show your math (e.g., do not write out '253.16 + 38.87...').
         
         - Provide a 2-3 bullet point summary of the overall financial activity.)

        ### 3. Key Counterparties
        (Analyze the transaction data to find the key partners.
         - Use bullet points to list the **Top 3 Senders** to this account (by total amount).
         - Use bullet points to list the **Top 3 Receivers** from this account (by total amount).
         If there are no senders or receivers for a category, state that clearly.)

        ### 4. Sanctions & Adverse Media
        (Use bullet points to state:
         - The result of the sanctions check.
         - A summary of any adverse media (negative news) found.
         If nothing was found, state that clearly.)

        ### 5. Conclusion & Final Risk Assessment
        (Conclude with a final risk assessment: Low, Medium, or High. 
         You must write a **concise (2-3 sentences) justification** for your choice, 
         referencing the key findings from all sections above.)
        """
    )

    messages_with_prompt = state.get('messages', []) + [summary_prompt]
    summary_response = llm.invoke(messages_with_prompt)
    return {"investigation_brief": summary_response.content}

# --- Chart function ---
def create_activity_chart(df: pd.DataFrame):
    try:
        df_copy = df.copy()
        df_copy['TIMESTAMP'] = pd.to_numeric(df_copy.get('TIMESTAMP', pd.Series(dtype=float)), errors='coerce').fillna(0)
        activity = df_copy.sort_values('TIMESTAMP')
        chart_activity = alt.Chart(activity).mark_line().encode(
            x=alt.X('TIMESTAMP', title='Time (by transaction step)'),
            y=alt.Y('TX_AMOUNT', title='Transaction Amount ($)'),
            tooltip=['TIMESTAMP', 'TX_AMOUNT', 'SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_TYPE']
        ).properties(title='Transaction Activity Over Time').interactive()
        st.subheader("Transaction Activity")
        st.altair_chart(chart_activity, width='stretch')
    except Exception as e:
        st.error(f"Error creating activity chart: {e}")

# --- Build Graph ---
@st.cache_resource
def get_graph():
    print("--- Compiling Graph (This should only run once) ---")
    
    tool_node = ToolNode(agent_tools) 
    
    workflow = StateGraph(AMLState)
    workflow.add_node("investigator", investigator_node)
    workflow.add_node("tool_executor", tool_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.set_entry_point("investigator")
    workflow.add_conditional_edges(
        "investigator",
        router_node,
        {"call_tools": "tool_executor", "summarize": "summarizer"}
    )
    workflow.add_edge("tool_executor", "investigator")
    workflow.add_edge("summarizer", END)
    return workflow.compile()

aml_graph = get_graph()

# --- UI Layout ---
st.title("🤖 AI AML Investigation Hub")
selected_tab = sac.tabs(
    items=[
        sac.TabsItem(label='New Alerts', icon='bell'),
        sac.TabsItem(label='Investigation Hub', icon='search'),
        sac.TabsItem(label='Closed Cases', icon='archive'),
    ],
    align='left',
    format_func='title',
)

# --- TAB 1: New Alerts ---
if selected_tab == 'New Alerts':
    alert_placeholder = st.empty()
    alerts = data_manager.get_suspicious_alerts()
    total_alerts = len(alerts)
    total_accounts = len(getattr(data_manager, "accounts_df", pd.DataFrame()))
    alert_rate = (total_alerts / total_accounts) * 100 if total_accounts > 0 else 0

    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
    with col1:
        st.header("Pending Alerts")
        st.write("This list shows suspicious accounts flagged by the system.")
    with col2:
        st.metric("Total Pending Alerts", f"{total_alerts}")
    with col3:
        st.metric("Total Accounts", f"{total_accounts:,}")
    with col4:
        st.metric("Alert Rate", f"{alert_rate:.2f}%")
    st.divider()

    cols = st.columns(3)
    col_index = 0
    for alert_id in alerts:
        if cols[col_index].button(f"Investigate Account: {alert_id}", use_container_width=True, type="primary"):
            st.session_state['current_case_id'] = alert_id
            st.session_state['current_case_details'] = data_manager.get_account_details(alert_id)
            st.session_state['investigation_brief'] = None
            st.session_state['transaction_df'] = None
            st.session_state['message_history'] = None
            st.session_state['follow_up_chat'] = []
            st.session_state['show_return_button'] = False # Reset flag
            alert_placeholder.success(f"Case {alert_id} selected. Please go to the 'Investigation Hub' tab to proceed.")
        col_index = (col_index + 1) % 3

# --- TAB 2: Investigation Hub ---
elif selected_tab == 'Investigation Hub':
    st.header("Active Case Investigation")
    if not st.session_state.get('current_case_id'):
        st.info("Please select an alert from the 'New Alerts' tab to begin.")
    else:
        case_id = st.session_state['current_case_id']
        case_details = st.session_state['current_case_details']
        st.subheader(f"Investigating Account: {case_id}")

        if not st.session_state.get('investigation_brief'):
            # --- ⬇️ THIS IS THE NEW ALIGNED LAYOUT (v36) ⬇️ ---
            
            col_left, col_right = st.columns([1, 1]) # 50/50 split

            with col_left:
                st.subheader("Account Details (KYC)")
                st.json(case_details, expanded=True)

            with col_right:
                # Add a matching subheader for alignment
                st.subheader("Investigation Control") 
                
                st.info("💡 Ready to begin? Click the button below to start the AI's investigation and generate the final report.") 
                
                if st.button("🚀 Start AI Investigation", type="primary"):
                    customer_name = case_details.get('NAME', case_id) if isinstance(case_details, dict) else case_id
                    initial_prompt = f"Please investigate account ID: {case_id}. Start by getting the account details and transaction cluster. Then check the customer's name, '{customer_name}', against the sanctions list and perform a web search for adverse media."
                    initial_messages = [
                        SystemMessage(content="You are a world-class AML investigator. Gather relevant facts about a customer and their transactions."),
                        HumanMessage(content=initial_prompt)
                    ]
                    initial_state = {
                        "case_id": case_id,
                        "customer_details": case_details or {},
                        "findings": {},
                        "investigation_brief": "",
                        "human_decision": "",
                        "messages": initial_messages
                    }
                    with st.spinner("🤖 AI Investigator at work... (This may take a moment)"):
                        final_state = aml_graph.invoke(initial_state)
                        st.session_state['investigation_brief'] = final_state.get('investigation_brief')
                        st.session_state['message_history'] = final_state.get('messages', [])
                        
                        transaction_csv_str = ""
                        for msg in final_state.get('messages', []):
                            if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and "SENDER_ACCOUNT_ID" in msg.content:
                                transaction_csv_str = msg.content
                                break
                        if transaction_csv_str:
                            st.session_state['transaction_df'] = pd.read_csv(StringIO(transaction_csv_str))
                        else:
                            st.session_state['transaction_df'] = pd.DataFrame()
                    st.rerun()
            # --- ⬆️ END OF NEW LAYOUT ⬆️ ---

        else:
            # Post-investigation: show brief, transactions, follow-up chat, decision
            sac.steps(
                items=[
                    sac.StepsItem(title='Alert Triggered', icon='bell'),
                    sac.StepsItem(title='AI Analysis', icon='robot'),
                    sac.StepsItem(title='Human Review', icon='user', description='Please review the findings.'),
                    sac.StepsItem(title='Case Closed', icon='check-square', disabled=True),
                ], index=2, format_func='title', placement='horizontal'
            )
            st.divider()
            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.subheader("AI-Generated Investigation Brief")
                # CSS Height Control for Alignment
                st.markdown(
                    f"<div style='height: 750px; overflow-y: scroll; padding-right: 15px; border: 1px solid #333333; border-radius: 5px; padding: 10px;'>{st.session_state['investigation_brief']}</div>", 
                    unsafe_allow_html=True
                )
            with col2:
                st.subheader("Account Details (KYC)")
                st.json(st.session_state['current_case_details'], expanded=True)
                st.subheader("Transaction Cluster")
                df = st.session_state.get('transaction_df')
                tab1, tab2 = st.tabs(["Data Table", "Visual Graph"])
                with tab1:
                    if df is not None and not df.empty:
                        st.dataframe(df)
                    else:
                        st.info("No transaction data was found for this account.")
                with tab2:
                    if df is not None and not df.empty:
                        create_activity_chart(df)
                    else:
                        st.info("No transaction data to visualize.")

            st.divider()
            # Follow-up form
            st.subheader("Analyst Follow-up")
            with st.form("follow_up_form"):
                follow_up_question = st.text_area("Request More Info (Optional)", placeholder="e.g., 'Check the sender of the largest inbound transaction for adverse media.'")
                ask_submitted = st.form_submit_button("Ask")

            if ask_submitted and follow_up_question:
                # append human message
                st.session_state['follow_up_chat'].append(HumanMessage(content=follow_up_question))

                main_history = st.session_state.get('message_history', []) or []
                context_messages = [msg for msg in main_history if not isinstance(msg, SystemMessage)]
                
                # --- V32 FIX: Filter tools for "lite" agent ---
                lite_agent_tools = [t for t in agent_tools if t.name != 'get_transaction_cluster']
                lite_agent_runnable = llm.bind_tools(lite_agent_tools)

                system_prompt = SystemMessage(content="""
                    You are a helpful AML analyst assistant. Use the provided transaction CSV and history to answer the user's follow-up concisely. 
                    Do not rewrite the full report.
                    **When asked to analyze existing data (like max transaction), look at the history before calling tools.**
                """)

                transaction_df = st.session_state.get('transaction_df')
                if transaction_df is not None and not transaction_df.empty:
                    csv_text = transaction_df.to_csv(index=False)
                    csv_message = AIMessage(content=f"__TRANSACTION_CSV_START__\n{csv_text}\n__TRANSACTION_CSV_END__")
                else:
                    csv_message = AIMessage(content="No transaction data available for this account.")

                messages = [system_prompt, csv_message] + context_messages + st.session_state['follow_up_chat']

                def normalize_ai_text(msg_content):
                    if isinstance(msg_content, str):
                        return msg_content
                    if isinstance(msg_content, list) and len(msg_content) > 0:
                        first = msg_content[0]
                        if isinstance(first, dict) and 'text' in first:
                            return first['text']
                        return " ".join([str(x) for x in msg_content])
                    if isinstance(msg_content, dict) and 'text' in msg_content:
                        return msg_content['text']
                    return str(msg_content)

                with st.spinner("🤖 AI is thinking..."):
                    while True:
                        # --- V32 FIX: Use the 'lite_agent_runnable' ---
                        ai_response = lite_agent_runnable.invoke(messages)
                        tool_calls = getattr(ai_response, "tool_calls", None)

                        if not tool_calls:
                            clean_text = normalize_ai_text(getattr(ai_response, "content", None))
                            final_ai_msg = AIMessage(content=clean_text)
                            st.session_state['follow_up_chat'].append(final_ai_msg)
                            break

                        st.session_state['follow_up_chat'].append(ai_response)
                        messages.append(ai_response)

                        tool_outputs = []
                        for tool_call in ai_response.tool_calls:
                            try:
                                name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
                                args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", None)
                                call_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
                            except Exception:
                                name = None; args = None; call_id = None

                            tool_to_call = tool_map.get(name)
                            if tool_to_call:
                                try:
                                    if isinstance(args, dict):
                                        output = tool_to_call.invoke(args)
                                    elif isinstance(args, (list, tuple)):
                                        output = tool_to_call.invoke(*args)
                                    elif args is None:
                                        output = tool_to_call.invoke()
                                    else:
                                        output = tool_to_call.invoke(args)
                                    tool_outputs.append(ToolMessage(content=str(output), tool_call_id=call_id))
                                except Exception as e:
                                    tool_outputs.append(ToolMessage(content=f"Error running tool '{name}': {e}", tool_call_id=call_id))
                            else:
                                tool_outputs.append(ToolMessage(content=f"Error: Unknown tool '{name}'.", tool_call_id=call_id))

                        messages.extend(tool_outputs)
                        st.session_state['follow_up_chat'].extend(tool_outputs)

                st.rerun()

            # Display follow-up chat
            def extract_ai_content_fast(content) -> str:
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    if content and isinstance(content[0], dict) and 'text' in content[0]:
                        return content[0]['text']
                if isinstance(content, dict) and 'text' in content:
                    return content['text']
                return str(content)

            if st.session_state.get('follow_up_chat'):
                st.write("---")
                st.subheader("Follow-up Conversation")
                chat_container = st.container(height=300) # Fixed height chat
                with chat_container:
                    for msg in st.session_state['follow_up_chat']:
                        if isinstance(msg, HumanMessage):
                            st.chat_message("human").write(msg.content)
                        elif isinstance(msg, AIMessage):
                            if getattr(msg, "content", None):
                                clean_text = extract_ai_content_fast(msg.content)
                                st.chat_message("assistant").write(clean_text)
                            if getattr(msg, "tool_calls", None):
                                with st.chat_message("assistant", avatar="🤖"):
                                    st.write("Thinking... I need to use a tool:")
                                    st.code(getattr(msg, "tool_calls", None), language="json")
                        elif isinstance(msg, ToolMessage):
                            with st.chat_message("tool", avatar="🛠️"):
                                st.write(f"Tool Result (ID: {getattr(msg,'tool_call_id', 'N/A')}):")
                                st.code(msg.content, language="json")
                        else:
                            st.write(str(msg))

            st.divider()
            # Final decision form
            st.subheader("Final Case Decision")
            with st.form("decision_form"):
                decision = sac.segmented(
                    items=[
                        sac.SegmentedItem(label='Approve SAR', icon='check-circle-fill'),
                        sac.SegmentedItem(label='Mark as False Positive', icon='x-circle-fill'),
                    ],
                    align='center',
                    format_func='title',
                )
                submitted = st.form_submit_button("Submit Decision")
                if submitted:
                    st.session_state['human_decision'] = decision
                    case_id = st.session_state['current_case_id']
                    brief = st.session_state['investigation_brief']
                    success = data_manager.archive_case(case_id, decision, brief)
                    
                    if success:
                        st.session_state['case_closed_status'] = f"Decision '{decision}' logged and archived successfully! This case is now closed."
                        st.session_state['show_return_button'] = True
                    else:
                        st.error("Failed to archive the case. Please check the logs.")
                    st.rerun()

            # Return Button Logic (Outside of Form)
            if st.session_state.get('show_return_button'):
                st.success(st.session_state.get('case_closed_status', 'Case Closed.'))
                if st.button("Return to New Alerts"):
                    st.session_state['current_case_id'] = None
                    st.session_state['current_case_details'] = None
                    st.session_state['investigation_brief'] = None
                    st.session_state['transaction_df'] = None
                    st.session_state['message_history'] = None
                    st.session_state['follow_up_chat'] = []
                    st.session_state['show_return_button'] = False 
                    st.rerun() 

# --- TAB 3: Closed Cases ---
elif selected_tab == 'Closed Cases':
    st.header("Archived Investigations")
    closed_cases_df = data_manager.get_closed_cases()
    if closed_cases_df.empty:
        st.info("No cases have been closed yet.")
    else:
        st.write(f"Showing {len(closed_cases_df)} closed cases, most recent first.")
        for _, row in closed_cases_df.iterrows():
            if row['decision'] == 'Approve SAR':
                icon = "⚠️"; color = "#FF4B4B"
            else:
                icon = "✅"; color = "#09AB3B"
            ts_display = row.get('timestamp', None)
            try:
                ts_str = pd.to_datetime(ts_display).strftime('%Y-%m-%d %H:%M')
            except Exception:
                ts_str = str(ts_display)
            with st.expander(f"**Case {row['case_id']} — Decision: {row['decision']}** (Closed on {ts_str}) {icon}"):
                st.markdown(f"**<span style='color:{color};'>{row['decision']}</span>**", unsafe_allow_html=True)
                st.subheader("Final Investigation Brief")
                st.markdown(row['brief'])