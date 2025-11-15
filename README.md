# 🤖 AI-Powered AML Investigation Hub

This project is a complete, human-in-the-loop (HITL) web application for Anti-Money Laundering (AML) analysts. It uses a multi-agent [LangGraph](https://langchain-ai.github.io/langgraph/) system to autonomously investigate suspicious accounts, generate detailed reports, and collaborate with a human analyst for final decision-making.



---

## 🚀 Core Features

* **Automated Data Gathering:** An autonomous AI agent investigates suspicious accounts by:
    * Fetching account KYC (Know Your Customer) details.
    * Pulling the entire transaction cluster.
    * Performing sanctions screening against an OFAC list.
    * Conducting adverse media searches on the web.
* **AI-Generated Summaries:** A second AI agent (the "Summarizer") takes all the raw data and generates a professional, 5-part investigation brief, including a final risk assessment.
* **Human-in-the-Loop Chat:** A "lite" Q&A agent allows the analyst to ask follow-up questions (e.g., "What was the largest transaction?" or "Check this name against the sanctions list") without re-running the full investigation.
* **Interactive UI:** A multi-page [Streamlit](https://streamlit.io/) dashboard provides a clean interface for:
    * Viewing a dashboard of pending alerts.
    * Reviewing investigation reports and raw data.
    * Visualizing transaction activity over time.
    * Archiving final decisions.
* **Case Management:** A "Closed Cases" tab provides a full, auditable history of all completed investigations, their final reports, and the analyst's decision.

## 📸 Screenshots

### 1. New Alerts Dashboard
*The home screen shows all pending alerts and key statistics.*
![New Alerts Dashboard](docs/images/new_alerts.png)

### 2. Pre-Investigation
*The analyst selects a case and reviews the initial KYC data.*
![Pre-Investigation View](docs/images/pre_investigation.png)

### 3. Full Investigation Report
*The AI-generated brief (left) is aligned with the raw data (right) for easy review.*
![Main Investigation Report](docs/images/post_investigation.png)

### 4. Analyst Follow-up Chat
*The analyst can ask follow-up questions to the "lite" agent.*
![Follow-up Chat](docs/images/chat_follow_up.png)

### 5. Closed Cases Archive
*All completed cases are archived for review and auditing.*
![Closed Cases](docs/images/closed_cases.png)

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **AI Agent Framework:** [LangGraph](https://langchain-ai.github.io/langgraph/)
* **LLM:** Google's Gemini Pro (via `langchain-google-genai`)
* **Tools & Data:** [Pandas](https://pandas.pydata.org/), [Tavily Search](https://tavily.com/) (Web Search), [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy) (Sanctions Matching)
* **UI Components:** [Streamlit Ant Design Components](https://github.com/gkasireddy/streamlit-antd-components)
* **Charting:** [Altair](https://altair-viz.github.io/)

---

## 📦 Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME

# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# .env
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
TAVILY_API_KEY="YOUR_TAVILY_API_KEY_HERE"

streamlit run app.py