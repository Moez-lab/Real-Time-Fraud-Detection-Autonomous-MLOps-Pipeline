import os
import time
import json
import warnings
from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# =====================================================================
# 1. MOCK DATABASE (Context Enrichment / RAG Layer)
# =====================================================================
# In production, this would query Postgres, Redis, or a Vector DB.
CUSTOMER_DATABASE = {
    "CUST_8831": {
        "name": "Sarah Jenkins",
        "card_type": "Visa Platinum",
        "avg_transaction_usd": 42.50,
        "max_historical_spend": 210.00,
        "home_city": "Chicago, IL, US",
        "common_categories": ["Groceries", "Coffee", "Local Gas"],
        "account_standing": "Good (No prior fraud)"
    }
}

# =====================================================================
# 2. OPERATIONAL TOOLS (Autonomous Actions)
# =====================================================================
@tool
def execute_account_freeze(customer_id: str, reason: str) -> str:
    """Instantly freezes the customer's credit card and blocks all pending auths."""
    print(f"\n[ACTION EXECUTED] 🔒 Core Banking API: Card for {customer_id} FROZEN.")
    print(f"  Reason: {reason}")
    return f"Card for {customer_id} successfully frozen."

@tool
def route_to_soc_analyst(customer_id: str, priority: str, technical_summary: str) -> str:
    """Dispatches an urgent ticket to the Security Operations Center queue."""
    print(f"\n[ACTION EXECUTED] 🎫 SOC Ticket Created [{priority.upper()} PRIORITY]")
    print(f"  Target: {customer_id}")
    print(f"  Details: {technical_summary}")
    return f"Ticket logged in SOC queue with priority {priority}."

@tool
def send_customer_verification_sms(customer_id: str, prompt_message: str) -> str:
    """Dispatches a two-factor SMS verification challenge to the customer's phone."""
    print(f"\n[ACTION EXECUTED] 📱 SMS Gateway: Verification text sent to {customer_id}.")
    print(f"  Message: \"{prompt_message}\"")
    return f"SMS challenge sent to {customer_id}."

TOOL_REGISTRY = {
    "FREEZE_ACCOUNT": execute_account_freeze,
    "ESCALATE_TO_SOC": route_to_soc_analyst,
    "SMS_CHALLENGE": send_customer_verification_sms
}

# =====================================================================
# 3. STRUCTURED OUTPUT SCHEMA (Multi-Audience Explainability)
# =====================================================================
class FraudInvestigationReport(BaseModel):
    risk_level: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        description="Assessed threat severity based on XGBoost confidence and historical discrepancy."
    )
    technical_soc_report: str = Field(
        description="Dense, technical security analysis for SOC analysts referencing PCA anomaly, amounts, and delta from normal profile."
    )
    customer_support_script: str = Field(
        description="Empathetic, clear, non-technical explanation designed for a phone support agent to read directly to the customer."
    )
    selected_action: Literal["FREEZE_ACCOUNT", "ESCALATE_TO_SOC", "SMS_CHALLENGE"] = Field(
        description="The autonomous action to execute immediately."
    )
    action_reasoning: str = Field(
        description="Concise justification for why this specific operational tool was chosen."
    )

# =====================================================================
# 4. AGENT INITIALIZATION & PROMPT
# =====================================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1
)

# Bind Pydantic model for guaranteed structured output
structured_llm = llm.with_structured_output(FraudInvestigationReport)

system_prompt = """You are an Autonomous Fraud Defense Orchestrator for an enterprise banking platform.
You receive real-time anomaly alerts from an Apache Spark/XGBoost streaming pipeline.

Your workflow:
1. Cross-reference the raw alert telemetry against the customer's historical profile.
2. Formulate a dense, technical analysis for cybersecurity engineers (SOC).
3. Formulate a non-technical, empathetic explanation for a customer service representative.
4. Autonomously choose the appropriate mitigation tool:
   - FREEZE_ACCOUNT: For severe anomalies (AI Score > 85% or massive spending delta).
   - ESCALATE_TO_SOC: For suspicious edge-cases needing manual human triage (AI Score 70-85%).
   - SMS_CHALLENGE: For lower-tier anomalies where 2FA can resolve the ambiguity."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", """Raw Anomaly Stream:
{raw_alert}

Customer Historical Profile:
{customer_profile}

Evaluate this incident, generate the multi-audience reports, and determine the operational action.""")
])

analyst_chain = prompt_template | structured_llm

# =====================================================================
# 5. STREAM INGESTION & DISPATCH LOOP
# =====================================================================
def watch_alerts_file(filepath: str):
    file = open(filepath, "r")
    file.seek(0, os.SEEK_END)
    print(f"Autonomous Sentinel Agent active. Watching '{filepath}'...\n" + "=" * 60)
    
    while True:
        line = file.readline()
        if not line:
            time.sleep(1)
            continue
        yield line.strip()

if __name__ == "__main__":
    alert_file = "flagged_alerts.txt"
    if not os.path.exists(alert_file):
        open(alert_file, "a").close()

    # Fixed mock customer for incoming simulated stream
    target_customer_id = "CUST_8831"
    customer_profile = CUSTOMER_DATABASE[target_customer_id]

    for alert in watch_alerts_file(alert_file):
        if "FRAUD ALERT" in alert:
            print(f"\n[🚨 INGESTED PIPELINE ANOMALY]\n{alert}")
            print("\n🔍 Cross-referencing database & running multi-agent synthesis...")
            
            # 1. Run LLM Evaluation with Context & Structured Output
            report: FraudInvestigationReport = analyst_chain.invoke({
                "raw_alert": alert,
                "customer_profile": json.dumps(customer_profile, indent=2)
            })

            # 2. Display Multi-Audience Deliverables
            print("\n" + "-" * 50)
            print(f"📊 INCIDENT LEVEL: {report.risk_level}")
            print("-" * 50)
            print(f"💻 [SOC TECHNICAL INTELLIGENCE]\n{report.technical_soc_report}")
            print("-" * 50)
            print(f"🎧 [CUSTOMER SERVICE SCRIPT]\n\"{report.customer_support_script}\"")
            print("-" * 50)
            print(f"⚙️ [ACTION SELECTED]: {report.selected_action} -> {report.action_reasoning}")

            # 3. Autonomous Tool Execution
            if report.selected_action == "FREEZE_ACCOUNT":
                execute_account_freeze.invoke({
                    "customer_id": target_customer_id, 
                    "reason": report.action_reasoning
                })
            elif report.selected_action == "ESCALATE_TO_SOC":
                route_to_soc_analyst.invoke({
                    "customer_id": target_customer_id,
                    "priority": report.risk_level,
                    "technical_summary": report.technical_soc_report
                })
            elif report.selected_action == "SMS_CHALLENGE":
                send_customer_verification_sms.invoke({
                    "customer_id": target_customer_id,
                    "prompt_message": f"Did you just attempt a transaction for this card? Reply YES or NO."
                })
            print("=" * 60)