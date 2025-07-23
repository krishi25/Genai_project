import pandas as pd
import os
import random
import logging
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline
import openai

# ------------------- Logging Setup ------------------- #
log_filename = "customer_feedback_app.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Starting Customer Interaction Insight Copilot Application")

# ------------------- OpenAI API Key ------------------- #
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key:
        logging.info("OpenAI API key loaded successfully.")
    else:
        logging.warning("OpenAI API key not found. Summarization may fail.")
except Exception as e:
    logging.error(f"Error setting OpenAI API key: {e}")
    openai.api_key = None

# ------------------- Toxicity Classifier ------------------- #
try:
    toxicity_classifier = pipeline(
        "text-classification",
        model="twitter-roberta-base-offensive",
        tokenizer="twitter-roberta-base-offensive",
        top_k=None
    )
    logging.info("Toxicity classifier loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load toxicity classifier: {e}")
    toxicity_classifier = None

# --- Feedback Samples ---
feedback_samples = {
    'Positive': [
        'Really appreciated the prompt service',
        'App is user-friendly and fast',
        'Customer support was excellent!',
        'Got a great rate and easy approval process',
        'Agent was very polite and helpful',
        'Quick response and resolution',
        'Impressed with the seamless experience',
        'Loan approval was faster than expected',
        'Great mobile app interface',
        'Very satisfied with the support team'
    ],
    'Negative': [
        'Agent was rude during call',
        'Still waiting for callback after 3 days',
        'Charges applied without notification',
        'App keeps crashing frequently',
        'Poor customer service experience',
        'Had to repeat the issue multiple times',
        'Interest rates were misleading',
        'Payment failed but amount deducted',
        'Difficult to reach a human agent',
        'Loan process is unnecessarily slow',
        'Your service is the worst I’ve seen',
        'I hate this bank and the people who run it',
        'This is garbage service, you people don’t care',
        'You idiots messed up my account again'
    ],
    'Neutral': [
        'Could not understand the interest breakdown',
        'Received all required documents',
        'No issue, but not very helpful either',
        'Information provided was average',
        'Experience was okay, nothing special',
        'Just browsing options, didn’t apply yet',
        'Took some time but got it resolved',
        'Had to wait but the issue was fixed',
        'Neutral experience overall',
        'Feedback already submitted before'
    ]
}

channels = ['Email', 'Phone', 'Chat', 'App', 'Branch']
products = ['Savings Account', 'Credit Card', 'Personal Loan', 'Home Loan',
            'Mobile Banking', 'Checking Account', 'Auto Loan', 'Mortgage']
sentiments = ['Positive', 'Negative', 'Neutral']

# ------------------- Helper Functions ------------------- #
def get_segment(feedback):
    try:
        keywords = ["app", "loan", "support", "rate", "approval", "interface", "crash",
                    "service", "agent", "payment", "waiting", "customer", "call", "document",
                    "brand", "campaign"]
        for kw in keywords:
            if kw in feedback.lower():
                return kw
        return "other"
    except Exception as e:
        logging.error(f"Error in get_segment: {e}")
        return "other"

def assign_role(feedback):
    try:
        feedback_lower = feedback.lower()
        if any(word in feedback_lower for word in ["feature", "bug", "app", "interface", "user", "experience",
                                                   "crash", "approval", "process", "rate", "loan", "mobile",
                                                   "checking", "savings"]):
            return "Product Manager"
        elif any(word in feedback_lower for word in ["brand", "campaign", "impressed", "satisfied",
                                                     "browsing", "special", "promotion"]):
            return "Marketing Analyst"
        elif any(word in feedback_lower for word in ["support", "agent", "callback", "service", "help",
                                                     "resolution", "rude", "waiting", "charges", "payment",
                                                     "billing", "document", "issue", "resolved", "fixed",
                                                     "team", "human", "customer", "call", "response",
                                                     "repeat", "poor", "failed"]):
            return "Support Team"
        else:
            return "Support Team"
    except Exception as e:
        logging.error(f"Error in assign_role: {e}")
        return "Support Team"

def assign_action_item(feedback, role, segment=None):
    try:
        feedback_lower = feedback.lower()
        seg = segment if segment else get_segment(feedback)
        compliment_keywords = [
            "impressed", "satisfied", "excellent", "great", "appreciated", "polite", "helpful",
            "quick response", "seamless", "very satisfied", "user-friendly", "fast", "prompt service",
            "loan approval was faster", "received all required documents", "resolved", "fixed", "no issue",
            "positive", "good", "thank you", "appreciate", "feedback already submitted"
        ]
        if any(kw in feedback_lower for kw in compliment_keywords):
            return "No action required (compliment or resolved)", "Low"

        if role == "Product Manager":
            if "crash" in feedback_lower or "bug" in feedback_lower:
                return f"Create Jira ticket for bug in '{seg}' segment: '{feedback}'", "High"
            elif "feature" in feedback_lower:
                return f"Document feature request/enhancement for '{seg}' segment: '{feedback}'", "Medium"
            elif "neutral" in feedback_lower or "okay" in feedback_lower or "average" in feedback_lower:
                return f"Analyze feedback in '{seg}' segment for usability improvements: '{feedback}'", "Medium"
            elif "slow" in feedback_lower or "approval" in feedback_lower:
                return f"Review and optimize approval process in '{seg}' segment: '{feedback}'", "High"
            else:
                return f"Investigate feedback in '{seg}' segment for product improvement: '{feedback}'", "Medium"

        elif role == "Marketing Analyst":
            if "brand" in feedback_lower or "campaign" in feedback_lower:
                return f"Analyze brand/campaign impact in '{seg}' segment: '{feedback}'", "Medium"
            elif "just browsing" in feedback_lower:
                return f"Target customer in '{seg}' segment with promotional offers: '{feedback}'", "Low"
            else:
                return f"Extract marketing insights from '{seg}' segment: '{feedback}'", "Medium"

        else:  # Support Team
            if "rude" in feedback_lower or "poor" in feedback_lower or "waiting" in feedback_lower:
                return f"Escalate support issue in '{seg}' segment: '{feedback}'", "High"
            elif "callback" in feedback_lower or "call" in feedback_lower:
                return f"Schedule callback and update customer in '{seg}' segment: '{feedback}'", "High"
            elif "payment" in feedback_lower or "charges" in feedback_lower:
                return f"Review billing issue and contact customer in '{seg}' segment: '{feedback}'", "High"
            else:
                return f"Investigate and respond to support request in '{seg}' segment: '{feedback}'", "Medium"
    except Exception as e:
        logging.error(f"Error in assign_action_item: {e}")
        return "Error assigning action", "Low"

def is_toxic_text(text, threshold=0.5):
    if toxicity_classifier is None:
        logging.warning("Toxicity classifier unavailable.")
        return False
    try:
        preds = toxicity_classifier(text)
        labels = [p['label'].upper() for p in preds if p['score'] > threshold]
        return any(lbl in ["OFFENSIVE", "TOXIC"] for lbl in labels)
    except Exception as e:
        logging.error(f"Error in is_toxic_text: {e}")
        return False

def filter_toxic_feedback(comments):
    if toxicity_classifier is None:
        return comments, []
    non_toxic, toxic = [], []
    for comment in comments:
        try:
            preds = toxicity_classifier(comment)[0]
            labels = [p['label'] for p in preds if p['score'] > 0.5]
            if "OFFENSIVE" in labels:
                toxic.append(comment)
            else:
                non_toxic.append(comment)
        except Exception as e:
            logging.error(f"Error filtering comment '{comment}': {e}")
            non_toxic.append(comment)
    return non_toxic, toxic

def summarize_complaints(comment_list):
    if not openai.api_key:
        return "OpenAI API key not configured."
    prompt = (
        "You are an AI assistant for a bank, tasked with summarizing customer complaints.\n"
        "- Summarize only what is explicitly stated in the comments.\n"
        "- Do NOT invent or assume any information not present in the comments (no hallucination).\n"
        "- Avoid including any opinions, guesses, or unverified information.\n"
        "- Focus only on key issues and common themes raised by customers.\n"
        "- Do not include offensive or sensitive content.\n\n"
        "Here are the customer complaints:\n"
        + "\n".join(f"- {comment}" for comment in comment_list)
        + "\n\nSummary:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Failed to summarize complaints: {e}")
        return f"Failed to summarize complaints: {e}"

# ------------------- Data Generation ------------------- #
if not os.path.exists("sample_customer_feedback.csv"):
    try:
        data = []
        start_date = datetime(2025, 6, 1, 8, 0, 0)
        for _ in range(100):
            sentiment = random.choice(list(feedback_samples.keys()))
            feedback = random.choice(feedback_samples[sentiment])
            assigned_to = assign_role(feedback)
            segment = get_segment(feedback)
            entry = {
                'Timestamp': (start_date + timedelta(minutes=random.randint(0, 50000))).strftime("%Y-%m-%d %H:%M:%S"),
                'Channel': random.choice(channels),
                'Product': random.choice(products),
                'Feedback': feedback,
                'Sentiment': sentiment,
                'Customer_ID': f'CUST{random.randint(1000, 9999)}',
                'Assigned_To': assigned_to,
                'Segment': segment
            }
            data.append(entry)
        pd.DataFrame(data).to_csv("sample_customer_feedback.csv", index=False)
        logging.info("Sample data saved to 'sample_customer_feedback.csv'")
    except Exception as e:
        logging.error(f"Error generating sample data: {e}")

# ------------------- Load Data ------------------- #
try:
    df = pd.read_csv("sample_customer_feedback.csv", encoding='ISO-8859-1')
    df["Channel"] = df["Channel"].str.strip().str.title()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
except Exception as e:
    logging.error(f"Failed to load CSV: {e}")
    df = pd.DataFrame()

# Fix missing columns
if 'Assigned_To' not in df.columns and not df.empty:
    df['Assigned_To'] = df['Feedback'].apply(assign_role)

# ------------------- Streamlit UI ------------------- #
try:
    st.sidebar.image("Bank.png", width=150)
    st.title("Customer Interaction Insight Copilot")

    sentiment_filter = st.selectbox("Filter by Sentiment", ["All"] + sentiments)
    channel_filter = st.selectbox("Select Channel", ["All"] + sorted(df["Channel"].unique()))

    filtered_df = df.copy()
    if sentiment_filter != "All":
        filtered_df = filtered_df[filtered_df["Sentiment"] == sentiment_filter]
    if channel_filter != "All":
        filtered_df = filtered_df[filtered_df["Channel"] == channel_filter]

    tab1, tab2, tab3 = st.tabs(["📋 View Feedback Data", "🧠 AI-Powered Summary", "👥 Role-Based Routing"])

    # --- Tab 1 ---
    with tab1:
        st.write(f"Showing {len(filtered_df)} feedback entries.")
        st.dataframe(filtered_df[["Timestamp", "Channel", "Product", "Feedback", "Sentiment", "Assigned_To"]].reset_index(drop=True))
        try:
            st.subheader("Sentiment Distribution")
            sentiment_counts = filtered_df["Sentiment"].value_counts()
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error plotting sentiment chart: {e}")
            logging.error(f"Plotting error: {e}")

        st.download_button(
            label="📥 Download Filtered Feedback",
            data=filtered_df.to_csv(index=False),
            file_name="filtered_feedbacks.csv",
            mime="text/csv"
        )

    # --- Tab 2 ---
    with tab2:
        st.write("Click to summarize filtered complaints.")
        summary_placeholder = st.empty()
        if st.button("🧠 Generate Summary"):
            try:
                raw_comments = filtered_df["Feedback"].tolist()[:30]
                clean_comments, toxic_comments = filter_toxic_feedback(raw_comments)
                if not clean_comments:
                    st.error("All comments flagged as toxic. No summary can be generated.")
                else:
                    if toxic_comments:
                        with st.expander("⚠️ Toxic Comments Removed"):
                            for tc in toxic_comments:
                                st.markdown(f"- {tc}")

                    summary = summarize_complaints(clean_comments)
                    if is_toxic_text(summary):
                        st.error("⚠️ Generated summary contains toxic content.")
                    else:
                        summary_placeholder.subheader("📝 Summary:")
                        summary_placeholder.write(summary)
            except Exception as e:
                st.error(f"Error generating summary: {e}")
                logging.error(f"Summary error: {e}")

    # --- Tab 3 ---
    if 'Segment' not in filtered_df.columns and not filtered_df.empty:
        filtered_df['Segment'] = filtered_df['Feedback'].apply(get_segment)

    with tab3:
        st.markdown("<h3 style='color:#1f77b4;'>Role-Based Routing Overview</h3>", unsafe_allow_html=True)
        
        role_counts = filtered_df["Assigned_To"].value_counts()
        st.bar_chart(role_counts)
        st.markdown("<h4 style='color:green; font-weight:bold;'>Following action items generated for negative feedbacks:</h4>", unsafe_allow_html=True)

        for role in role_counts.index:
            role_df = filtered_df[filtered_df["Assigned_To"] == role].copy()
            try:
                role_df["Action_Item"], role_df["Priority"] = zip(*role_df.apply(lambda row: assign_action_item(row["Feedback"], role, row["Segment"]), axis=1))
            except Exception as e:
                logging.error(f"Error assigning action items for {role}: {e}")
                continue

            actionable_df = role_df[role_df["Action_Item"] != "No action required (compliment or resolved)"]
            if actionable_df.empty:
                continue

            segment_groups = actionable_df.groupby("Segment")
            segment_summary = []
            for seg, seg_df in segment_groups:
                feedbacks = seg_df["Feedback"].tolist()
                action_item, priority = assign_action_item(feedbacks[0], role, seg)
                button_label = "Create Jira Ticket" if role == "Product Manager" else "Create Campaign Analysis" if role == "Marketing Analyst" else "Create Incident"
                segment_summary.append({
                    "Segment": seg.title(),
                    "Action_Item": action_item,
                    "Priority": priority,
                    "Count": len(feedbacks),
                    "Button": button_label
                })

            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            segment_summary.sort(key=lambda x: (priority_order.get(x["Priority"], 99), -x["Count"]))

            with st.expander(f"{role} ({len(actionable_df)})", expanded=False):
                st.write(f"### Action Items for {role}")
                cols = st.columns([2, 5, 2, 2, 3])
                cols[0].write("Segment")
                cols[1].write("Action Item")
                cols[2].write("Priority")
                cols[3].write("Count")
                cols[4].write("Action")
                for seg in segment_summary:
                    seg_key = f"{role}_{seg['Segment']}"
                    cols = st.columns([2, 5, 2, 2, 3])
                    cols[0].write(seg['Segment'])
                    cols[1].write(seg['Action_Item'])
                    cols[2].write(seg['Priority'])
                    cols[3].write(seg['Count'])
                    if cols[4].button(seg['Button'], key=f"btn_{seg_key}"):
                        st.success(f"{seg['Button']} created for segment '{seg['Segment']}'!")

except Exception as e:
    logging.critical(f"Fatal UI error: {e}")
    st.error(f"🚨 A critical error occurred: {e}")
