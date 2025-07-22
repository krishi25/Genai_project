import pandas as pd
import os
import random
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline
import openai
import torch

# Set OpenAI API key from environment variable (securely)
#
openai.api_key = os.getenv("OPENAI_API_KEY")
# Load toxicity classifier (Roberta-based)

toxicity_classifier = pipeline(
    "text-classification",
    model="twitter-roberta-base-offensive",
    tokenizer="twitter-roberta-base-offensive",
    top_k=None
)
# Predefined values
channels = ['Email', 'Phone', 'Chat', 'App', 'Branch']
products = ['Savings Account', 'Credit Card', 'Personal Loan', 'Home Loan',
            'Mobile Banking', 'Checking Account', 'Auto Loan', 'Mortgage']
sentiments = ['Positive', 'Negative', 'Neutral']

# Feedback samples
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
        # Toxic/hate examples
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

# Generate synthetic feedback data
def generate_sample_data(num_entries=500):
    data = []
    start_date = datetime(2025, 6, 1, 8, 0, 0)
    for _ in range(num_entries):
        sentiment = random.choice(sentiments)
        feedback = random.choice(feedback_samples[sentiment])
        entry = {
            'Timestamp': (start_date + timedelta(minutes=random.randint(0, 50000))).strftime("%Y-%m-%d %H:%M:%S"),
            'Channel': random.choice(channels),
            'Product': random.choice(products),
            'Feedback': feedback,
            'Sentiment': sentiment,
            'Customer_ID': f'CUST{random.randint(1000, 9999)}'
        }
        data.append(entry)
    return pd.DataFrame(data)

# summarize_complaints function with hallucination guardrails
def summarize_complaints(comment_list):
    prompt = (
        "You are an AI assistant for a bank, tasked with summarizing customer complaints.\n"
        "Please follow these strict rules:\n"
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
            temperature=0.3,            # Low randomness
            max_tokens=300,
            top_p=1,
            frequency_penalty=0.5,      # Penalize repeated info
            presence_penalty=0.0
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Failed to summarize complaints: {str(e)}"
# Function to check if text is toxic
def is_toxic_text(text, threshold=0.5):
    """
    Checks if the input text contains toxic/offensive content based on the toxicity_classifier.

    Args:
        text (str): Text to classify.
        threshold (float): Minimum score to consider label as toxic.

    Returns:
        bool: True if toxic content detected, else False.
    """
    try:
        preds = toxicity_classifier(text)
        labels = [p['label'].upper() for p in preds if p['score'] > threshold]
        if any(lbl in ["OFFENSIVE", "TOXIC"] for lbl in labels):
            return True
    except Exception:
        # In case of error, default to non-toxic to avoid false blocking
        pass
    return False

# Filter toxic comments using classification model
def filter_toxic_feedback(comments):
    non_toxic = []
    toxic = []
    for comment in comments:
        try:
            preds = toxicity_classifier(comment)[0]
            labels = [p['label'] for p in preds if p['score'] > 0.5]
            if "OFFENSIVE" in labels:
                toxic.append(comment)
            else:
                non_toxic.append(comment)
        except Exception:
            non_toxic.append(comment)  # Fail-safe
    return non_toxic, toxic

# ------------------- Streamlit UI Starts ------------------- #

# Generate or load data
if not os.path.exists("sample_customer_feedback.csv"):
    df = generate_sample_data()
    df.to_csv("sample_customer_feedback.csv", index=False)
else:
    df = pd.read_csv("sample_customer_feedback.csv")

df["Channel"] = df["Channel"].str.strip().str.title()
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Sidebar logo
st.sidebar.image("Bank.png", width=150)

# Title
st.title("📊 Customer Interaction Insight Copilot")

# Filters
sentiment_filter = st.selectbox("Filter by Sentiment", ["All"] + sentiments)
channel_filter = st.selectbox("Select Channel", ["All"] + sorted(df["Channel"].unique()))

filtered_df = df.copy()
if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df["Sentiment"] == sentiment_filter]
if channel_filter != "All":
    filtered_df = filtered_df[filtered_df["Channel"] == channel_filter]

# Tabs
tab1, tab2 = st.tabs(["📋 View Feedback Data", "🧠 AI-Powered Summary"])

# Tab 1: Data + Chart
with tab1:
    st.write(f"Showing {len(filtered_df)} feedback entries.")
    st.dataframe(filtered_df[["Timestamp", "Channel", "Product", "Feedback", "Sentiment"]])

    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = filtered_df["Sentiment"].value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax)
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Sentiment")
    st.pyplot(fig)

    # Download button
    st.download_button(
        label="📥 Download Filtered Feedback",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_feedbacks.csv",
        mime="text/csv"
    )

# Tab 2: AI Summary
with tab2:
    st.write("Click below to summarize filtered customer feedback (toxic content will be excluded).")
    summary_placeholder = st.empty()

if st.button("🧠 Generate Summary"):
    with st.spinner("Filtering and summarizing complaints..."):
        raw_comments = filtered_df["Feedback"].tolist()[:30]

        if not raw_comments:
            st.warning("No feedback available for summary.")
        else:
            clean_comments, toxic_comments = filter_toxic_feedback(raw_comments)

            if not clean_comments:
                st.error("All comments flagged as toxic. No summary can be generated.")
            else:
                if toxic_comments:
                    with st.expander("⚠️ Toxic Comments Removed"):
                        for tc in toxic_comments:
                            st.markdown(f"- {tc}")

                summary = summarize_complaints(clean_comments)

                # Guardrail: check if summary itself is toxic
                if is_toxic_text(summary):
                    st.error("⚠️ Generated summary contains potentially toxic content and has been blocked.")
                else:
                    summary_placeholder.subheader("📝 Summary:")
                    summary_placeholder.write(summary)

                    st.download_button(
                        label="📥 Download Summary",
                        data=summary,
                        file_name="complaint_summary.txt",
                        mime="text/plain"
                    )

    


