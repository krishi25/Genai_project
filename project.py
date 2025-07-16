import pandas as pd
import os
import random
from datetime import datetime, timedelta
import openai
import streamlit as st
import matplotlib.pyplot as plt

# Predefined values for the fields
channels = ['Email', 'Phone', 'Chat', 'App', 'Branch']
products = ['Savings Account', 'Credit Card', 'Personal Loan', 'Home Loan', 'Mobile Banking', 'Checking Account', 'Auto Loan', 'Mortgage']
sentiments = ['Positive', 'Negative', 'Neutral']

# Sample feedbacks
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
        'Loan process is unnecessarily slow'
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

# Generate 100 sample entries
data = []
start_date = datetime(2025, 6, 1, 8, 0, 0)
for i in range(100):
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
data = pd.DataFrame(data)
# Save to CSV
data.to_csv("sample_customer_feedback.csv", index=False)
print(" Sample data saved to 'sample_customer_feedback.csv'")

# Set API key securely from environment
#openai.api_key = 'sk-xxxxxxxxxxxxxxxxxxxxxxxx'  # Replace with your actual OpenAI API key or set it as an environment variable
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
# Function to summarize complaints using OpenAI API
def summarize_complaints(comment_list):
    prompt = (
        "You are an AI assistant for a bank. Summarize the key issues and trends from the following customer complaints:\n\n"
        + "\n".join(f"- {comment}" for comment in comment_list)
        + "\n\nSummary:"
    )

    try: 
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f" Failed to summarize complaints: {str(e)}"

# Load the sample data
df = pd.read_csv("sample_customer_feedback.csv",encoding='ISO-8859-1')
# Display logo image at the top (adjust width as needed)
st.sidebar.image("Bank.png", width=150)
st.title("📊Customer Interaction Insight Copilot")
df["Channel"] = df["Channel"].str.strip().str.title()

# Streamlit filters
sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "Negative", "Neutral", "Positive"])
channel_filter = st.selectbox("Select Channel", ["All"] + sorted(df["Channel"].unique()))

# Apply filters
filtered_df = df.copy()
if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df["Sentiment"] == sentiment_filter]
if channel_filter != "All":
    filtered_df = filtered_df[filtered_df["Channel"] == channel_filter]

# st.write(f"Showing {len(filtered_df)} feedback entries.")
# st.dataframe(filtered_df[["Timestamp", "Channel", "Product", "Customer_Comment", "Sentiment"]])

# Streamlit app to visualize and summarize customer feedback data
# Create tabs for viewing data and AI summary

tab1, tab2 = st.tabs(["📋 View Feedback Data", "🧠 AI-Powered Summary"])

with tab1:
    st.write(f"Showing {len(filtered_df)} feedback entries.")
    st.dataframe(filtered_df[["Timestamp", "Channel", "Product", "Feedback", "Sentiment"]])

    # Chart: Sentiment Breakdown
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = filtered_df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

with tab2:
    st.write("Click the button below to summarize the filtered customer complaints.")
    summary_placeholder = st.empty()

    if st.button("🧠 Generate Summary"):
        with st.spinner("Summarizing complaints..."):
            comments_to_summarize = filtered_df["Feedback"].tolist()[:30]  # limit for efficiency
            if len(comments_to_summarize) == 0:
                st.warning("No complaints to summarize with current filters.")
            else:
                summary = summarize_complaints(comments_to_summarize)
                summary_placeholder.subheader("📝 Summary:")
                summary_placeholder.write(summary)

                # Download button for summary
                st.download_button(
                    label="📥 Download Summary as Text File",
                    data=summary,
                    file_name="complaint_summary.txt",
                    mime="text/plain"
                )


