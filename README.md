**Customer Interaction Insight Copilot**
This project is a Streamlit-based interactive dashboard designed to analyze, summarize, and route customer feedback. It leverages OpenAI GPT-4 for summarizing complaints, a Roberta-based toxicity classifier for detecting offensive content, and dynamic role-based action routing for customer service, product management, and marketing teams.
________________________________________
Features
**1. Customer Feedback Dashboard**
o	View and filter feedback by sentiment (Positive, Negative, Neutral) and communication channels (Email, Chat, Phone, App, Branch).
o	Download filtered feedback as a CSV file.
**2.	AI-Powered Summarization**
o	Summarizes customer complaints using OpenAI GPT-4 while enforcing hallucination guardrails.
o	Excludes toxic comments from the summarization process using Roberta-based toxicity classification.
**3.	Role-Based Routing**
o	Automatically assigns each feedback entry to Product Manager, Marketing Analyst, or Support Team based on the text content.
o	Generates action items with priority levels (High, Medium, Low).
o	Displays segments for actionable feedback with interactive action buttons (e.g., "Create Jira Ticket").
**4.	Toxicity Detection**
o	Identifies toxic or offensive content in comments using the twitter-roberta-base-offensive model.
**5.	Data Generation**
o	Generates synthetic feedback samples if no existing data file (sample_customer_feedback.csv) is found.

________________________________________
Tech Stack
•	Language: Python 3.8+
•	Framework: Streamlit
•	Visualization: Matplotlib, Streamlit Charts
•	AI Models:
o	OpenAI GPT-4
o	HuggingFace Transformers – Roberta Toxicity Classifier
•	Data Processing: Pandas
•	Logging: Python's built-in logging module
________________________________________
**Installation**

1. Clone the Repository
git clone https://github.com/yourusername/customer-insight-copilot.git
cd customer-insight-copilot
2. Create and Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
Sample requirements.txt
pandas
streamlit
matplotlib
transformers
torch
openai
________________________________________
**Configuration**
1. OpenAI API Key
Set your OpenAI API key in the environment:
export OPENAI_API_KEY="your_openai_api_key_here"

On Windows:
setx OPENAI_API_KEY "your_openai_api_key_here"
2. Bank Logo
•	Add your bank or company logo image as Bank.png in the root directory (used in the Streamlit sidebar).
________________________________________
**Usage**
Run the Streamlit App
streamlit run app.py
This will start the app on http://localhost:8501.
________________________________________
**Application Tabs**

1. View Feedback Data
•	Displays all customer feedback with columns:
o	Timestamp, Channel, Product, Feedback, Sentiment, Assigned Role
•	Provides a Sentiment Distribution chart.
•	Allows downloading filtered feedback as filtered_feedbacks.csv.
2. AI-Powered Summary
•	Filters out toxic comments.
•	Summarizes remaining feedback using GPT-4.
•	Provides a Download Summary option (complaint_summary.txt).
3. Role-Based Routing
•	Classifies feedback into roles (Product Manager, Marketing Analyst, Support Team).
•	Generates Action Items and Priority Levels.
•	Provides interactive buttons to simulate task creation (e.g., Jira Ticket).
________________________________________
**Logging**
•	All logs are saved in customer_feedback_app.log.
•	Includes:
o	Data generation logs
o	Summarization attempts
o	Toxicity checks
o	Role/action assignments
________________________________________
**Data File**
•	Default sample data is saved to sample_customer_feedback.csv if it doesn't exist.
•	Columns:
o	Timestamp, Channel, Product, Feedback, Sentiment, Customer_ID, Assigned_To, Segment
________________________________________
**Error Handling**
•	All critical operations (API calls, data loading, plotting) are wrapped in try-except blocks.
•	Errors are logged, and user-friendly messages are displayed in the Streamlit UI.

________________________________________
Future Enhancements
•	Real-time feedback ingestion from external sources (e.g., databases, APIs).
•	Live log viewer inside the Streamlit app.
•	Integration with task management tools (Jira, Trello) for real ticket creation.
•	Enhanced sentiment analysis using transformer-based models.
________________________________________
License
This project is licensed under the MIT License.
________________________________________







