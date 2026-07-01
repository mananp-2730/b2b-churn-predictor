import streamlit as st
import pandas as pd
import joblib

# 1. Page Setup
st.set_page_config(page_title="B2B Churn Predictor", page_icon="📊", layout="wide")

st.title("B2B Pipeline & Churn Predictor")
st.write("An interactive 'What-If' simulator to evaluate active leads and predict pipeline outcomes.")
st.markdown("---")

# 2. Load the Predictive Engine
# The @st.cache_resource decorator ensures we only load the model once, keeping the app lightning fast
@st.cache_resource
def load_model():
    return joblib.load('churn_pipeline_model.pkl')

model = load_model()

# 3. Sidebar: The "What-If" Controls
st.sidebar.header("Lead Characteristics")
st.sidebar.write("Adjust the parameters below to simulate different pipeline scenarios.")

with st.sidebar:
    company_size = st.selectbox("Company Size", ['Startup', 'Mid-Market', 'Enterprise'])
    lead_source = st.selectbox("Lead Source", ['Inbound Demo', 'Cold Outbound', 'Partner Referral', 'Webinar'])
    deal_value = st.number_input("Deal Value (USD)", min_value=1000, max_value=500000, value=25000, step=1000)
    
    st.markdown("### Engagement Metrics")
    days_in_pipeline = st.slider("Days in Pipeline", min_value=1, max_value=120, value=30)
    follow_up_calls = st.slider("Follow-Up Calls", min_value=0, max_value=20, value=3)
    emails_sent = st.slider("Emails Sent", min_value=0, max_value=50, value=5)
    dm_engaged = st.radio("Decision Maker Engaged?", ['Yes', 'No'])

# 4. Format Data for the Model
dm_engaged_val = 1 if dm_engaged == 'Yes' else 0

input_data = pd.DataFrame({
    'Company_Size': [company_size],
    'Lead_Source': [lead_source],
    'Deal_Value_USD': [deal_value],
    'Days_In_Pipeline': [days_in_pipeline],
    'Follow_Up_Calls': [follow_up_calls],
    'Emails_Sent': [emails_sent],
    'Decision_Maker_Engaged': [dm_engaged_val]
})

# 5. The Prediction Display
st.subheader("Real-Time Deal Analysis")

# Get probability of class 1 (Won)
win_prob = model.predict_proba(input_data)[0][1]
churn_risk = 1 - win_prob

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Probability to Close (Win)", value=f"{win_prob * 100:.1f}%")
with col2:
    st.metric(label="Risk of Churn (Loss)", value=f"{churn_risk * 100:.1f}%")

st.progress(win_prob)

# 6. Strategic Business Logic
st.markdown("### Strategic Recommendation")
if win_prob > 0.6:
    st.success("High viability lead. Maintain current engagement sequence. Prioritize closing.")
elif win_prob > 0.3:
    st.warning("At-risk pipeline. Consider escalating to leadership, offering a strategic discount, or increasing high-value touchpoints.")
else:
    st.error("Critical churn probability. The deal has likely stalled. Assess resource allocation before continuing pursuit.")

st.markdown("---")
st.info(" **What-If Analysis:** Try moving the 'Days in Pipeline' slider past 90 days, or change 'Decision Maker Engaged' to 'Yes', and watch how the predicted outcomes shift in real-time.")