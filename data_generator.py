import pandas as pd
import numpy as np
import random
import os

# Set a random seed so our "random" data is reproducible every time we run it
np.random.seed(42)
random.seed(42)

def generate_crm_data(num_records=1000):
    print(f"Generating {num_records} simulated B2B CRM records...")
    
    # 1. Base Variables
    company_sizes = ['Startup', 'Mid-Market', 'Enterprise']
    lead_sources = ['Inbound Demo', 'Cold Outbound', 'Partner Referral', 'Webinar']
    
    data = []
    
    for i in range(num_records):
        size = random.choices(company_sizes, weights=[0.4, 0.4, 0.2])[0]
        source = random.choices(lead_sources, weights=[0.3, 0.4, 0.2, 0.1])[0]
        
        # 2. Simulate Deal Value based on Company Size (in USD)
        if size == 'Startup':
            deal_value = np.random.normal(5000, 1500)
        elif size == 'Mid-Market':
            deal_value = np.random.normal(25000, 5000)
        else: # Enterprise
            deal_value = np.random.normal(100000, 20000)
            
        deal_value = max(1000, round(deal_value, -2)) # Round to nearest 100, minimum 1k
        
        # 3. Simulate Engagement Metrics
        days_in_pipeline = int(max(7, np.random.normal(45, 20)))
        follow_up_calls = int(max(0, np.random.normal(5, 3)))
        emails_sent = int(max(2, np.random.normal(12, 5)))
        decision_maker_engaged = random.choices([1, 0], weights=[0.6, 0.4])[0]
        
        # 4. The Logic Engine: Determine Win/Loss based on realistic conditions
        win_probability = 0.1 # Base probability
        
        if decision_maker_engaged == 1:
            win_probability += 0.3
        if follow_up_calls > 3 and follow_up_calls < 10:
            win_probability += 0.2
        if source == 'Partner Referral':
            win_probability += 0.15
        if days_in_pipeline > 90:
            win_probability -= 0.3 # Deals that drag on usually die
            
        # Cap probability between 0.05 and 0.95
        win_probability = max(0.05, min(0.95, win_probability))
        
        # 5. Final Status
        status = np.random.choice(['Won', 'Lost'], p=[win_probability, 1 - win_probability])
        
        # Append to our list
        data.append({
            'Lead_ID': f"LD-{1000 + i}",
            'Company_Size': size,
            'Lead_Source': source,
            'Deal_Value_USD': deal_value,
            'Days_In_Pipeline': days_in_pipeline,
            'Follow_Up_Calls': follow_up_calls,
            'Emails_Sent': emails_sent,
            'Decision_Maker_Engaged': decision_maker_engaged,
            'Status': status
        })
        
    # Convert to a Pandas DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = 'historical_crm_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Success! Data saved to {os.path.abspath(output_path)}")
    
    return df

if __name__ == "__main__":
    df = generate_crm_data(1000)
    print("\n--- Data Preview ---")
    print(df.head())
    print("\n--- Win/Loss Distribution ---")
    print(df['Status'].value_counts())