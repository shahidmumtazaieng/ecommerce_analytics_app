import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# --- 1. Configure Google API Key ---
# This block is crucial for deployment and local development.
# It tries to get the key from Streamlit secrets first (for cloud deployment),
# then from environment variables (for local dev).
# def get_google_api_key():
#     if "GOOGLE_API_KEY" in st.secrets:
#         return st.secrets["GOOGLE_API_KEY"]
#     elif os.getenv("GOOGLE_API_KEY"):
#         return os.getenv("GOOGLE_API_KEY")
#     else:
#         return None

# google_api_key = get_google_api_key()

# if not google_api_key:
#     st.error("Google API Key not found. Please set it in `.streamlit/secrets.toml` (for Streamlit Cloud) or as an environment variable `GOOGLE_API_KEY` (for local development).")
#     st.stop()

# --- 2. Initialize LangChain Components ---
# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7) # Pass key directly
# Using gemini-1.5-flash for potentially better performance/cost for this task

# Define the dynamic prompt template
template = """
You are an AI e-commerce analytics expert. Your task is to generate a detailed report based on the user's selections.

**Platform**: {platform}
**Analytics Focus**: {tool_type}

Please generate a comprehensive, actionable analytics report for the {platform} platform, specifically focusing on "{tool_type}".

The report should include:
1.  **Overview**: A brief introduction to the selected focus area for {platform}.
2.  **Key Findings/Insights**:
    *   Provide 3-5 specific, *simulated* data points or trends relevant to {tool_type} on {platform}.
    *   For example, if "Best Sellers" is chosen, list simulated top-selling products, their characteristics, and reasons for success. If "Trending Products," list emerging products, their growth rate, and target demographics.
    *   Include simulated metrics like sales velocity, customer engagement, growth rate, or market share.
3.  **Actionable Recommendations**:
    *   Suggest 2-3 concrete, strategic steps a merchant on {platform} could take based on these findings to improve their business.
4.  **Potential Challenges/Considerations**:
    *   Briefly mention any potential hurdles or factors to keep in mind.
5.  **Future Outlook**:
    *   A short paragraph on what the future might hold for {tool_type} on {platform}.

Format the output as a clear, markdown-formatted report.
"""

prompt = PromptTemplate(
    input_variables=["platform", "tool_type"],
    template=template,
)

# Create the LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False) # verbose=True for debugging prompt construction

# --- 3. Mock Data & Chart Generation Function ---
def generate_mock_data_and_chart(tool_type, platform):
    """
    Generates simulated data and a Plotly chart based on the selected tool type.
    """
    chart_data = None
    chart_title = f"{tool_type} - Simulated Data for {platform}"

    if tool_type == "Best Sellers":
        products = [f"Product A ({platform})", f"Product B ({platform})", f"Product C ({platform})", "Product D", "Product E"]
        sales = np.random.randint(5000, 25000, size=5)
        revenue = sales * np.random.uniform(20, 150, size=5)
        df = pd.DataFrame({
            "Product": products,
            "Units Sold": sales,
            "Revenue": revenue
        }).sort_values("Units Sold", ascending=False)
        chart_data = px.bar(df, x="Product", y="Units Sold", color="Revenue",
                            title=chart_title, labels={"Units Sold": "Total Units Sold", "Revenue": "Total Revenue ($)"},
                            hover_data=["Revenue"])
        chart_data.update_layout(xaxis_title="Top Products", yaxis_title="Units Sold")

    elif tool_type == "Trending Products":
        products = [f"Trendy Item X ({platform})", f"Hot Gadget Y ({platform})", "Emerging Niche Z", "Fast-Growth P", "Rising Star Q"]
        growth_rates = np.random.uniform(15, 75, size=5) # Percentage growth
        df = pd.DataFrame({
            "Product": products,
            "Growth Rate (%)": growth_rates,
            "Current Sales (Units)": np.random.randint(1000, 10000, size=5)
        }).sort_values("Growth Rate (%)", ascending=False)
        chart_data = px.bar(df, x="Product", y="Growth Rate (%)", color="Current Sales (Units)",
                            title=chart_title, labels={"Growth Rate (%)": "Monthly Growth Rate (%)"},
                            hover_data=["Current Sales (Units)"])
        chart_data.update_layout(xaxis_title="Trending Products", yaxis_title="Growth Rate (%)")

    elif tool_type == "Customer Demographics":
        age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        distribution = np.random.rand(len(age_groups))
        distribution = distribution / distribution.sum() # Normalize to 100%
        df_age = pd.DataFrame({"Age Group": age_groups, "Percentage": distribution * 100})

        genders = ["Female", "Male", "Non-binary", "Prefer not to say"]
        gender_dist = np.random.rand(len(genders))
        gender_dist = gender_dist / gender_dist.sum()
        df_gender = pd.DataFrame({"Gender": genders, "Percentage": gender_dist * 100})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Age Group Distribution - {platform}**")
            chart_data_age = px.pie(df_age, values="Percentage", names="Age Group",
                                    title="Customer Age Distribution", hole=0.3)
            st.plotly_chart(chart_data_age, use_container_width=True)
        with col2:
            st.markdown(f"**Gender Distribution - {platform}**")
            chart_data_gender = px.pie(df_gender, values="Percentage", names="Gender",
                                       title="Customer Gender Distribution", hole=0.3)
            st.plotly_chart(chart_data_gender, use_container_width=True)
        return None # Return None as charts are directly plotted

    elif tool_type == "Competitor Analysis":
        competitors = [f"Comp-A ({platform})", "Comp-B", "Comp-C", "Comp-D"]
        market_share = np.random.uniform(5, 30, size=4)
        market_share = market_share / market_share.sum() * 100 # Normalize
        pricing_index = np.random.uniform(80, 120, size=4) # 100 = parity
        df = pd.DataFrame({
            "Competitor": competitors,
            "Market Share (%)": market_share,
            "Pricing Index (vs Avg)": pricing_index
        }).sort_values("Market Share (%)", ascending=False)
        chart_data = px.bar(df, x="Competitor", y="Market Share (%)", color="Pricing Index (vs Avg)",
                            title=chart_title, labels={"Market Share (%)": "Estimated Market Share (%)"},
                            hover_data=["Pricing Index (vs Avg)"])
        chart_data.update_layout(xaxis_title="Competitor", yaxis_title="Market Share (%)")

    elif tool_type == "Seasonal Trends":
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        base_sales = np.random.randint(50000, 150000)
        seasonal_factors = np.array([0.8, 0.9, 1.1, 1.0, 1.2, 1.3, 1.1, 1.0, 1.2, 1.5, 1.4, 0.9])
        sales_volume = base_sales * seasonal_factors * np.random.uniform(0.9, 1.1, size=12)
        df = pd.DataFrame({"Month": months, "Sales Volume": sales_volume})
        chart_data = px.line(df, x="Month", y="Sales Volume", title=chart_title,
                             labels={"Sales Volume": "Simulated Sales Volume ($)"},
                             markers=True)
        chart_data.update_layout(xaxis_title="Month", yaxis_title="Sales Volume ($)")

    elif tool_type == "Marketing Campaign Effectiveness":
        campaigns = [f"Campaign A ({platform})", "Campaign B", "Campaign C", "Campaign D"]
        conversion_rates = np.random.uniform(1.5, 7.0, size=4)
        roi = np.random.uniform(50, 300, size=4) # Return on Investment
        ad_spend = np.random.randint(5000, 50000, size=4)
        df = pd.DataFrame({
            "Campaign": campaigns,
            "Conversion Rate (%)": conversion_rates,
            "ROI (%)": roi,
            "Ad Spend ($)": ad_spend
        }).sort_values("Conversion Rate (%)", ascending=False)
        chart_data = px.bar(df, x="Campaign", y="Conversion Rate (%)", color="ROI (%)",
                            title=chart_title, labels={"Conversion Rate (%)": "Avg. Conversion Rate (%)"},
                            hover_data=["Ad Spend ($)"])
        chart_data.update_layout(xaxis_title="Marketing Campaign", yaxis_title="Conversion Rate (%)")

    elif tool_type == "Inventory Optimization":
        items = [f"SKU 001 ({platform})", "SKU 002", "SKU 003", "SKU 004", "SKU 005"]
        stock_levels = np.random.randint(50, 500, size=5)
        demand_forecast = np.random.randint(70, 400, size=5)
        safety_stock = np.random.randint(10, 100, size=5)
        df = pd.DataFrame({
            "Item": items,
            "Current Stock": stock_levels,
            "Demand Forecast (Next Month)": demand_forecast,
            "Safety Stock": safety_stock
        })
        # Identify potential issues
        df['Status'] = np.where(df['Current Stock'] < df['Demand Forecast (Next Month)'] + df['Safety Stock'], 'Potential Stockout', 'Healthy')
        chart_data = px.bar(df, x="Item", y=["Current Stock", "Demand Forecast (Next Month)", "Safety Stock"],
                            title=chart_title,
                            barmode='group',
                            color="Status",
                            color_discrete_map={'Potential Stockout': 'red', 'Healthy': 'green'},
                            labels={"value": "Units", "variable": "Metric"})
        chart_data.update_layout(xaxis_title="Inventory Item", yaxis_title="Units")

    return chart_data


# --- 4. Streamlit UI Layout ---
st.set_page_config(page_title="GenAI E-commerce Analytics Tool", layout="wide", initial_sidebar_state="expanded")

st.title("🛍️ GenAI E-commerce Analytics Dashboard")
st.markdown("Unlock AI-powered insights for your e-commerce business with **simulated data** visualizations!")

# Sidebar for selections
st.sidebar.header("⚙️ Report Configuration")
st.sidebar.markdown("Select your preferences below and generate an AI-powered analytics report with key visualizations.")

platform_options = ["Shopify", "Amazon", "Etsy", "WooCommerce", "eBay", "BigCommerce"]
selected_platform = st.sidebar.selectbox(
    "1. Select E-commerce Platform",
    platform_options,
    help="Choose the platform you want to analyze."
)

tool_options = [
    "Best Sellers",
    "Trending Products",
    "Customer Demographics",
    "Competitor Analysis",
    "Seasonal Trends",
    "Marketing Campaign Effectiveness",
    "Inventory Optimization"
]
selected_tool = st.sidebar.selectbox(
    "2. Select Analytics Focus",
    tool_options,
    help="Choose the type of analytics report you need. Each focus area will have relevant visualizations."
)

st.sidebar.markdown("---")
generate_button = st.sidebar.button("🚀 Generate Analytics Report", type="primary")
st.sidebar.markdown(
    """
    <small>Powered by Google Gemini & LangChain</small>
    """,
    unsafe_allow_html=True
)

# --- 5. Display Results ---
# Initialize session state for report output and chart data
if 'report_output' not in st.session_state:
    st.session_state.report_output = "Select your options and click 'Generate Analytics Report' to view the AI-powered insights here."
if 'chart_output' not in st.session_state:
    st.session_state.chart_output = None
if 'last_platform' not in st.session_state:
    st.session_state.last_platform = None
if 'last_tool' not in st.session_state:
    st.session_state.last_tool = None

# Only regenerate if button is clicked or selections have changed
if generate_button or (st.session_state.last_platform != selected_platform or st.session_state.last_tool != selected_tool):
    st.session_state.last_platform = selected_platform
    st.session_state.last_tool = selected_tool

    with st.spinner("Generating your AI-powered analytics report and visualizations... This might take a moment."):
        try:
            # Invoke the LangChain with dynamic variables
            response = llm_chain.invoke(
                {"platform": selected_platform, "tool_type": selected_tool}
            )
            st.session_state.report_output = response['text']

            # Generate mock data and chart
            st.session_state.chart_output = generate_mock_data_and_chart(selected_tool, selected_platform)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.report_output = "Failed to generate report. Please check your API key and try again."
            st.session_state.chart_output = None

st.subheader(f"Insights for {st.session_state.last_platform} - {st.session_state.last_tool}")

# Use tabs for a cleaner separation of text report and visualizations
tab1, tab2 = st.tabs(["📊 AI Analytics Report", "📈 Key Visualizations"])

with tab1:
    st.markdown(st.session_state.report_output)

with tab2:
    if st.session_state.chart_output is not None:
        st.plotly_chart(st.session_state.chart_output, use_container_width=True)
    elif st.session_state.last_tool == "Customer Demographics":
        # Customer Demographics plots directly within the generate_mock_data_and_chart function
        # So we just need to re-call it to display if it's the current tool
        generate_mock_data_and_chart(st.session_state.last_tool, st.session_state.last_platform)
    else:
        st.info("No specific chart generated yet, or an error occurred. Generate a report to see visualizations here.")

st.markdown("---")
st.info("Disclaimer: This tool generates **simulated analytics reports and data visualizations** based on AI interpretation and predefined mock data generation. For real-world business decisions, always consult actual, up-to-date data from your e-commerce platform.")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)
