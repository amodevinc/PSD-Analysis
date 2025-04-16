import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os
import re
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_FILE = './psd_failures_cleaned_filtered.csv'
PARAMS_FILE = './component_regression_params.json'
INSIGHTS_FILE = './survival_insights_summary.csv'

# --- Column Names ---
COMPONENT_COL = 'Component'  # Korean component name
COMPONENT_EN_COL = 'Component_EN'  # English component name
LOCATION_COL = 'Location_Type_EN'  # Location type
STATION_COL = 'Station'  # Korean station name
STATION_EN_COL = 'Station_EN'  # English station name
STATION_RUNS_COL = 'Station_Daily_Runs'  # Continuous covariate

# --- Covariate Names ---
STATION_RUNS_STD_COEF = f"Q('{STATION_RUNS_COL}_std')"
LOCATION_UNDERGROUND_COEF = f"Q('{LOCATION_COL}_Underground')"
LOCATION_UNKNOWN_COEF = f"Q('{LOCATION_COL}_Unknown')"

# --- Time Horizons ---
TIME_HORIZONS_DAYS = [365, 365*2, 365*3, 365*5, 365*7, 365*10]
TIME_HORIZONS_LABELS = ["1 Year", "2 Years", "3 Years", "5 Years", "7 Years", "10 Years"]

# --- Translations ---
translations = {
    'en': {
        "page_title": "PSD Failure Analysis Dashboard",
        "dashboard_title": "🚪 PSD Failure Analysis Dashboard",
        "dashboard_subtitle": "Platform Screen Door Component Failure Probability Visualization",
        "loading_data": "Loading data...",
        "data_load_error": "Failed to load data. Please check the data files.",
        "sidebar_header": "Data Filters",
        "select_language": "Select Language:",
        "select_components": "Select Components:",
        "select_location_type": "Select Location Type:",
        "location_all": "All",
        "location_overall": "Overall",
        "location_above_ground": "Above Ground",
        "location_underground": "Underground",
        "tab_failure_curves": "Failure Curves",
        "tab_median_ttf": "Median TTF",
        "tab_custom_prediction": "Custom Prediction",
        "failure_curves_title": "Component Failure Probability Over Time",
        "failure_curves_desc": """**Methodology:** This chart visualizes how the probability of a component failing accumulates over time (in years). It uses a statistical technique called **Survival Analysis**, specifically the **Weibull Accelerated Failure Time (AFT) model**. This model is well-suited for understanding time-to-event data, like component failures.\n\n*   **Calculation:** The curves are generated from mathematical models fitted to historical failure data for each component type. Each model learns a baseline failure pattern (shape and scale parameters of the Weibull distribution) and how factors like **Location Type** and average **Station Daily Runs** influence the expected lifespan. The probability of failure by a certain time is calculated from these learned model parameters.\n*   **Interpretation:** A higher curve indicates a greater chance of failure occurring earlier. The steepness of the curve reflects how quickly the failure risk increases over time. Use the sidebar filters to compare specific components or focus on particular location types.""",
        "median_ttf_title": "Median Time to Failure Comparison",
        "median_ttf_desc": """**Methodology:** This chart compares the estimated **Median Time To Failure (Median TTF)** across different components and location types. Median TTF represents the estimated time by which 50% of components within a specific group are expected to have failed. It provides a typical lifespan estimate.\n\n*   **Calculation:** Median TTF is derived directly from the parameters of the same Weibull AFT models used for the failure curves. Specifically, it depends on the model's **shape parameter** (which describes the failure rate pattern) and its **scale parameter** (which represents the characteristic life). The scale parameter is adjusted based on the average characteristics (like daily runs) of the group being analyzed (e.g., 'Overall' represents the average across all locations for that component).\n*   **Interpretation:** Taller bars signify a longer typical operational lifespan before failure is expected. Comparing bars helps identify components or groups with significantly different expected longevities.""",
        "custom_prediction_title": "Custom Prediction Tool",
        "custom_prediction_desc": """**Methodology:** This tool allows you to generate a tailored failure prediction for a component under specific operating conditions that you define. It goes beyond the pre-calculated averages shown in the other tabs.\n\n*   **Calculation:** It starts with the base Weibull AFT model established for the selected component. Then, it **adjusts the model's parameters** (specifically, the scale or characteristic life) based on the **exact Location Type** and **Station Daily Runs** you input. This adjustment uses the relationships (coefficients) the model learned during its training on historical data, quantifying how much these specific factors accelerate or decelerate the time to failure compared to the baseline.\n*   **Usage:** Input the characteristics of a specific scenario (e.g., a particular high-traffic underground station). The tool then calculates and displays the resulting failure probability curve and Median TTF estimate for that precise case, providing a more granular risk assessment.\n\n**Important Limitation Note:** The current model shows that higher **Station Daily Runs** are associated with *longer* times to failure (higher Median TTF). This is counter-intuitive, as higher usage would typically be expected to lead to earlier failures. This likely occurs because the model does not account for **confounding factors**, such as **maintenance practices** (stations with higher usage might receive more frequent or better maintenance) or **station/component age** (newer stations might have both higher usage and more reliable components). Therefore, predictions heavily influenced by the 'Daily Runs' input should be interpreted with caution, as they may not fully reflect the real-world impact of usage without considering these other unmeasured factors.""",
        "find_station_title": "Find a Station",
        "search_station_label": "Search for a station (Korean or English name):",
        "matching_stations_title": "Matching Stations",
        "station_kr_name": "Korean Name",
        "station_en_name": "English Name",
        "station_daily_runs": "Daily Runs",
        "no_stations_found": "No stations found matching your search.",
        "configure_prediction_title": "Configure Prediction",
        "select_component_label": "Select Component:",
        "select_location_type_label": "Select Location Type:",
        "daily_runs_label": "Daily Runs:",
        "daily_runs_help": "Average number of train runs per day at the station",
        "generate_prediction_button": "Generate Prediction",
        "calculating_prediction": "Calculating prediction...",
        "failure_probabilities_title": "Failure Probabilities",
        "median_ttf_metric_label": "Estimated Median Time to Failure",
        "median_ttf_metric_unit_days": "days",
        "median_ttf_metric_unit_years": "years",
        "no_data_warning": "No data available for the selected filters.",
        "years_axis_label": "Years",
        "failure_prob_axis_label": "Failure Probability",
        "component_axis_label": "Component",
        "median_ttf_days_axis_label": "Median Time to Failure (Days)",
        "location_type_legend_label": "Location Type",
        "days_axis_label": "Days",
        "hover_failure_prob": "failure probability at",
        "hover_years": "years",
        "custom_pred_plot_title": "Custom Failure Prediction for",
        "no_model_warning": "No model parameters available for",
        # Time horizon labels
        "1_year": "1 Year", "2_years": "2 Years", "3_years": "3 Years", "5_years": "5 Years", "7_years": "7 Years", "10_years": "10 Years"
    },
    'ko': {
        "page_title": "PSD 고장 분석 대시보드",
        "dashboard_title": "🚪 PSD 고장 분석 대시보드",
        "dashboard_subtitle": "승강장 스크린도어 구성요소 고장 확률 시각화",
        "loading_data": "데이터 로딩 중...",
        "data_load_error": "데이터 로드 실패. 데이터 파일을 확인하십시오.",
        "sidebar_header": "데이터 필터",
        "select_language": "언어 선택:",
        "select_components": "구성요소 선택:",
        "select_location_type": "위치 유형 선택:",
        "location_all": "전체",
        "location_overall": "전체 평균",
        "location_above_ground": "지상",
        "location_underground": "지하",
        "tab_failure_curves": "고장 확률 곡선",
        "tab_median_ttf": "고장까지의 중위 시간 (Median TTF)",
        "tab_custom_prediction": "사용자 정의 예측",
        "failure_curves_title": "시간에 따른 구성요소 고장 확률",
        "failure_curves_desc": """**방법론:** 이 차트는 특정 시간(년)까지 구성요소가 고장날 누적 확률을 시각화합니다. **생존 분석**이라는 통계 기법, 특히 **Weibull 가속 수명 시간(AFT) 모델**을 사용합니다. 이 모델은 구성요소 고장과 같은 시간-이벤트 데이터를 이해하는 데 적합합니다.\n\n*   **계산:** 곡선은 각 구성요소 유형의 과거 고장 데이터에 맞춰진 수학적 모델에서 생성됩니다. 각 모델은 기준 고장 패턴(Weibull 분포의 형태 및 척도 모수)과 **위치 유형** 및 평균 **역별 일일 운행 횟수**와 같은 요인이 예상 수명에 미치는 영향을 학습합니다. 특정 시간까지의 고장 확률은 이러한 학습된 모델 모수에서 계산됩니다.\n*   **해석:** 곡선이 높을수록 조기 고장 가능성이 높다는 것을 의미합니다. 곡선의 기울기가 가파를수록 시간 경과에 따른 고장 위험 증가 속도가 빠르다는 것을 나타냅니다. 사이드바 필터를 사용하여 특정 구성요소를 비교하거나 특정 위치 유형에 초점을 맞출 수 있습니다.""",
        "median_ttf_title": "고장까지의 중위 시간 비교",
        "median_ttf_desc": """**방법론:** 이 차트는 다양한 구성요소 및 위치 유형에 걸쳐 예상되는 **고장까지의 중위 시간(Median TTF)**을 비교합니다. Median TTF는 특정 그룹 내 구성요소의 50%가 고장날 것으로 예상되는 시간을 나타내며, 일반적인 수명 추정치를 제공합니다.\n\n*   **계산:** Median TTF는 고장 곡선에 사용된 것과 동일한 Weibull AFT 모델의 모수에서 직접 파생됩니다. 특히 모델의 **형태 모수**(고장률 패턴 설명)와 **척도 모수**(특성 수명 나타냄)에 따라 달라집니다. 척도 모수는 분석되는 그룹(예: '전체 평균'은 해당 구성요소의 모든 위치에 대한 평균을 나타냄)의 평균 특성(예: 일일 운행 횟수)을 기반으로 조정됩니다.\n*   **해석:** 막대가 높을수록 고장 전에 예상되는 일반적인 작동 수명이 길다는 것을 의미합니다. 막대를 비교하면 예상 수명이 크게 다른 구성요소 또는 그룹을 식별하는 데 도움이 됩니다.""",
        "custom_prediction_title": "사용자 정의 예측 도구",
        "custom_prediction_desc": """**방법론:** 이 도구를 사용하면 정의한 특정 운영 조건에서 구성요소에 대한 맞춤형 고장 예측을 생성할 수 있습니다. 다른 탭에 표시된 미리 계산된 평균값을 넘어섭니다.\n\n*   **계산:** 선택한 구성요소에 대해 설정된 기본 Weibull AFT 모델로 시작합니다. 그런 다음 입력한 **정확한 위치 유형** 및 **역별 일일 운행 횟수**를 기반으로 모델의 모수(특히 척도 또는 특성 수명)를 **조정**합니다. 이 조정은 모델이 과거 데이터 학습 중에 학습한 관계(계수)를 사용하여 이러한 특정 요인이 기준선과 비교하여 고장까지의 시간을 얼마나 가속 또는 감속시키는지를 정량화합니다.\n*   **사용법:** 특정 시나리오(예: 특정 교통량이 많은 지하역)의 특성을 입력합니다. 그런 다음 도구는 해당 특정 사례에 대한 결과적인 고장 확률 곡선 및 Median TTF 추정치를 계산하고 표시하여 보다 세분화된 위험 평가를 제공합니다.\n\n**중요 제한사항 참고:** 현재 모델은 **역별 일일 운행 횟수**가 높을수록 고장까지의 시간(더 높은 Median TTF)이 *길어지는* 연관성을 보여줍니다. 이는 일반적으로 사용량이 많을수록 조기 고장으로 이어질 것으로 예상되기 때문에 직관에 반합니다. 이는 모델이 **교란 변수**를 고려하지 않기 때문에 발생할 가능성이 높습니다. 예를 들어, **유지보수 관행**(사용량이 많은 역이 더 빈번하거나 더 나은 유지보수를 받을 수 있음) 또는 **역/구성요소 노후도**(신규 역은 사용량이 많고 더 신뢰할 수 있는 구성요소를 가질 수 있음) 등이 있습니다. 따라서 '일일 운행 횟수' 입력에 크게 영향을 받는 예측은 이러한 측정되지 않은 다른 요인을 고려하지 않고 사용량의 실제 영향을 완전히 반영하지 못할 수 있으므로 주의해서 해석해야 합니다.""",
        "find_station_title": "역 찾기",
        "search_station_label": "역 검색 (한글 또는 영문명):",
        "matching_stations_title": "일치하는 역",
        "station_kr_name": "한글 역명",
        "station_en_name": "영문 역명",
        "station_daily_runs": "일일 운행 횟수",
        "no_stations_found": "검색과 일치하는 역을 찾을 수 없습니다.",
        "configure_prediction_title": "예측 구성",
        "select_component_label": "구성요소 선택:",
        "select_location_type_label": "위치 유형 선택:",
        "daily_runs_label": "일일 운행 횟수:",
        "daily_runs_help": "역의 일일 평균 열차 운행 횟수",
        "generate_prediction_button": "예측 생성",
        "calculating_prediction": "예측 계산 중...",
        "failure_probabilities_title": "고장 확률",
        "median_ttf_metric_label": "예상 고장까지의 중위 시간",
        "median_ttf_metric_unit_days": "일",
        "median_ttf_metric_unit_years": "년",
        "no_data_warning": "선택한 필터에 사용할 수 있는 데이터가 없습니다.",
        "years_axis_label": "년",
        "failure_prob_axis_label": "고장 확률",
        "component_axis_label": "구성요소",
        "median_ttf_days_axis_label": "고장까지의 중위 시간 (일)",
        "location_type_legend_label": "위치 유형",
        "days_axis_label": "일",
        "hover_failure_prob": "고장 확률",
        "hover_years": "년",
        "custom_pred_plot_title": "사용자 정의 고장 예측:",
        "no_model_warning": "사용 가능한 모델 매개변수 없음:",
        # Time horizon labels (Korean)
        "1_year": "1년", "2_years": "2년", "3_years": "3년", "5_years": "5년", "7_years": "7년", "10_years": "10년"
    }
}

# --- Helper Functions ---

@st.cache_data
def load_data():
    """Load and prepare all necessary data files."""
    try:
        # Load main data file
        df = pd.read_csv(DATA_FILE, low_memory=False)
        
        # Load insights summary
        insights_df = pd.read_csv(INSIGHTS_FILE)
        
        # Load model parameters
        with open(PARAMS_FILE, 'r') as f:
            params_data = json.load(f)
        
        return df, insights_df, params_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def calculate_survival_prob(shape, scale, horizon_days):
    """Calculates survival probability using Weibull CDF."""
    if shape <= 0 or scale <= 0 or np.isnan(shape) or np.isnan(scale):
        return np.nan
    try:
        fail_prob = stats.weibull_min.cdf(float(horizon_days), c=shape, scale=scale)
        return 1.0 - fail_prob
    except Exception as e:
        return np.nan

def calculate_median_ttf(shape, scale):
    """Calculates median Time To Failure for Weibull."""
    if shape <= 0 or scale <= 0 or np.isnan(shape) or np.isnan(scale):
        return np.nan
    try:
        # Median = scale * (ln(2))^(1/shape)
        return scale * (np.log(2)**(1/shape))
    except Exception as e:
        return np.nan

def adjust_scale_for_covariates(base_log_lambda, coefficients, scenario, std_stats):
    """
    Adjusts the Weibull scale parameter based on scenario covariates.
    """
    log_lambda = base_log_lambda

    # 1. Handle Station_Daily_Runs (Standardized Continuous)
    if STATION_RUNS_STD_COEF in coefficients and STATION_RUNS_COL in std_stats:
        mean = std_stats[STATION_RUNS_COL].get('mean', 0)
        std = std_stats[STATION_RUNS_COL].get('std', 1)
        if std > 0:
            raw_value = scenario.get(STATION_RUNS_COL, mean)
            standardized_value = (raw_value - mean) / std
            applied_coef = coefficients[STATION_RUNS_STD_COEF]
            lambda_change = applied_coef * standardized_value
            log_lambda += lambda_change

    # 2. Handle Location_Type_EN (Categorical Dummy)
    current_location = scenario.get(LOCATION_COL)
    
    # Apply Underground coefficient if applicable
    if current_location == 'Underground' and LOCATION_UNDERGROUND_COEF in coefficients:
        log_lambda += coefficients[LOCATION_UNDERGROUND_COEF] * 1
        
    # Apply Unknown coefficient if applicable
    elif current_location == 'Unknown' and LOCATION_UNKNOWN_COEF in coefficients:
        log_lambda += coefficients[LOCATION_UNKNOWN_COEF] * 1

    return np.exp(log_lambda)

def calculate_custom_survival_probabilities(component_name, station_runs, location_type, params_data):
    """
    Calculate custom survival probabilities for a component based on station runs and location.
    """
    # Get component parameters
    component_params = params_data['component_models'].get(component_name, None)
    std_stats = params_data['standardization_stats']
    
    if component_params is None:
        return None
    
    # Get model parameters
    log_shape = component_params.get('log_rho')
    base_log_lambda = component_params.get('log_lambda')
    coefficients = component_params.get('coef', {})
    
    if log_shape is None or base_log_lambda is None:
        return None
    
    # Calculate actual Weibull shape parameter
    actual_shape = np.exp(log_shape)
    
    # Create scenario
    scenario = {
        STATION_RUNS_COL: station_runs,
        LOCATION_COL: location_type
    }
    
    # Adjust scale parameter for covariates
    adjusted_scale = adjust_scale_for_covariates(base_log_lambda, coefficients, scenario, std_stats)
    
    # Calculate median time to failure
    median_ttf = calculate_median_ttf(actual_shape, adjusted_scale)
    
    # Calculate survival probabilities for each time horizon
    results = {'Median_TTF_Days': median_ttf}
    for horizon in TIME_HORIZONS_DAYS:
        surv_prob = calculate_survival_prob(actual_shape, adjusted_scale, horizon)
        results[f'Survival_Prob_{horizon}d'] = surv_prob
    
    return results

def plot_failure_curves(filtered_insights_df, lang, components=None, location_type=None):
    """
    Plot failure probability curves for selected components and location type.
    Pass selected language `lang`.
    """
    # Determine component column based on language
    lang_component_col = COMPONENT_EN_COL if lang == 'en' else COMPONENT_COL
    display_df = filtered_insights_df.copy()
    # Add the language-specific component name for display if not present
    if lang_component_col not in display_df.columns and (COMPONENT_COL in display_df.columns and COMPONENT_EN_COL in display_df.columns):
         # Simple mapping assuming EN_COL is the key
         comp_map = display_df.set_index(COMPONENT_EN_COL)[COMPONENT_COL].to_dict()
         display_df[lang_component_col] = display_df[COMPONENT_EN_COL].map(comp_map)
         
    # Translate Location Type for display
    loc_map = {
        'Overall': translations[lang]['location_overall'],
        'Above Ground': translations[lang]['location_above_ground'],
        'Underground': translations[lang]['location_underground']
    }
    display_df['Location_Display'] = display_df[LOCATION_COL].map(loc_map).fillna(display_df[LOCATION_COL]) # Keep original if no map
    
    if components:
        # Filter based on internal English name
        df = display_df[display_df[COMPONENT_EN_COL].isin(components)]
    else:
        df = display_df

    # CORRECTED: Check against internal key "All" for filtering logic
    if location_type and location_type != "All": 
         # Filter based on English key for location type
         # We receive the internal key ('All', 'Overall', 'Above Ground', 'Underground') from main()
         # No need to map back from display name here. Direct comparison is correct.
         df = df[df[LOCATION_COL] == location_type]
         # If location_type was 'All', this block is skipped.

    if df.empty:
        st.warning(translations[lang]['no_data_warning'])
        return

    fig = make_subplots(rows=1, cols=1)
    time_horizons_years = [d/365 for d in TIME_HORIZONS_DAYS]

    for component_en in df[COMPONENT_EN_COL].unique(): # Iterate using English key
        component_df = df[df[COMPONENT_EN_COL] == component_en]
        # Get the display name (English or Korean)
        component_display_name = component_df[lang_component_col].iloc[0] 

        for _, row in component_df.iterrows():
            location_display = row['Location_Display']
            failure_probs = [1 - row[f'Survival_Prob_{horizon}d'] for horizon in TIME_HORIZONS_DAYS]
            line_name = f"{component_display_name} - {location_display}"

            fig.add_trace(
                go.Scatter(
                    x=time_horizons_years,
                    y=failure_probs,
                    mode='lines+markers',
                    name=line_name,
                    hovertemplate=f"<b>%{{y:.2%}}</b> {translations[lang]['hover_failure_prob']} %{{x:.1f}} {translations[lang]['hover_years']}<extra></extra>"
                )
            )

    fig.update_layout(
        title=translations[lang]['failure_curves_title'],
        xaxis_title=translations[lang]['years_axis_label'],
        yaxis_title=translations[lang]['failure_prob_axis_label'],
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, title=translations[lang]['location_type_legend_label']),
        height=600,
        hovermode="closest"
    )
    return fig

def plot_ttf_comparison(filtered_insights_df, lang, components=None, location_type=None):
    """
    Create a bar chart comparing median time to failure.
    Pass selected language `lang`.
    """
    # Determine component column based on language
    lang_component_col = COMPONENT_EN_COL if lang == 'en' else COMPONENT_COL
    display_df = filtered_insights_df.copy()
    # Add the language-specific component name for display if not present
    if lang_component_col not in display_df.columns and (COMPONENT_COL in display_df.columns and COMPONENT_EN_COL in display_df.columns):
         comp_map = display_df.set_index(COMPONENT_EN_COL)[COMPONENT_COL].to_dict()
         display_df[lang_component_col] = display_df[COMPONENT_EN_COL].map(comp_map)
         
    # Translate Location Type for display
    loc_map = {
        'Overall': translations[lang]['location_overall'],
        'Above Ground': translations[lang]['location_above_ground'],
        'Underground': translations[lang]['location_underground']
    }
    display_df['Location_Display'] = display_df[LOCATION_COL].map(loc_map).fillna(display_df[LOCATION_COL])

    if components:
        # Filter based on internal English name
        df = display_df[display_df[COMPONENT_EN_COL].isin(components)]
    else:
        df = display_df

    # CORRECTED: Check against internal key "All" for filtering logic
    if location_type and location_type != "All": 
         # Filter based on English key for location type
         # We receive the internal key ('All', 'Overall', 'Above Ground', 'Underground') from main()
         # No need to map back from display name here. Direct comparison is correct.
         df = df[df[LOCATION_COL] == location_type]
         # If location_type was 'All', this block is skipped.

    if df.empty:
        st.warning(translations[lang]['no_data_warning'])
        return

    fig = px.bar(
        df,
        x=lang_component_col, # Use language-specific component name for x-axis
        y='Median_TTF_Days',
        color='Location_Display', # Use translated location for color legend
        barmode='group',
        labels={
            lang_component_col: translations[lang]['component_axis_label'],
            'Median_TTF_Days': translations[lang]['median_ttf_days_axis_label'],
            'Location_Display': translations[lang]['location_type_legend_label']
        },
        title=translations[lang]['median_ttf_title'],
        height=500
    )

    fig.update_layout(
        yaxis=dict(title=translations[lang]['days_axis_label']),
        yaxis2=dict(
            title=translations[lang]['years_axis_label'],
            overlaying="y", side="right", showgrid=False,
            tickvals=[365, 730, 1095, 1825, 2555, 3650],
            ticktext=["1", "2", "3", "5", "7", "10"]
        ),
        legend_title_text=translations[lang]['location_type_legend_label']
    )
    return fig

def plot_custom_prediction(component_name, station_runs, location_type_key, params_data, lang):
    """
    Plot custom prediction for a specific component based on station runs and location.
    Pass selected language `lang` and the English `location_type_key`.
    """
    results = calculate_custom_survival_probabilities(component_name, station_runs, location_type_key, params_data)

    if results is None:
        st.warning(f"{translations[lang]['no_model_warning']} {component_name}")
        return None

    time_horizons_years = [d/365 for d in TIME_HORIZONS_DAYS]
    failure_probs = [1 - results[f'Survival_Prob_{horizon}d'] for horizon in TIME_HORIZONS_DAYS]

    # Translate component and location for title/legend
    # Assuming component_name is passed as English key
    comp_display_name = component_name # Keep EN for now, need main df access or param file change for KO name
    loc_display_name = translations[lang].get(f'location_{location_type_key.lower().replace(" ", "_")}', location_type_key)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_horizons_years,
            y=failure_probs,
            mode='lines+markers',
            name=f"{comp_display_name} - {loc_display_name}",
            hovertemplate=f"<b>%{{y:.2%}}</b> {translations[lang]['hover_failure_prob']} %{{x:.1f}} {translations[lang]['hover_years']}<extra></extra>"
        )
    )

    plot_title = f"{translations[lang]['custom_pred_plot_title']} {comp_display_name} ({loc_display_name}, {station_runs} {translations[lang]['daily_runs_label']})"
    fig.update_layout(
        title=plot_title,
        xaxis_title=translations[lang]['years_axis_label'],
        yaxis_title=translations[lang]['failure_prob_axis_label'],
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        height=500,
        hovermode="closest"
    )

    for horizon, prob in zip(TIME_HORIZONS_DAYS, failure_probs):
        results[f'Failure_Prob_{horizon}d'] = prob

    return fig, results

# --- Main Dashboard ---
def main():
    # Initialize session state for language if it doesn't exist
    if 'language' not in st.session_state:
        st.session_state.language = 'en' # Default to English
        
    # Get current language
    lang = st.session_state.language
    
    # Set page config - must be the first Streamlit command
    st.set_page_config(
        page_title=translations[lang]["page_title"],
        page_icon="🚪",
        layout="wide"
    )
    
    # --- Language Selector --- 
    # Place it early, maybe at the top of the sidebar
    st.sidebar.radio(
        label=translations[lang]["select_language"],
        options=['en', 'ko'],
        format_func=lambda x: "English" if x == 'en' else "한국어",
        key='language', # Link to session state key
        horizontal=True,
    )
    # Update lang variable after potential change from radio button
    lang = st.session_state.language 
    
    st.title(translations[lang]["dashboard_title"])
    st.markdown(f"### {translations[lang]['dashboard_subtitle']}")
    
    # Load all data
    with st.spinner(translations[lang]["loading_data"]):
        df, insights_df, params_data = load_data()
    
    if df is None or insights_df is None or params_data is None:
        st.error(translations[lang]["data_load_error"])
        return
    
    # Filter out 'Unknown' location type from insights for display
    display_insights_df = insights_df[insights_df[LOCATION_COL] != 'Unknown'].copy()

    # Add Korean component name if missing (needed for display options)
    if COMPONENT_COL not in display_insights_df.columns and COMPONENT_EN_COL in display_insights_df.columns and COMPONENT_COL in df.columns:
         comp_map = df[[COMPONENT_EN_COL, COMPONENT_COL]].drop_duplicates().set_index(COMPONENT_EN_COL)[COMPONENT_COL].to_dict()
         display_insights_df[COMPONENT_COL] = display_insights_df[COMPONENT_EN_COL].map(comp_map)

    # Sidebar filters
    st.sidebar.header(translations[lang]["sidebar_header"])
    
    # --- Component Selection Logic --- 
    # 1. Get unique English component names from the insights data
    available_components_en = sorted(display_insights_df[COMPONENT_EN_COL].unique())
    
    # 2. Create the mapping from EN to KR using the main dataframe `df`
    component_en_to_kr_map = {}
    if COMPONENT_COL in df.columns and COMPONENT_EN_COL in df.columns:
        component_en_to_kr_map = df[[COMPONENT_EN_COL, COMPONENT_COL]].drop_duplicates().set_index(COMPONENT_EN_COL)[COMPONENT_COL].to_dict()
        # Ensure all available EN components have a mapping, use EN name as fallback
        for en_name in available_components_en:
            if en_name not in component_en_to_kr_map:
                 component_en_to_kr_map[en_name] = en_name # Fallback if KR name missing in main df
    else: # Fallback if KR column doesn't exist in main df
        component_en_to_kr_map = {en_name: en_name for en_name in available_components_en}

    # 3. Determine the display options and create mapping back to EN key
    if lang == 'ko':
        # Create KR options, ensure order matches available_components_en
        available_components_display = [component_en_to_kr_map.get(en_name, en_name) for en_name in available_components_en]
        # Map displayed KR name back to EN key
        component_display_to_en_map = {kr_name: en_name for en_name, kr_name in component_en_to_kr_map.items() if en_name in available_components_en}
    else: # lang == 'en'
        available_components_display = available_components_en
        # Map displayed EN name back to EN key (identity map)
        component_display_to_en_map = {en_name: en_name for en_name in available_components_en}

    # 4. Populate the multiselect widget
    selected_components_display = st.sidebar.multiselect(
        translations[lang]["select_components"],
        options=sorted(available_components_display), # Sort display names for the user
        default=sorted(available_components_display)[:3] 
    )
    
    # 5. Convert selected display names back to internal English keys for filtering
    selected_components = [component_display_to_en_map[disp_name] for disp_name in selected_components_display if disp_name in component_display_to_en_map]
    # --- End Component Selection Logic ---

    # Location type selection - Generate options from filtered data
    loc_display_map = {
        'All': translations[lang]['location_all'],
        'Overall': translations[lang]['location_overall'],
        'Above Ground': translations[lang]['location_above_ground'],
        'Underground': translations[lang]['location_underground']
    }
    available_location_keys = sorted(display_insights_df[display_insights_df[LOCATION_COL] != 'Overall'][LOCATION_COL].unique().tolist())
    location_display_options = [loc_display_map['All'], loc_display_map['Overall']] + [loc_display_map[key] for key in available_location_keys if key in loc_display_map]
    
    selected_location_display = st.sidebar.selectbox(
        translations[lang]["select_location_type"],
        options=location_display_options,
        index=0 
    )
    # Convert selected display location back to internal key or display name itself if 'All'
    selected_location = next((key for key, value in loc_display_map.items() if value == selected_location_display), selected_location_display)

    # Filter insights data based on selections
    filtered_display_data = display_insights_df.copy()
    if selected_components: # Use internal English keys for filtering
        filtered_display_data = filtered_display_data[filtered_display_data[COMPONENT_EN_COL].isin(selected_components)]
    if selected_location != "All": # Use internal English keys for filtering
        filtered_display_data = filtered_display_data[filtered_display_data[LOCATION_COL] == selected_location]
    
    # Map time horizon labels
    time_horizon_labels_map = {
        "1 Year": translations[lang]["1_year"], "2 Years": translations[lang]["2_years"], "3 Years": translations[lang]["3_years"],
        "5 Years": translations[lang]["5_years"], "7 Years": translations[lang]["7_years"], "10 Years": translations[lang]["10_years"]
    }
    time_horizon_labels_display = [time_horizon_labels_map.get(lbl, lbl) for lbl in TIME_HORIZONS_LABELS]

    # Tabs for different visualizations
    tab_labels = [translations[lang]["tab_failure_curves"], translations[lang]["tab_median_ttf"], translations[lang]["tab_custom_prediction"]]
    tab1, tab2, tab3 = st.tabs(tab_labels)

    # Tab 1: Failure curves over time
    with tab1:
        st.markdown(f"### {translations[lang]['failure_curves_title']}")
        st.markdown(translations[lang]['failure_curves_desc'])

        if filtered_display_data.empty:
            st.warning(translations[lang]['no_data_warning'])
        else:
            fig = plot_failure_curves(filtered_display_data, lang, selected_components, selected_location)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Median Time to Failure comparison
    with tab2:
        st.markdown(f"### {translations[lang]['median_ttf_title']}")
        st.markdown(translations[lang]['median_ttf_desc'])

        if filtered_display_data.empty:
            st.warning(translations[lang]['no_data_warning'])
        else:
            fig = plot_ttf_comparison(filtered_display_data, lang, selected_components, selected_location)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Custom prediction based on station runs
    with tab3:
        st.markdown(f"### {translations[lang]['custom_prediction_title']}")
        st.markdown(translations[lang]['custom_prediction_desc'])

        # Station search - with Korean and English names
        st.markdown(f"#### {translations[lang]['find_station_title']}")
        search_query = st.text_input(translations[lang]["search_station_label"], "")

        stations_info = df[[STATION_COL, STATION_EN_COL, STATION_RUNS_COL]].drop_duplicates()

        if search_query:
            filtered_stations = stations_info[
                stations_info[STATION_COL].str.contains(search_query, case=False, na=False) | 
                stations_info[STATION_EN_COL].str.contains(search_query, case=False, na=False)
            ]

            if not filtered_stations.empty:
                st.markdown(f"#### {translations[lang]['matching_stations_title']}")
                # Choose display columns based on language
                station_display_cols = {
                     STATION_COL: translations[lang]["station_kr_name"],
                     STATION_EN_COL: translations[lang]["station_en_name"],
                     STATION_RUNS_COL: translations[lang]["station_daily_runs"]
                 }
                st.dataframe(
                    filtered_stations.rename(columns=station_display_cols),
                    hide_index=True
                )
            else:
                st.warning(translations[lang]["no_stations_found"])

        st.markdown(f"#### {translations[lang]['configure_prediction_title']}")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Use display names for options, map back to EN key
            custom_component_display = st.selectbox(
                translations[lang]["select_component_label"],
                options=available_components_display
            )
            custom_component = component_display_to_en_map.get(custom_component_display, None)

        with col2:
             # Use translated location options, map back to EN key
            custom_location_options_map = {
                 translations[lang]['location_above_ground']: 'Above Ground',
                 translations[lang]['location_underground']: 'Underground'
             }
            custom_location_display = st.selectbox(
                translations[lang]["select_location_type_label"],
                options=list(custom_location_options_map.keys()),
                index=0
            )
            custom_location_key = custom_location_options_map.get(custom_location_display)

        with col3:
            mean_runs = params_data['standardization_stats'][STATION_RUNS_COL]['mean']
            custom_station_runs = st.number_input(
                translations[lang]["daily_runs_label"],
                min_value=0,
                max_value=1000, # Increased max value slightly
                value=int(mean_runs),
                help=translations[lang]["daily_runs_help"]
            )

        if st.button(translations[lang]["generate_prediction_button"]):
             if custom_component and custom_location_key:
                 with st.spinner(translations[lang]["calculating_prediction"]):
                     fig, results = plot_custom_prediction(
                         custom_component, 
                         custom_station_runs, 
                         custom_location_key,
                         params_data,
                         lang
                     )
                     
                     if fig and results:
                         st.plotly_chart(fig, use_container_width=True)
                         st.markdown(f"#### {translations[lang]['failure_probabilities_title']}")
                         metrics_col1, metrics_col2 = st.columns(2)
                         with metrics_col1:
                             st.metric(
                                 translations[lang]["median_ttf_metric_label"], 
                                 f"{results['Median_TTF_Days']:.1f} {translations[lang]['median_ttf_metric_unit_days']}",
                                 f"{results['Median_TTF_Days']/365:.1f} {translations[lang]['median_ttf_metric_unit_years']}"
                             )
                         prob_cols = st.columns(len(TIME_HORIZONS_DAYS))
                         for i, (horizon, label_key) in enumerate(zip(TIME_HORIZONS_DAYS, ["1_year", "2_years", "3_years", "5_years", "7_years", "10_years"])):
                             with prob_cols[i]:
                                 prob_value = results[f'Failure_Prob_{horizon}d'] 
                                 st.metric(
                                     translations[lang][label_key], 
                                     f"{prob_value:.1%}"
                                 )

if __name__ == "__main__":
    main() 