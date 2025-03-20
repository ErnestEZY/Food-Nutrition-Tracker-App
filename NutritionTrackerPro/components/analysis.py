import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
import random
from statsmodels.tsa.arima.model import ARIMA
from database import daily_log_collection
from utils import DIET_GOALS, calculate_bmi_adjusted_goals
import pytz

# Define Malaysia timezone
MALAYSIA_TZ = pytz.timezone('Asia/Kuala_Lumpur')

def format_number(num):
    if num >= 1000:
        return f"{num / 1000:.1f}k"
    return f"{num:.1f}"

def get_day_boundary(dt):
    """Return the datetime of the most recent 12 AM (midnight) before or at the given datetime in MST."""
    dt = dt.astimezone(MALAYSIA_TZ) if dt.tzinfo else MALAYSIA_TZ.localize(dt)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def nutrition_analysis():
    st.title("📊 Nutrition Analysis")
    
    if 'bmi_calculated' not in st.session_state or not st.session_state.bmi_calculated:
        st.warning("Please calculate your BMI on the Home page first!")
        if st.button("Go to Home Page"):
            st.session_state.page = "Home"
            st.rerun()
        st.stop()
    
    st.subheader("Personalize Your Analysis")
    st.info(f"Current BMI: {st.session_state.last_bmi:.1f}")
    diet_type = st.selectbox("Select Diet Type", list(DIET_GOALS.keys()))
    adjusted_goals = calculate_bmi_adjusted_goals(st.session_state.last_bmi, DIET_GOALS[diet_type])
    st.session_state.adjusted_goals = adjusted_goals
    st.info(f"Your BMI-adjusted daily goals: {adjusted_goals['calories']} calories, "
            f"{adjusted_goals['protein']}g protein, {adjusted_goals['carbs']}g carbs, {adjusted_goals['fat']}g fat")
    
    # Define day boundaries (12 AM) in MST
    now = datetime.now(MALAYSIA_TZ)
    today_start = get_day_boundary(now)  # 12 AM today in MST
    today_end = today_start + timedelta(days=1)  # 12 AM tomorrow in MST
    historical_start = today_start - timedelta(days=7)  # 12 AM 7 days ago in MST
    
    # Convert boundaries to UTC for querying
    today_start_utc = today_start.astimezone(pytz.UTC)
    today_end_utc = today_end.astimezone(pytz.UTC)
    historical_start_utc = historical_start.astimezone(pytz.UTC)

    # Query logs for today (from 12 AM today to 12 AM tomorrow)
    today_logs = list(daily_log_collection.find({
        "date": {"$gte": today_start_utc, "$lt": today_end_utc}
    }))

    # Initialize total_nutrients and food_breakdown
    total_nutrients = {"Calories": 0, "Protein": 0, "Carbohydrates": 0, "Fat": 0}
    food_breakdown = {}
    
    if not today_logs:
        st.warning("No food logs for today (since 12 AM). Please log some food to see your daily analysis!")
        if st.button("Go to Daily Food Log"):
            st.session_state.page = "Daily Food Log"
            st.rerun()
    else:
        for log in today_logs:
            nutrients = log.get('nutrients', {})
            food_name = log.get('food_name', 'Unknown')
            total_nutrients["Calories"] += nutrients.get('energy-kcal', 0)
            total_nutrients["Protein"] += nutrients.get('proteins', 0)
            total_nutrients["Carbohydrates"] += nutrients.get('carbohydrates', 0)
            total_nutrients["Fat"] += nutrients.get('fat', 0)
            food_breakdown[food_name] = nutrients.get('energy-kcal', 0)
    
        top_col1, top_col2 = st.columns(2)
        with top_col1:
            pie_df = pd.DataFrame.from_dict(total_nutrients, orient='index', columns=['Value'])
            pie_df = pie_df[pie_df['Value'] > 0]
            fig_pie = px.pie(pie_df, values='Value', names=pie_df.index, title='Macronutrient Distribution')
            st.plotly_chart(fig_pie)
        with top_col2:
            fig_bar = px.bar(x=list(food_breakdown.keys()), y=list(food_breakdown.values()), title='Calorie Contribution by Food')
            fig_bar.update_xaxes(title='Foods')
            fig_bar.update_yaxes(title='Calories')
            st.plotly_chart(fig_bar)
    
    # Query historical logs (from 7 days ago at 12 AM to 12 AM today) for the streak
    historical_logs = list(daily_log_collection.find({
        "date": {"$gte": historical_start_utc, "$lt": today_start_utc}
    }))
    # Include today's logs in historical_logs for streak calculation
    historical_logs.extend(today_logs)
    
    if not historical_logs:
        st.warning("No historical data available for the past 7 days.")
        hist_df = pd.DataFrame(columns=['Date', 'Calories', 'Protein', 'Carbohydrates'])
    else:
        hist_data = {}
        for log in historical_logs:
            # Convert UTC timestamp to MST
            log_date = log['date'].astimezone(MALAYSIA_TZ)
            date = get_day_boundary(log_date).strftime('%Y-%m-%d')
            nutrients = log.get('nutrients', {})
            if date not in hist_data:
                hist_data[date] = {'Calories': 0, 'Protein': 0, 'Carbohydrates': 0}
            hist_data[date]['Calories'] += nutrients.get('energy-kcal', 0)
            hist_data[date]['Protein'] += nutrients.get('proteins', 0)
            hist_data[date]['Carbohydrates'] += nutrients.get('carbohydrates', 0)
        hist_df = pd.DataFrame.from_dict(hist_data, orient='index').reset_index()
        hist_df.columns = ['Date', 'Calories', 'Protein', 'Carbohydrates']

    # 7-Day Goal Streak (uses historical data, not reset)
    st.subheader("7-Day Goal Streak")   
    if not historical_logs:
        st.warning("No historical data available for the past 7 days to calculate streak.")
    else:
        # Define goal check (within 80%-130% of target)
        def goal_met(day_data):
            if day_data is None:
                return False
    
            if not isinstance(day_data, dict):
                return False
    
            required_keys = ['Calories', 'Protein', 'Carbohydrates']
            if not all(key in day_data for key in required_keys):
                return False
    
            if any(day_data.get(key, 0) == 0 for key in required_keys):
                return False
    
            conditions = []
            for nutrient, goal_key in [('Calories', 'calories'), ('Protein', 'protein'), ('Carbohydrates', 'carbs')]:
                actual = day_data.get(nutrient, 0)
                target = adjusted_goals.get(goal_key, 0)
                if target > 0:
                    percentage = (actual / target) * 100
                    within_range = 80 <= percentage <= 130
                    conditions.append(within_range)
                else:
                    conditions.append(True)
    
            result = all(conditions)
            return result

        # Build 7-day streak grid (Monday to Sunday)
        today = get_day_boundary(now)  # Use 12 AM boundary as "today" in MST
        start_day = today - timedelta(days=today.weekday())  # Start of current week (Monday)
        streak_grid = {}

        today_str = today.strftime('%Y-%m-%d')
        today_in_hist_df = False

        if not hist_df.empty:
            hist_df['Date'] = pd.to_datetime(hist_df['Date']).dt.strftime('%Y-%m-%d')
            today_in_hist_df = today_str in hist_df['Date'].values

        if not today_in_hist_df and sum(total_nutrients.values()) > 0:
            today_row = pd.DataFrame({
                'Date': [today_str],
                'Calories': [total_nutrients['Calories']],
                'Protein': [total_nutrients['Protein']],
                'Carbohydrates': [total_nutrients['Carbohydrates']]
            })
            hist_df = pd.concat([hist_df, today_row], ignore_index=True)

        for i in range(7):
            day = (start_day + timedelta(days=i)).strftime('%Y-%m-%d')
            day_data = None
            if not hist_df.empty:
                matches = hist_df[hist_df['Date'] == day]
                if not matches.empty:
                    day_data = matches.iloc[0].to_dict()
    
            is_today = day == today_str
            if is_today:
                if day_data is None and sum(total_nutrients.values()) > 0:
                    day_data = {
                        'Date': day,
                        'Calories': total_nutrients['Calories'],
                        'Protein': total_nutrients['Protein'],
                        'Carbohydrates': total_nutrients['Carbohydrates']
                    }
    
            streak_result = goal_met(day_data) if day_data is not None else False
            streak_grid[day] = '✅' if streak_result else '❌' if day_data is not None else ' '

        # Calculate current streak
        current_streak = 0
        max_streak = 0
        for day_status in reversed(list(streak_grid.values())):
            if day_status == '✅':
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_html = ''.join([f"<td>{days[i]}</td>" for i in range(7)])
        status_html = ''.join([f"<td>{streak_grid[(start_day + timedelta(days=i)).strftime('%Y-%m-%d')]}</td>" for i in range(7)])
        
        streak_html = f"""
        <div style="background-color: #ff4444; color: white; padding: 5px; text-align: center; border-top-left-radius: 10px; border-top-right-radius: 10px;">
            7-Day Goal Streak
        </div>
        <table style="width: 100%; border-collapse: collapse; background-color: white; overflow: hidden;">
            <tr style="border-bottom: 2px solid #ddd; color: black;">
                {day_html}
            </tr>
            <tr>
                {status_html}
            </tr>
        </table>
        <div style="text-align: center; padding: 10px; background-color: #f9f9f9; border-bottom-left-radius: 10px; border-bottom-right-radius: 10px; color: black;">
            <strong>Current Streak:</strong> {current_streak} | <strong>Best Streak:</strong> {max_streak}
        </div>
        <small style="color: #666; text-align: center; display: block; margin-top: 5px;">
            ✅ = Goal Met (between 80% and 130% of target) | ❌ = Goal Missed
        </small>
        """
        st.markdown(streak_html, unsafe_allow_html=True)
    
    if st.button("Refresh Streak Table"):
        st.rerun()

    # Add Toggle for 7-Day Nutrient Trend Chart
    show_trend = st.checkbox("Show 7-Day Nutrient Trend", value=False)
    if show_trend:
        st.subheader("7-Day Nutrient Trend")
        if not hist_df.empty:
            fig_line = go.Figure()
            for nutrient in ['Calories', 'Protein', 'Carbohydrates']:
                fig_line.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df[nutrient], mode='lines+markers', name=nutrient))
            fig_line.update_layout(title='7-Day Nutrient Trend', xaxis_title='Date', yaxis_title='Amount')
            st.plotly_chart(fig_line, key="nutrient_trend_chart")
            st.markdown(
                "<small style='color: #666; text-align: center; display: block; margin-top: 5px;'>"
                "This chart shows your nutrient intake over the past 7 days, updated daily."
                "</small>",
                unsafe_allow_html=True
            )
        else:
            st.info("No historical data available for the past 7 days to display the nutrient trend.")
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
    
    # Nutritional Status (resets at 12 AM)
    st.subheader("Nutritional Status")
    st.markdown("<small style='color: #666;'>Note: Daily totals reset at 12 AM each day.</small>", unsafe_allow_html=True)
    if not today_logs:
        total_nutrients = {"Calories": 0, "Protein": 0, "Carbohydrates": 0, "Fat": 0}  # Reset to zero
    # Inject CSS to adjust font size of metric values and deltas (slightly larger)
    st.markdown("""
        <style>
        [data-testid="stMetricValue"] { font-size: 28px !important; }  /* Main value (actual/goal) */
        [data-testid="stMetricDelta"] { font-size: 18px !important; }  /* Delta (percentage) */
        </style>
    """, unsafe_allow_html=True)
    goal_cols = st.columns(4)
    goal_key_map = {"Calories": "calories", "Protein": "protein", "Carbohydrates": "carbs", "Fat": "fat"}
    for i, (nutrient, actual) in enumerate(total_nutrients.items()):
        goal_key = goal_key_map.get(nutrient, nutrient.lower())
        goal = adjusted_goals.get(goal_key, 0)
        percentage = (actual / goal) * 100 if goal > 0 else 0
        actual_str = format_number(actual)
        goal_str = format_number(goal)
        goal_cols[i].metric(nutrient, f"{actual_str} / {goal_str}", f"{percentage:.1f}%")
    
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
    
    # Daily Progress (resets at 12 AM)
    st.subheader("Daily Progress")
    st.markdown("<small style='color: #666;'>Note: Daily totals reset at 12 AM each day.</small>", unsafe_allow_html=True)
    nutrient_values = {
        "Calories": (total_nutrients["Calories"], adjusted_goals["calories"]),
        "Protein": (total_nutrients["Protein"], adjusted_goals["protein"]),
        "Carbohydrates": (total_nutrients["Carbohydrates"], adjusted_goals["carbs"]),
        "Fat": (total_nutrients["Fat"], adjusted_goals["fat"])
    }
    progress_cols = st.columns(4)
    for i, (nutrient, (current, goal)) in enumerate(nutrient_values.items()):
        with progress_cols[i]:
            st.write(nutrient)
            current_progress = min(1.0, current / goal) if goal > 0 else 0
            progress_html = f"""
            <div style="position: relative; width: 100%; height: 20px;">
                <div style="position: absolute; width: 100%; height: 100%; background-color: #4e8cff; border-radius: 5px; z-index: 1; opacity: 0.7;" title="Goal: {goal:.1f}"></div>
                <div style="position: absolute; width: {current_progress * 100}%; height: 100%; background-color: #ffeb3b; border-radius: 5px; z-index: 2; opacity: 0.9;" title="Current: {current:.1f}"></div>
            </div>
            <div style="text-align: center; margin-top: 5px;">
                <span style="color: #ffeb3b;">{current:.1f}</span> / <span style="color: #4e8cff;">{goal:.1f}</span>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
    st.markdown("""
    <small style="color: #666;">
        <span style="color: #ffeb3b;">Yellow</span> = Amount taken | 
        <span style="color: #4e8cff;">Blue</span> = Adjusted goal
    </small>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
    
    # Personal Insights (resets at 12 AM)
    st.subheader("Personal Insights")
    st.markdown("<small style='color: #666;'>Note: Daily totals reset at 12 AM each day.</small>", unsafe_allow_html=True)
    col_score, col_predict = st.columns(2)
    with col_score:
        score = sum(max(0, 100 - abs(current - goal) / goal * 100) for _, (current, goal) in nutrient_values.items() if goal > 0) / 4
        score = round(score)
        label, color = ("Good", "green") if score >= 75 else ("Fair", "orange") if score >= 50 else ("Needs Improvement", "red")
        st.metric("Daily Nutrition Score", f"{score}/100", label, delta_color="off")
        st.markdown(f"<small style='color: {color};'>Your nutrition balance: {label}</small>", unsafe_allow_html=True)
    
    with col_predict:
        st.markdown("<u>Tomorrow's Predicted Calorie Intake:</u>", unsafe_allow_html=True)
        if len(hist_df) >= 3:
            try:
                hist_df['Calories'] = pd.to_numeric(hist_df['Calories'], errors='coerce').fillna(0)
                hist_df['DayOfWeek'] = pd.to_datetime(hist_df['Date']).dt.dayofweek
                hist_df['Rolling_Mean'] = hist_df['Calories'].rolling(window=3, min_periods=1).mean()
                hist_df['Rolling_Std'] = hist_df['Calories'].rolling(window=3, min_periods=1).std().fillna(0)
                
                tomorrow = now + timedelta(days=1)
                tomorrow_day = tomorrow.weekday()
                
                if hist_df['Calories'].std() < 50:
                    forecast = hist_df['Calories'].mean() * random.uniform(0.95, 1.05)
                else:
                    try:
                        model = ARIMA(hist_df['Calories'], order=(1, 1, 1))
                        forecast = model.fit(maxiter=100).forecast(steps=1)[0]
                    except:
                        try:
                            model = ARIMA(hist_df['Calories'], order=(1, 1, 0))
                            forecast = model.fit(maxiter=100).forecast(steps=1)[0]
                        except:
                            weights = np.linspace(0, 1, len(hist_df))
                            weights = weights / weights.sum()
                            forecast = (hist_df['Calories'] * weights).sum()
                    if forecast < 500 or forecast > 5000:
                        forecast = hist_df['Calories'].mean()
                    if len(hist_df) >= 5:
                        same_day_data = hist_df[pd.to_datetime(hist_df['Date']).dt.dayofweek == tomorrow_day]
                        if not same_day_data.empty:
                            forecast = 0.7 * forecast + 0.3 * same_day_data['Calories'].mean()
                
                st.write(f"**{forecast:.0f} calories** (prediction for {tomorrow.strftime('%A')})")
                std_dev = hist_df['Calories'].std() if len(hist_df) > 1 else hist_df['Calories'].mean() * 0.1
                st.write(f"Range: {max(0, forecast - 1.5 * std_dev):.0f} - {forecast + 1.5 * std_dev:.0f} calories")
                last_value = hist_df['Calories'].iloc[-1]
                trend = "📈 Trending up" if forecast > last_value * 1.1 else "📉 Trending down" if forecast < last_value * 0.9 else "➡️ Stable"
                st.markdown(f"{trend} compared to today")
            except Exception:
                forecast = hist_df['Calories'].mean() * random.uniform(0.9, 1.1)
                st.write(f"{forecast:.0f} calories (simplified estimate)")
        else:
            st.write("Not enough data for prediction (minimum 3 days required)")
            if 'adjusted_goals' in st.session_state:
                st.write(f"Suggested target: {st.session_state.adjusted_goals.get('calories', 2000)} calories")
    
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
    
    # BMI-Based Recommendations (does not reset)
    st.subheader("BMI-Based Recommendations")
    bmi_value = st.session_state.last_bmi
    if bmi_value < 18.5:
        st.info("💡 Increase calorie intake with nutrient-dense foods. Include more protein and healthy fats.")
    elif 18.5 <= bmi_value < 25:
        st.success("💡 Maintain balanced nutrition and regular physical activity.")
    elif 25 <= bmi_value < 30:
        st.warning("💡 Moderately reduce calorie intake and increase activity. Focus on whole foods.")
    else:
        st.warning("💡 Gradual weight loss through reduced calories and exercise. Prioritize nutrient-dense foods.")
