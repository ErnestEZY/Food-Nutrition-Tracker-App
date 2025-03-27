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

    # Make today_logs dates offset-aware (UTC)
    for log in today_logs:
        if log['date'].tzinfo is None:
            log['date'] = pytz.UTC.localize(log['date'])

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
    # Make historical_logs dates offset-aware (UTC)
    for log in historical_logs:
        if log['date'].tzinfo is None:
            log['date'] = pytz.UTC.localize(log['date'])
    # Include today's logs in historical_logs for streak calculation
    historical_logs.extend(today_logs)
    
    if not historical_logs:
        st.warning("No historical data available for the past 7 days.")
        hist_df = pd.DataFrame(columns=['Date', 'Calories', 'Protein', 'Carbohydrates', 'Fat'])
    else:
        hist_data = {}
        for log in historical_logs:
            # Convert UTC timestamp to MST for aggregation
            log_date_mst = log['date'].astimezone(MALAYSIA_TZ)
            date = get_day_boundary(log_date_mst).strftime('%Y-%m-%d')
            nutrients = log.get('nutrients', {})
            if date not in hist_data:
                hist_data[date] = {'Calories': 0, 'Protein': 0, 'Carbohydrates': 0, 'Fat': 0}
            hist_data[date]['Calories'] += nutrients.get('energy-kcal', 0)
            hist_data[date]['Protein'] += nutrients.get('proteins', 0)
            hist_data[date]['Carbohydrates'] += nutrients.get('carbohydrates', 0)
            hist_data[date]['Fat'] += nutrients.get('fat', 0)
        hist_df = pd.DataFrame.from_dict(hist_data, orient='index').reset_index()
        hist_df.columns = ['Date', 'Calories', 'Protein', 'Carbohydrates', 'Fat']

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
            required_keys = ['Calories', 'Protein', 'Carbohydrates', 'Fat']
            if not all(key in day_data for key in required_keys):
                return False
            # Check if all nutrients are 0 (treat as no data for goal_met, but we'll handle this separately)
            if all(day_data.get(key, 0) == 0 for key in required_keys):
                return False
            conditions = []
            for nutrient, goal_key in [('Calories', 'calories'), ('Protein', 'protein'), ('Carbohydrates', 'carbs'), ('Fat', 'fat')]:
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

        # Determine the current week (Mon-Sun)
        today = get_day_boundary(now)  # Use 12 AM boundary as "today" in MST
        start_day = today - timedelta(days=today.weekday())  # Start of current week (Monday)
        week_identifier = start_day.strftime('%Y-%m-%d')  # Monday's date as the week identifier

        # Determine today's position in the week (0 = Monday, 6 = Sunday)
        today_index = today.weekday()  # Thursday = 3

        # Initialize session state for streak tracking
        if 'streak_week' not in st.session_state:
            st.session_state.streak_week = week_identifier
            st.session_state.current_streak = 0
            st.session_state.best_streak = 0
        else:
            # Check if the week has changed
            if st.session_state.streak_week != week_identifier:
                # Reset streaks at the start of a new week
                st.session_state.streak_week = week_identifier
                st.session_state.current_streak = 0
                st.session_state.best_streak = 0

        # Build 7-day streak grid (Monday to Sunday)
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
                'Carbohydrates': [total_nutrients['Carbohydrates']],
                'Fat': [total_nutrients['Fat']],
            })
            hist_df = pd.concat([hist_df, today_row], ignore_index=True)

        # Check for logs for each day in the current week
        for i in range(7):
            day = start_day + timedelta(days=i)
            day_str = day.strftime('%Y-%m-%d')
            day_start = day.astimezone(pytz.UTC)
            day_end = (day + timedelta(days=1)).astimezone(pytz.UTC)

            # If the day is in the future (after today), mark it as blank
            if i > today_index:
                streak_grid[day_str] = ' '
                continue

            # Check if there are any logs for this day
            day_logs = [log for log in historical_logs if day_start <= log['date'] < day_end]
            has_logs = len(day_logs) > 0

            day_data = None
            if not hist_df.empty:
                matches = hist_df[hist_df['Date'] == day_str]
                if not matches.empty:
                    day_data = matches.iloc[0].to_dict()
            is_today = day_str == today_str
            if is_today:
                if day_data is None and sum(total_nutrients.values()) > 0:
                    day_data = {
                        'Date': day_str,
                        'Calories': total_nutrients['Calories'],
                        'Protein': total_nutrients['Protein'],
                        'Carbohydrates': total_nutrients['Carbohydrates'],
                        'Fat': total_nutrients['Fat'],
                    }

            # If there are no logs for the day, mark it as ❌ (missed)
            if not has_logs and not is_today:
                streak_grid[day_str] = '❌'  # No logs for the day, mark as missed
            else:
                # If there are logs (or it's today with data), evaluate the goal
                streak_result = goal_met(day_data) if day_data is not None else False
                streak_grid[day_str] = '✅' if streak_result else '❌'

        # Calculate current streak and best streak for the days that have occurred (up to today)
        current_streak = 0
        max_streak = 0
        current_sequence = 0
        last_miss_index = -1

        # First pass: Find the most recent ❌ and calculate the best streak (only for days up to today)
        streak_values = list(streak_grid.values())[:today_index + 1]  # Only consider Mon to Thu
        for i in range(len(streak_values)):
            day_status = streak_values[i]
            if day_status == '✅':
                current_sequence += 1
                max_streak = max(max_streak, current_sequence)
            else:
                current_sequence = 0
            if day_status == '❌':
                last_miss_index = i

        # Second pass: Calculate the current streak (count consecutive ✅ days after the most recent ❌)
        if last_miss_index == -1:
            # No ❌ found, count all consecutive ✅ days from the end
            for day_status in reversed(streak_values):
                if day_status == '✅':
                    current_streak += 1
                else:
                    break
        else:
            # Count consecutive ✅ days after the most recent ❌
            current_sequence = 0
            for i in range(last_miss_index + 1, len(streak_values)):
                day_status = streak_values[i]
                if day_status == '✅':
                    current_sequence += 1
                    current_streak = current_sequence
                else:
                    current_sequence = 0
                    break

        # Update session state with the new streak values
        st.session_state.current_streak = current_streak
        st.session_state.best_streak = max(st.session_state.best_streak, max_streak)

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
            <strong>Current Streak:</strong> {st.session_state.current_streak} | <strong>Best Streak:</strong> {st.session_state.best_streak}
        </div>
        <small style="color: #666; text-align: center; display: block; margin-top: 5px;">
            ✅ = Goal Met (between 80% and 130% of target) | ❌ = Goal Missed
        </small>
        """
        st.markdown(streak_html, unsafe_allow_html=True)

    if st.button("Refresh Streak Table"):
        st.rerun()

    # Add Toggle for Daily Goal Achievement Trend Chart
    show_goal_trend = st.checkbox("Show Daily Goal Achievement Trend", value=False)
    if show_goal_trend:
        st.subheader("Daily Goal Achievement Trend (Past 7 Days)")
        if not hist_df.empty:
            goal_data = {
                'Date': [],
                'Calories': [],
                'Protein': [],
                'Carbohydrates': [],
                'Fat': []
            }
            for _, row in hist_df.iterrows():
                date = row['Date']
                goal_data['Date'].append(date)
                for nutrient, goal_key in [('Calories', 'calories'), ('Protein', 'protein'), ('Carbohydrates', 'carbs'), ('Fat', 'fat')]:
                    actual = row.get(nutrient, 0)
                    target = adjusted_goals.get(goal_key, 0)
                    percentage = (actual / target * 100) if target > 0 else 0
                    goal_data[nutrient].append(min(150, percentage))

            goal_df = pd.DataFrame(goal_data)

            fig_goal_trend = go.Figure()
            for nutrient in ['Calories', 'Protein', 'Carbohydrates', 'Fat']:
                fig_goal_trend.add_trace(go.Bar(
                    y=goal_df['Date'],
                    x=goal_df[nutrient],
                    name=nutrient,
                    orientation='h',
                ))

            fig_goal_trend.add_shape(
                type="rect",
                xref="x", yref="paper",
                x0=80, x1=130,
                y0=0, y1=1,
                fillcolor="LightGreen",
                opacity=0.3,
                layer="below",
                line_width=0,
            )
            fig_goal_trend.add_shape(
                type="line",
                xref="x", yref="paper",
                x0=100, x1=100,
                y0=0, y1=1,
                line=dict(color="Black", dash="dash")
            )
            fig_goal_trend.update_layout(
                title='Daily Goal Achievement Trend (Past 7 Days)',
                xaxis_title='Percentage of Goal (%)',
                yaxis_title='Date',
                xaxis=dict(range=[0, 150]),
                height=400,
                barmode='group',
                showlegend=True,
                margin=dict(t=50, l=25, r=25, b=25),
            )
            st.plotly_chart(fig_goal_trend, key="goal_trend_chart")
            st.markdown(
                "<small style='color: #666; text-align: center; display: block; margin-top: 5px;'>"
                "This chart shows the percentage of your daily goals achieved for each nutrient over the past 7 days. "
                "The green band indicates the acceptable range (80%–130%)."
                "</small>",
                unsafe_allow_html=True
            )
        else:
            st.info("No historical data available for the past 7 days to display the goal achievement trend.")
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
    
    # Daily Progress (resets at 12 AM) - Enhanced with percentage status
    st.subheader("Daily Progress")
    st.markdown("<small style='color: #666;'>Note: Daily totals reset at 12 AM each day.</small>", unsafe_allow_html=True)
    if not today_logs:
        total_nutrients = {"Calories": 0, "Protein": 0, "Carbohydrates": 0, "Fat": 0}
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
            percentage = (current / goal * 100) if goal > 0 else 0
            percentage_color = "#00cc00" if percentage > 0 else "#666666" 
            arrow_color = "#00cc00"  
            if percentage > 0:
                percentage_display = f"(<span style='color: {arrow_color};'>↑</span>{percentage:.1f}%)"
            else:
                percentage_display = f"({percentage:.1f}%)"
            progress_html = f"""
            <div style="position: relative; width: 100%; height: 20px;">
                <div style="position: absolute; width: 100%; height: 100%; background-color: #4e8cff; border-radius: 5px; z-index: 1; opacity: 0.7;" title="Goal: {goal:.1f}"></div>
                <div style="position: absolute; width: {current_progress * 100}%; height: 100%; background-color: #ffeb3b; border-radius: 5px; z-index: 2; opacity: 0.9;" title="Current: {current:.1f}"></div>
            </div>
            <div style="text-align: center; margin-top: 5px; font-size: 16px;">
                <span style="color: #ffeb3b;">{current:.1f}</span> / <span style="color: #4e8cff;">{goal:.1f}</span>
            </div>
            <div style="text-align: center; margin-top: 2px;">
                <span style="color: {percentage_color};">{percentage_display}</span>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
    st.markdown("""
    <small style="color: #666;">
        <span style="color: #ffeb3b;">Yellow</span> = Amount taken | 
        <span style="color: #4e8cff;">Blue</span> = Adjusted goal 
    </small>
    """, unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Add Toggle for 7-Day Nutrient Trend Chart
    show_trend = st.checkbox("Show 7-Day Nutrient Trend", value=False)
    if show_trend:
        st.subheader("7-Day Nutrient Trend")
        if not hist_df.empty:
            fig_line = go.Figure()
            for nutrient in ['Calories', 'Protein', 'Carbohydrates', 'Fat']:  # Added 'Fat'
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
    
    # Personal Insights (Daily Nutrition Score resets at 12 AM, but prediction does not)
    st.subheader("Personal Insights")
    col_score, col_predict = st.columns(2)
    with col_score:
        st.markdown("<small style='color: #666;'>Note: Daily Nutrition Score resets at 12 AM each day.</small>", unsafe_allow_html=True)
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
                
                if hist_df['Calories'].std() < 100:
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

    # Add Toggle for Top Foods Consumed Chart
    show_top_foods = st.checkbox("Show Top Foods Consumed", value=False)
    if show_top_foods:
        st.subheader("Top Foods Consumed (Past 7 Days)")
        if historical_logs:
            food_calories = {}
            for log in historical_logs:
                food_name = log.get('food_name', 'Unknown')
                calories = log.get('nutrients', {}).get('energy-kcal', 0)
                if food_name in food_calories:
                    food_calories[food_name] += calories
                else:
                    food_calories[food_name] = calories
            food_df = pd.DataFrame.from_dict(food_calories, orient='index', columns=['Calories'])
            food_df = food_df.sort_values(by='Calories', ascending=False).head(5)
            food_df = food_df[food_df['Calories'] > 0]
            if not food_df.empty:
                fig_top_foods = px.treemap(
                    food_df,
                    path=[food_df.index],
                    values='Calories',
                    title='Top Foods by Calorie Contribution (Past 7 Days)',
                    color='Calories',
                    color_continuous_scale='Blues',
                )
                fig_top_foods.update_traces(
                    textinfo="label+value+percent parent",
                    textfont=dict(size=14),
                )
                fig_top_foods.update_layout(
                    margin=dict(t=50, l=25, r=25, b=25),
                )
                st.plotly_chart(fig_top_foods, key="top_foods_chart")
                st.markdown(
                    "<small style='color: #666; text-align: center; display: block; margin-top: 5px;'>"
                    "This treemap shows the top foods contributing to your calorie intake over the past 7 days. "
                    "The size of each rectangle represents the calorie contribution."
                    "</small>",
                    unsafe_allow_html=True
                )
            else:
                st.info("No calorie data available for foods in the past 7 days.")
        else:
            st.info("No historical data available for the past 7 days to display top foods consumed.")
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
