import streamlit as st
from datetime import datetime, timedelta
import time
from database import food_collection, daily_log_collection, safe_mongodb_operation
from utils import DIET_GOALS, calculate_bmi_adjusted_goals
import pytz  # Add pytz for timezone handling

# Define Malaysia timezone
MALAYSIA_TZ = pytz.timezone('Asia/Kuala_Lumpur')

def get_day_boundary(dt):
    """Return the datetime of the most recent 12 AM (midnight) before or at the given datetime in MST."""
    dt = dt.astimezone(MALAYSIA_TZ) if dt.tzinfo else MALAYSIA_TZ.localize(dt)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def daily_food_log():
    st.title("📝 Daily Food Log")
    
    st.subheader("Personalize Your Nutrition")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'bmi_calculated' in st.session_state and st.session_state.bmi_calculated:
            st.info(f"Your BMI: {st.session_state.last_bmi:.1f}")
            if 'height' in st.session_state and 'weight' in st.session_state:
                with st.expander("BMI Calculation Details"):
                    st.write(f"Weight: {st.session_state.weight:.1f} kg")
                    st.write(f"Height: {st.session_state.height:.2f} m")
                    st.write(f"BMI = Weight / (Height)² = {st.session_state.weight:.1f} / ({st.session_state.height:.2f})² = {st.session_state.last_bmi:.1f}")
        else:
            st.warning("Please calculate your BMI on the Home page first!")
            if st.button("Go to Home Page"):
                st.session_state.page = "Home"
                st.rerun()
            st.stop()
    
    with col2:
        diet_type = st.selectbox("Select Diet Type", list(DIET_GOALS.keys()))
    
    adjusted_goals = calculate_bmi_adjusted_goals(st.session_state.last_bmi, DIET_GOALS[diet_type])
    st.info(f"Your BMI-adjusted daily goals: {adjusted_goals['calories']} calories, "
            f"{adjusted_goals['protein']}g protein, {adjusted_goals['carbs']}g carbs, {adjusted_goals['fat']}g fat")
    st.session_state.adjusted_goals = adjusted_goals
    
    search_query = st.text_input("Search Food Items", "")
    
    def format_food_display(food):
        product_name = food.get('product_name', 'Unknown')
        brand = food.get('brands', '')
        return f"{product_name} ({brand})" if brand else product_name
    
    def has_valid_brand(food):
        return bool(food.get('brands', '').strip())
    
    if search_query:
        def search_food_operation():
            # Search using the product_name index
            product_name_results = list(food_collection.aggregate([
                {
                    "$search": {
                        "index": "product_name",
                        "text": {
                            "query": search_query,
                            "path": "product_name"
                        }
                    }
                },
                {"$limit": 30} 
            ]))

            # Search using the brands index
            brands_results = list(food_collection.aggregate([
                {
                    "$search": {
                        "index": "brands",
                        "text": {
                            "query": search_query,
                            "path": "brands"
                        }
                    }
                },
                {"$limit": 30}
            ]))

            # Combine results and deduplicate by _id
            combined_results = {doc["_id"]: doc for doc in product_name_results + brands_results}.values()
            return list(combined_results)[:20]  # Limit total results to 20

        search_results = safe_mongodb_operation(search_food_operation, "Food search failed") or []
        search_results = [food for food in search_results if has_valid_brand(food)]
        
        if not search_results:
            st.info("No matching foods found with valid brands.")
            def get_default_foods_operation():
                return list(food_collection.find({"brands": {"$ne": "", "$exists": True}}).limit(20))
            all_foods = safe_mongodb_operation(get_default_foods_operation, "Failed to retrieve default foods") or []
            food_display_names = {format_food_display(food): food['product_name'] for food in all_foods}
        else:
            food_display_names = {format_food_display(food): food['product_name'] for food in search_results}
    else:
        def get_recent_foods_operation():
            return list(food_collection.find({"brands": {"$ne": "", "$exists": True}}).sort("_id", -1).limit(20))
        recent_foods = safe_mongodb_operation(get_recent_foods_operation, "Failed to retrieve recent foods") or []
        food_display_names = {format_food_display(food): food['product_name'] for food in recent_foods}
    
    if food_display_names:
        selected_display = st.selectbox("Select Food", list(food_display_names.keys()))
        selected_food = food_display_names[selected_display]
    else:
        st.warning("No foods with brands available.")
        st.stop()
    
    food_details = food_collection.find_one({"product_name": selected_food})
    if food_details:
        st.subheader("Food Nutrition Details")
        st.markdown(f"**{food_details.get('product_name', 'Unknown')}** by **{food_details.get('brands', '')}**")
        nutritional_cols = st.columns(3)
        nutrients = [
            ("Calories", food_details.get('nutriments', {}).get('energy-kcal', 0)),
            ("Protein (g)", food_details.get('nutriments', {}).get('proteins', 0)),
            ("Carbs (g)", food_details.get('nutriments', {}).get('carbohydrates', 0)),
            ("Fat (g)", food_details.get('nutriments', {}).get('fat', 0))
        ]
        for i, (name, value) in enumerate(nutrients):
            nutritional_cols[i % 3].metric(name, f"{value:.1f}")
    
    quantity = st.number_input("Quantity (servings)", min_value=0.25, max_value=10.0, value=1.0, step=0.25)
    
    if st.button("Add to Daily Log"):
        def add_to_log_operation():
            scaled_nutrients = {k: v * quantity if isinstance(v, (int, float)) else v for k, v in food_details.get('nutriments', {}).items()}
            # Record timestamp in UTC, but ensure it's based on MST
            mst_time = datetime.now(MALAYSIA_TZ)
            utc_time = mst_time.astimezone(pytz.UTC)
            daily_log_collection.insert_one({
                "food_name": selected_food,
                "brand": food_details.get('brands', ''),
                "display_name": format_food_display(food_details),
                "date": utc_time,  # Store in UTC
                "quantity": quantity,
                "nutrients": scaled_nutrients
            })
            st.success(f"{selected_food} ({quantity} serving{'s' if quantity > 1 else ''}) added to daily log!")
            time.sleep(1)
            st.rerun()
        safe_mongodb_operation(add_to_log_operation, "Failed to add food to daily log")
    
    now = datetime.now(MALAYSIA_TZ)
    today_start = get_day_boundary(now)  # 12 AM today in MST
    today_end = today_start + timedelta(days=1)  # 12 AM tomorrow in MST
    today_start_utc = today_start.astimezone(pytz.UTC)
    today_end_utc = today_end.astimezone(pytz.UTC)
    # Display Recent Additions (since 12 AM MST)
    st.subheader("Recent Additions (since 12 AM)")
    recent_logs = list(daily_log_collection.find({
        "date": {"$gte": today_start_utc, "$lt": today_end_utc},
        "brand": {"$ne": ""}
    }).sort("date", -1).limit(5))
    
    if recent_logs:
        for log in recent_logs:
            # Convert UTC timestamp back to MST for display
            log_date = log.get('date', datetime.now(pytz.UTC))
            log_date_mst = log_date.astimezone(MALAYSIA_TZ)
            display_name = log.get("display_name", f"{log.get('food_name', 'Unknown')} ({log.get('brand', '')})")
            date = log_date_mst.strftime("%Y-%m-%d %H:%M")
            quantity = log.get('quantity', 1)
            calories = log.get('nutrients', {}).get('energy-kcal', 0)
            st.write(f"• {display_name} ({quantity} serving{'s' if quantity > 1 else ''}) - {calories:.1f} calories - {date}")
    else:
        st.info("No recent additions with valid brands to display (since 12 AM).")
