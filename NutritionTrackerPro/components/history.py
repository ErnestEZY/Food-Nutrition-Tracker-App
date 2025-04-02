import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from database import daily_log_collection, safe_mongodb_operation
import io
import pytz

# Define Malaysia timezone
MALAYSIA_TZ = pytz.timezone('Asia/Kuala_Lumpur')

def get_day_boundary(dt):
    """Return the datetime of the most recent 12 AM (midnight) before or at the given datetime in MST."""
    dt = dt.astimezone(MALAYSIA_TZ) if dt.tzinfo else MALAYSIA_TZ.localize(dt)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def food_history():
    st.title("🕰️ Food History")
    
    # Initialize session state for pagination and delete confirmation
    if 'show_delete_confirmation' not in st.session_state:
        st.session_state.show_delete_confirmation = False
    if 'show_delete_latest_confirmation' not in st.session_state:
        st.session_state.show_delete_latest_confirmation = False
    if 'history_page' not in st.session_state:
        st.session_state.history_page = 0
    if 'records_per_page' not in st.session_state:
        st.session_state.records_per_page = 10
    
    # Check for all logs in the database
    total_logs = safe_mongodb_operation(
        lambda: daily_log_collection.count_documents({}),
        "Failed to count logs"
    ) or 0
    
    if total_logs == 0:
        st.warning("No historical data available.")
    else:
        records_per_page = st.session_state.records_per_page
        total_pages = (total_logs + records_per_page - 1) // records_per_page
        
        # Fetch the current page of logs
        history_logs = safe_mongodb_operation(
            lambda: list(daily_log_collection.find({})
                        .sort("date", -1)
                        .skip(st.session_state.history_page * records_per_page)
                        .limit(records_per_page)),
            "Failed to retrieve logs"
        ) or []
        
        if history_logs:
            history_df = pd.DataFrame(history_logs)
            # Convert UTC timestamps to MST for display
            history_df['date'] = history_df['date'].apply(
                lambda x: pytz.UTC.localize(x).astimezone(MALAYSIA_TZ).strftime("%Y-%m-%d %H:%M") if x.tzinfo is None else x.astimezone(MALAYSIA_TZ).strftime("%Y-%m-%d %H:%M")
            )
            history_df['calories'] = history_df['nutrients'].apply(lambda x: round(x.get('energy-kcal', 0), 1) if isinstance(x, dict) else 0)
            
            start_index = st.session_state.history_page * records_per_page + 1
            history_df.index = range(start_index, start_index + len(history_df))
            history_df.index.name = 'Entry No.'
            
            display_columns = ['food_name', 'brand', 'date', 'quantity', 'calories']
            display_columns = [col for col in display_columns if col in history_df.columns]
            column_renames = {'food_name': 'Food Name', 'brand': 'Brand', 'date': 'Date & Time', 'quantity': 'Servings', 'calories': 'Calories'}
            display_df = history_df[display_columns].rename(columns=column_renames)
            st.dataframe(display_df)
            
            # Pagination section
            st.write(f"Showing page {st.session_state.history_page + 1} of {total_pages} (Total records: {total_logs})")
            pagination_cols = st.columns([1, 1, 3, 1, 1])
            if pagination_cols[0].button("⏮️ First"):
                st.session_state.history_page = 0
                st.rerun()
            if pagination_cols[1].button("◀️ Prev", disabled=(st.session_state.history_page <= 0)):
                st.session_state.history_page -= 1
                st.rerun()
            new_page = pagination_cols[2].number_input("Page", min_value=1, max_value=total_pages, value=st.session_state.history_page + 1)
            if new_page != st.session_state.history_page + 1:
                st.session_state.history_page = new_page - 1
                st.rerun()
            if pagination_cols[3].button("Next ▶️", disabled=(st.session_state.history_page >= total_pages - 1)):
                st.session_state.history_page += 1
                st.rerun()
            if pagination_cols[4].button("Last ⏭️"):
                st.session_state.history_page = total_pages - 1
                st.rerun()
            
            if st.button("Export Complete History to CSV"):
                all_logs = safe_mongodb_operation(
                    lambda: list(daily_log_collection.find({}).sort("date", -1)),
                    "Failed to retrieve logs for export"
                ) or []
                export_df = pd.DataFrame(all_logs)
                export_df['date'] = export_df['date'].apply(
                    lambda x: pytz.UTC.localize(x).astimezone(MALAYSIA_TZ).strftime("%Y-%m-%d %H:%M") if x.tzinfo is None else x.astimezone(MALAYSIA_TZ).strftime("%Y-%m-%d %H:%M")
                )
                export_df['calories'] = export_df['nutrients'].apply(lambda x: round(x.get('energy-kcal', 0), 1) if isinstance(x, dict) else 0)
                export_display_df = export_df[display_columns].rename(columns=column_renames)
                csv_buffer = io.StringIO()
                export_display_df.to_csv(csv_buffer, index=True)
                st.download_button(
                    label="Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name="food_history.csv",
                    mime="text/csv"
                )
    
    if st.button("Refresh History Page"):
        st.rerun()
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    st.subheader("Modify History Data")
    # Delete Latest Food
    if total_logs > 0:  # Only show if there are logs to delete
        latest_food = safe_mongodb_operation(
            lambda: daily_log_collection.find_one({}, sort=[("date", -1)]),
            "Failed to retrieve the latest food entry"
        )
        
        if latest_food:
            # Fix timezone handling for latest_food_date
            latest_food_date_obj = latest_food['date']
            if latest_food_date_obj.tzinfo is None:
                latest_food_date_obj = pytz.UTC.localize(latest_food_date_obj)
            latest_food_date = latest_food_date_obj.astimezone(MALAYSIA_TZ).strftime("%Y-%m-%d %H:%M")
            latest_food_name = latest_food.get('food_name', 'Unknown')
            
            st.markdown(f"**Latest Food Entry:** {latest_food_name} (Logged on {latest_food_date})")
            
            def toggle_delete_latest_confirmation():
                st.session_state.show_delete_latest_confirmation = True
            
            if not st.session_state.show_delete_latest_confirmation:
                st.button("Delete Latest Food", on_click=toggle_delete_latest_confirmation)
            
            if st.session_state.show_delete_latest_confirmation:
                st.warning(f"⚠️ This action will permanently delete the latest food entry: {latest_food_name}!")
                st.info("📌 Check the box below to confirm deletion.")
                delete_latest_confirmed = st.checkbox(f"I understand and want to delete the latest food entry: {latest_food_name}", key="confirm_delete_latest")
                
                if delete_latest_confirmed:
                    def delete_latest_food_operation():
                        result = daily_log_collection.delete_one({"_id": latest_food['_id']})
                        st.session_state.show_delete_latest_confirmation = False
                        # Adjust pagination if necessary
                        if st.session_state.history_page > 0 and total_logs - 1 <= st.session_state.history_page * records_per_page:
                            st.session_state.history_page -= 1
                        if result.deleted_count == 1:
                            st.success(f"Successfully deleted the latest food entry: {latest_food_name}")
                        else:
                            st.warning("Failed to delete the latest food entry.")
                    safe_mongodb_operation(delete_latest_food_operation, "Failed to delete the latest food entry")
                    if st.button("Done"):
                        st.rerun()
                
                if st.button("Cancel"):
                    st.session_state.show_delete_latest_confirmation = False
                    st.rerun()
    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
    
    # Clear History Data
    def toggle_confirmation():
        st.session_state.show_delete_confirmation = True
    
    if not st.session_state.show_delete_confirmation:
        st.button("Clear All History Data", on_click=toggle_confirmation)
    
    if st.session_state.show_delete_confirmation:
        st.warning("⚠️ This action will permanently delete all history data!")
        st.info("📌 Check the box below to delete your food history data.")
        delete_confirmed = st.checkbox("I understand and want to delete all history data", key="confirm_delete_history")
        
        if delete_confirmed:
            def delete_history_operation():
                result = daily_log_collection.delete_many({})
                st.session_state.show_delete_confirmation = False
                st.session_state.history_page = 0
                if result.deleted_count > 0:
                    st.success(f"Successfully deleted {result.deleted_count} history records!")
                else:
                    st.warning("No records were found to delete.")
            safe_mongodb_operation(delete_history_operation, "Failed to delete history data")
            if st.button("Done"):
                st.rerun()
        
        if st.button("Cancel"):
            st.session_state.show_delete_confirmation = False
            st.rerun()
