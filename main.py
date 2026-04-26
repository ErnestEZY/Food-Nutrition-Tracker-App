import streamlit as st
import random
import time
from datetime import datetime
from components.auth import auth_page, is_logged_in, current_name, current_user, logout, SESSION_DURATION
from components.home import home_page, load_bmi_from_db
from components.daily_log import daily_food_log
from components.analysis import nutrition_analysis
from components.history import food_history
from components.settings import settings


def main():
    # ── Auth gate ─────────────────────────────────────────────────────────────
    if not is_logged_in():
        auth_page()
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    # Restore BMI from DB once per session
    load_bmi_from_db()

    st.sidebar.title("🧭 Tracker Pro Navigation")
    st.sidebar.markdown(f"👤 **{current_name()}** (`{current_user()}`)")

    # ── Session countdown timer (JS-driven, browser-side) ─────────────────────
    login_time = st.session_state.get("login_time")
    if login_time:
        from datetime import timezone
        remaining_secs = int(
            SESSION_DURATION.total_seconds()
            - (datetime.now(timezone.utc) - login_time).total_seconds()
        )
        remaining_secs = max(0, remaining_secs)

        # If already expired on this rerun, is_logged_in() will have caught it above.
        # Inject a JS countdown that:
        #   - shows a subtle pill in the sidebar at all times
        #   - turns amber when ≤ 5 min remain
        #   - turns red + pulses when ≤ 2 min remain
        #   - shows a modal dialog at 0 and reloads the page
        st.sidebar.markdown(
            f"""
            <div id="session-timer-wrap" style="
                display:flex; align-items:center; gap:6px;
                padding:5px 10px; border-radius:20px;
                background:#f0f4ff; border:1px solid #c8d8ff;
                font-size:13px; color:#3a5bd9; margin-bottom:4px;
                transition: background 0.5s, color 0.5s, border 0.5s;">
              <span id="session-timer-icon">⏱</span>
              <span>Session: <b id="session-timer">--:--:--</b></span>
            </div>

            <div id="session-expired-modal" style="
                display:none; position:fixed; inset:0; z-index:9999;
                background:rgba(0,0,0,0.55); align-items:center; justify-content:center;">
              <div style="
                  background:#fff; border-radius:14px; padding:36px 40px;
                  max-width:360px; text-align:center; box-shadow:0 8px 32px rgba(0,0,0,0.25);">
                <div style="font-size:48px; margin-bottom:12px;">⏰</div>
                <h2 style="margin:0 0 8px; color:#d32f2f;">Session Expired</h2>
                <p style="color:#555; margin:0 0 24px;">
                  Your 3-hour session has ended.<br>Please sign in again to continue.
                </p>
                <button onclick="location.reload()" style="
                    background:#d32f2f; color:#fff; border:none;
                    padding:10px 28px; border-radius:8px; font-size:15px;
                    cursor:pointer;">Sign In Again</button>
              </div>
            </div>

            <script>
            (function() {{
                var remaining = {remaining_secs};
                var wrap  = document.getElementById('session-timer-wrap');
                var timer = document.getElementById('session-timer');
                var icon  = document.getElementById('session-timer-icon');
                var modal = document.getElementById('session-expired-modal');

                function fmt(s) {{
                    var h = Math.floor(s / 3600);
                    var m = Math.floor((s % 3600) / 60);
                    var sec = s % 60;
                    return (h > 0 ? String(h).padStart(2,'0') + ':' : '')
                         + String(m).padStart(2,'0') + ':'
                         + String(sec).padStart(2,'0');
                }}

                function tick() {{
                    if (remaining <= 0) {{
                        timer.textContent = '00:00';
                        modal.style.display = 'flex';
                        return;
                    }}
                    timer.textContent = fmt(remaining);

                    // Style transitions
                    if (remaining <= 120) {{
                        // Red + pulse under 2 min
                        wrap.style.background  = '#fff0f0';
                        wrap.style.border      = '1px solid #ffaaaa';
                        wrap.style.color       = '#c62828';
                        icon.textContent       = '🔴';
                        wrap.style.animation   = 'pulse 1s infinite';
                    }} else if (remaining <= 300) {{
                        // Amber under 5 min
                        wrap.style.background  = '#fffbe6';
                        wrap.style.border      = '1px solid #ffe082';
                        wrap.style.color       = '#e65100';
                        icon.textContent       = '⚠️';
                        wrap.style.animation   = 'none';
                    }} else {{
                        wrap.style.background  = '#f0f4ff';
                        wrap.style.border      = '1px solid #c8d8ff';
                        wrap.style.color       = '#3a5bd9';
                        icon.textContent       = '⏱';
                        wrap.style.animation   = 'none';
                    }}

                    remaining--;
                    setTimeout(tick, 1000);
                }}

                // Inject pulse keyframes once
                if (!document.getElementById('pulse-style')) {{
                    var s = document.createElement('style');
                    s.id = 'pulse-style';
                    s.textContent = '@keyframes pulse {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:0.5 }} }}';
                    document.head.appendChild(s);
                }}

                tick();
            }})();
            </script>
            """,
            unsafe_allow_html=True,
        )

    # Initialize session state
    if 'show_tip' not in st.session_state:
        st.session_state.show_tip = False
        st.session_state.tip_text = ""
        st.session_state.tip_time = 0
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    if 'show_layout_info' not in st.session_state:
        st.session_state.show_layout_info = False
        st.session_state.layout_info_time = 0
    if 'show_theme_info' not in st.session_state:
        st.session_state.show_theme_info = False
        st.session_state.theme_info_time = 0

    pages = {
        "Home": home_page,
        "Daily Food Log": daily_food_log,
        "Nutrition Analysis": nutrition_analysis,
        "Food History": food_history,
        "Food Database Config": settings,
    }

    def on_page_change():
        st.session_state.page = st.session_state.page_selection

    current_index = list(pages.keys()).index(st.session_state.page)
    st.sidebar.radio(
        "Navigate",
        list(pages.keys()),
        index=current_index,
        key="page_selection",
        on_change=on_page_change,
    )

    # Daily Tip
    if st.sidebar.button("🌟 Daily Nutrition Tip"):
        nutrition_tips = [
            "Drink 8 glasses of water daily!",
            "Include protein in every meal.",
            "Eat more colorful vegetables.",
            "Balance your macronutrients.",
            "Avoid processed foods.",
        ]
        st.session_state.show_tip = True
        st.session_state.tip_text = random.choice(nutrition_tips)
        st.session_state.tip_time = time.time()

    if st.session_state.show_tip:
        elapsed_time = time.time() - st.session_state.tip_time
        if elapsed_time < 6:
            opacity = (
                min(1.0, elapsed_time / 0.3)
                if elapsed_time < 0.3
                else max(0, 6 - elapsed_time)
                if elapsed_time >= 5
                else 1.0
            )
            tip_html = f"""
            <div style="padding: 10px; background-color: #e8f4f8; border-left: 4px solid #4e8cff;
                        border-radius: 4px; margin: 0; opacity: {opacity}; transition: opacity 0.2s ease;">
                <p style="margin: 0; color: #1f618d;"><strong>Tip:</strong> {st.session_state.tip_text}</p>
            </div>
            """
            st.sidebar.markdown(tip_html, unsafe_allow_html=True)
            time.sleep(0.1)
            st.rerun()
        else:
            st.session_state.show_tip = False

    st.sidebar.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

    # Layout & Theme
    st.sidebar.markdown("<hr style='margin: 10px 0; border: 1px solid'>", unsafe_allow_html=True)
    st.sidebar.subheader("🖥️ Layout & Theme Info")

    if st.sidebar.button("ℹ️ Enable Wide Mode"):
        st.session_state.show_layout_info = True
        st.session_state.layout_info_time = time.time()

    if st.session_state.show_layout_info:
        elapsed_time = time.time() - st.session_state.layout_info_time
        if elapsed_time < 5:
            st.sidebar.info("To enable Wide Mode: \n\nClick '⋮' > Settings > Tick Wide Mode.")
            time.sleep(0.1)
            st.rerun()
        else:
            st.session_state.show_layout_info = False
            st.rerun()

    st.sidebar.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

    if st.sidebar.button("ℹ️ Change Theme"):
        st.session_state.show_theme_info = True
        st.session_state.theme_info_time = time.time()

    if st.session_state.show_theme_info:
        elapsed_time = time.time() - st.session_state.theme_info_time
        if elapsed_time < 5:
            st.sidebar.info("To change app theme: \n\nClick '⋮' > Settings > Pick Light/Dark.")
            time.sleep(0.1)
            st.rerun()
        else:
            st.session_state.show_theme_info = False
            st.rerun()

    # Logout
    st.sidebar.markdown("<hr style='margin: 10px 0; border: 1px solid'>", unsafe_allow_html=True)
    if st.sidebar.button("🚪 Sign Out", use_container_width=True):
        logout()

    # Render selected page
    pages[st.session_state.page]()


if __name__ == "__main__":
    main()
