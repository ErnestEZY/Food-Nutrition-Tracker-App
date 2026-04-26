import streamlit as st
import bcrypt
from datetime import datetime
from database import users_collection


# ── helpers ──────────────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _check_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def _get_user(username: str):
    return users_collection.find_one({"username": username.lower().strip()})


def _create_user(username: str, name: str, password: str) -> bool:
    """Insert a new user. Returns False if username already exists."""
    username = username.lower().strip()
    if _get_user(username):
        return False
    users_collection.insert_one({
        "username": username,
        "name": name.strip(),
        "password": _hash_password(password),
        "created_at": datetime.utcnow(),
    })
    return True


# ── session helpers ───────────────────────────────────────────────────────────

def is_logged_in() -> bool:
    return st.session_state.get("authenticated", False)


def current_user() -> str:
    """Return the logged-in username, or empty string."""
    return st.session_state.get("username", "")


def current_name() -> str:
    return st.session_state.get("name", "")


def logout():
    for key in ("authenticated", "username", "name"):
        st.session_state.pop(key, None)
    st.rerun()


# ── UI ────────────────────────────────────────────────────────────────────────

def auth_page():
    """
    Renders the login / register page.
    Returns True once the user is authenticated so the caller can proceed.
    """
    if is_logged_in():
        return True

    st.title("🥗 Nutrition Tracker Pro")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    # ── Login tab ─────────────────────────────────────────────────────────────
    with tab_login:
        st.subheader("Welcome back!")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Please fill in all fields.")
            else:
                user = _get_user(username)
                if user and _check_password(password, user["password"]):
                    st.session_state.authenticated = True
                    st.session_state.username = user["username"]
                    st.session_state.name = user["name"]
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    # ── Register tab ──────────────────────────────────────────────────────────
    with tab_register:
        st.subheader("Create an account")
        with st.form("register_form"):
            new_name = st.text_input("Full Name")
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            reg_submitted = st.form_submit_button("Register", use_container_width=True)

        if reg_submitted:
            if not all([new_name, new_username, new_password, confirm_password]):
                st.error("Please fill in all fields.")
            elif len(new_username.strip()) < 3:
                st.error("Username must be at least 3 characters.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                if _create_user(new_username, new_name, new_password):
                    st.success(f"Account created! You can now log in as **{new_username.lower().strip()}**.")
                else:
                    st.error("Username already taken. Please choose another.")

    return False
