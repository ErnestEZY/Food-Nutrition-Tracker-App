import re
import streamlit as st
import bcrypt
from datetime import datetime
from database import users_collection


# ── helpers ──────────────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _check_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def _get_user(email: str):
    return users_collection.find_one({"email": email.lower().strip()})


def _validate_password(password: str) -> str | None:
    """
    Returns an error message string if invalid, or None if valid.
    Rules: 8–12 chars, at least 1 uppercase, 1 lowercase, 1 digit, 1 special char.
    """
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if len(password) > 12:
        return "Password must be at most 12 characters."
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least 1 uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must contain at least 1 lowercase letter."
    if not re.search(r"\d", password):
        return "Password must contain at least 1 number."
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?`~]", password):
        return "Password must contain at least 1 special character. (!, @, #, $, %, ^, *, &)"
    return None


def _validate_email(email: str) -> str | None:
    """Basic email format check."""
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    if not re.match(pattern, email):
        return "Please enter a valid email address."
    return None


def _create_user(email: str, username: str, password: str) -> bool:
    """Insert a new user. Returns False if email already exists."""
    email = email.lower().strip()
    if _get_user(email):
        return False
    users_collection.insert_one({
        "email": email,
        "username": username.strip(),
        "password": _hash_password(password),
        "created_at": datetime.utcnow(),
    })
    return True


# ── session helpers ───────────────────────────────────────────────────────────

def is_logged_in() -> bool:
    return st.session_state.get("authenticated", False)


def current_user() -> str:
    """Return the logged-in email (used as unique user key), or empty string."""
    return st.session_state.get("email", "")


def current_name() -> str:
    """Return the logged-in display username."""
    return st.session_state.get("username", "")


def logout():
    for key in ("authenticated", "email", "username"):
        st.session_state.pop(key, None)
    st.rerun()


# ── UI ────────────────────────────────────────────────────────────────────────

def auth_page():
    """
    Renders the Sign In / Sign Up page.
    Returns True once the user is authenticated so the caller can proceed.
    """
    if is_logged_in():
        return True

    st.title("🥗 Nutrition Tracker Pro")

    tab_signin, tab_signup = st.tabs(["Sign In", "Sign Up"])

    # ── Sign In tab ───────────────────────────────────────────────────────────
    with tab_signin:
        st.subheader("Welcome back!")
        with st.form("login_form"):
            email    = st.text_input("Email", max_chars=50, placeholder="you@example.com")
            password = st.text_input("Password", type="password", max_chars=12, placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please fill in all fields.")
            else:
                email_error = _validate_email(email)
                if email_error:
                    st.error(email_error)
                else:
                    user = _get_user(email)
                    if user is None:
                        st.error("Unknown email. Please check your email or sign up for an account.")
                    elif not _check_password(password, user["password"]):
                        st.error("Incorrect password. Please try again.")
                    else:
                        st.session_state.authenticated = True
                        st.session_state.email    = user["email"]
                        st.session_state.username = user["username"]
                        st.rerun()

        st.markdown(
            "<div style='text-align: center;'>No account yet? Switch to the <b>Sign Up</b> tab above to create one.</div>",
            unsafe_allow_html=True,
        )

    # ── Sign Up tab ───────────────────────────────────────────────────────────
    with tab_signup:
        st.subheader("Create an account")
        with st.form("register_form"):
            new_username = st.text_input("Username", max_chars=30,
                                         placeholder="Enter your name")
            new_email    = st.text_input("Email", max_chars=50,
                                         placeholder="you@example.com")
            new_password = st.text_input(
                "Password", type="password", max_chars=12,
                help="Format: 8–12 characters · uppercase · lowercase · number · special character", placeholder="Create your password"
            )
            confirm_password = st.text_input("Confirm Password", type="password", max_chars=12,
                                             placeholder="Re-enter your password",
                                             help="Re-enter your password to confirm. Make sure it matches the one above.")
            reg_submitted = st.form_submit_button("Sign Up", use_container_width=True)

        if reg_submitted:
            if not all([new_username, new_email, new_password, confirm_password]):
                st.error("Please fill in all fields.")
            elif len(new_username.strip()) < 3:
                st.error("Username must be at least 3 characters.")
            else:
                email_error = _validate_email(new_email)
                if email_error:
                    st.error(email_error)
                else:
                    pw_error = _validate_password(new_password)
                    if pw_error:
                        st.error(pw_error)
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        if _create_user(new_email, new_username, new_password):
                            st.success(
                                f"Account created! Switch to **Sign In** and log in with "
                                f"**{new_email.lower().strip()}**."
                            )
                        else:
                            st.error("An account with that email already exists.")

        st.markdown(
            "<div style='text-align: center;'>Already have an account? Switch to the <b>Sign In</b> tab above.</div>",
            unsafe_allow_html=True,
        )

    return False
