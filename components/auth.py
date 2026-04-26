import re
import secrets
import streamlit as st
import bcrypt
from datetime import datetime, timezone, timedelta
from database import users_collection, db

# ── constants ─────────────────────────────────────────────────────────────────

SESSION_DURATION    = timedelta(hours=3)
sessions_collection = db["sessions"] if db is not None else None


# ── helpers ───────────────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _check_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def _get_user(email: str):
    return users_collection.find_one({"email": email.lower().strip()})


def _validate_password(password: str) -> str | None:
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
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return "Please enter a valid email address."
    return None


def _create_user(email: str, username: str, password: str) -> bool:
    email = email.lower().strip()
    if _get_user(email):
        return False
    users_collection.insert_one({
        "email":      email,
        "username":   username.strip(),
        "password":   _hash_password(password),
        "created_at": datetime.utcnow(),
    })
    return True


# ── session token helpers ─────────────────────────────────────────────────────

def _create_session_token(email: str, username: str) -> str:
    sessions_collection.delete_many({"email": email})
    token      = secrets.token_urlsafe(32)
    login_time = datetime.now(timezone.utc)
    sessions_collection.insert_one({
        "token":      token,
        "email":      email,
        "username":   username,
        "login_time": login_time,
        "expires_at": login_time + SESSION_DURATION,
    })
    return token


def _get_session(token: str):
    if not token:
        return None
    return sessions_collection.find_one({
        "token":      token,
        "expires_at": {"$gt": datetime.now(timezone.utc)},
    })


def _delete_session(token: str):
    if token:
        sessions_collection.delete_one({"token": token})


# ── session state helpers ─────────────────────────────────────────────────────

def _restore_session():
    if st.session_state.get("authenticated"):
        return
    token = st.query_params.get("token")
    if not token:
        return
    session = _get_session(token)
    if not session:
        st.query_params.clear()
        return
    login_time = session["login_time"]
    if login_time.tzinfo is None:
        login_time = login_time.replace(tzinfo=timezone.utc)
    st.session_state.authenticated = True
    st.session_state.email         = session["email"]
    st.session_state.username      = session["username"]
    st.session_state.login_time    = login_time
    st.session_state.session_token = token


def is_logged_in() -> bool:
    _restore_session()
    if not st.session_state.get("authenticated", False):
        return False
    login_time = st.session_state.get("login_time")
    if login_time is None:
        return False
    if login_time.tzinfo is None:
        login_time = login_time.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) - login_time > SESSION_DURATION:
        _clear_session()
        st.warning("Your session has expired. Please sign in again.")
        return False
    return True


def _clear_session():
    token = st.session_state.get("session_token")
    _delete_session(token)
    st.query_params.clear()
    for key in ("authenticated", "email", "username", "login_time",
                "session_token", "bmi_loaded"):
        st.session_state.pop(key, None)


def current_user() -> str:
    return st.session_state.get("email", "")


def current_name() -> str:
    return st.session_state.get("username", "")


def logout():
    _clear_session()
    st.rerun()


# ── UI ────────────────────────────────────────────────────────────────────────

def auth_page():
    if is_logged_in():
        return True

    st.title("🥗 Nutrition Tracker Pro")

    tab_signin, tab_signup = st.tabs(["Sign In", "Sign Up"])

    # ── Sign In ───────────────────────────────────────────────────────────────
    with tab_signin:
        st.subheader("Welcome back!")

        # Mask the password field using CSS -webkit-text-security (type="text"
        # so Google password manager never triggers) and block spaces via JS.
        st.markdown(
            """
            <style>
            /* Visually mask the password text input with bullet dots */
            input[placeholder="Enter your password"] {
                -webkit-text-security: disc !important;
                text-security: disc !important;
            }
            </style>
            <script>
            (function() {
                function patch() {
                    var doc = window.parent.document;
                    var inputs = doc.querySelectorAll(
                        'input[placeholder="Enter your password"]'
                    );
                    inputs.forEach(function(el) {
                        if (el.dataset.patched) return;
                        el.dataset.patched = '1';
                        el.setAttribute('autocomplete', 'off');
                        // Block spaces
                        el.addEventListener('keydown', function(e) {
                            if (e.key === ' ') e.preventDefault();
                        });
                        el.addEventListener('input', function() {
                            var pos = el.selectionStart;
                            var clean = el.value.replace(/ /g, '');
                            if (clean !== el.value) {
                                el.value = clean;
                                el.selectionStart = el.selectionEnd = Math.max(0, pos - 1);
                                el.dispatchEvent(new Event('input', {bubbles: true}));
                            }
                        });
                    });
                }
                patch();
                new MutationObserver(patch).observe(
                    window.parent.document.body,
                    {childList: true, subtree: true}
                );
            })();
            </script>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            email    = st.text_input("Email", max_chars=50,
                                     placeholder="you@example.com")
            # type="text" — Google password manager only triggers on type="password"
            password = st.text_input("Password", max_chars=12,
                                     placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            password = (password or "").replace(" ", "")
            if not email or not password:
                st.error("Please fill in all fields.")
            else:
                err = _validate_email(email)
                if err:
                    st.error(err)
                else:
                    user = _get_user(email)
                    if user is None:
                        st.error("Unknown email. Please check your email or sign up for an account.")
                    elif not _check_password(password, user["password"]):
                        st.error("Incorrect password. Please try again.")
                    else:
                        token      = _create_session_token(user["email"], user["username"])
                        login_time = datetime.now(timezone.utc)
                        st.session_state.authenticated = True
                        st.session_state.email         = user["email"]
                        st.session_state.username      = user["username"]
                        st.session_state.login_time    = login_time
                        st.session_state.session_token = token
                        st.query_params["token"]       = token
                        st.rerun()

        st.markdown(
            "<div style='text-align:center;'>No account yet? "
            "Switch to the <b>Sign Up</b> tab above to create one.</div>",
            unsafe_allow_html=True,
        )

    # ── Sign Up ───────────────────────────────────────────────────────────────
    with tab_signup:
        st.subheader("Create an account")
        with st.form("register_form"):
            new_username = st.text_input("Username", max_chars=30,
                                         placeholder="Enter your name")
            new_email    = st.text_input("Email", max_chars=50,
                                         placeholder="you@example.com")
            new_password = st.text_input(
                "Password", type="password", max_chars=12,
                help="Format: 8-12 characters · uppercase · lowercase · number · special character",
                placeholder="Create your password",
            )
            confirm_password = st.text_input(
                "Confirm Password", type="password", max_chars=12,
                placeholder="Re-enter your password",
                help="Re-enter your password to confirm. Make sure it matches the one above.",
            )
            reg_submitted = st.form_submit_button("Sign Up", use_container_width=True)

        if reg_submitted:
            if not all([new_username, new_email, new_password, confirm_password]):
                st.error("Please fill in all fields.")
            elif len(new_username.strip()) < 3:
                st.error("Username must be at least 3 characters.")
            else:
                err = _validate_email(new_email)
                if err:
                    st.error(err)
                else:
                    pw_err = _validate_password(new_password)
                    if pw_err:
                        st.error(pw_err)
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
            "<div style='text-align:center;'>Already have an account? "
            "Switch to the <b>Sign In</b> tab above.</div>",
            unsafe_allow_html=True,
        )

    return False
