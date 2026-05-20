import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Late Delivery Prediction App",
    page_icon="🚚",
    layout="wide"
)

# -------------------------------
# Login credentials
# -------------------------------
USER_ID = "admin"
PASSWORD = "admin123"


# -------------------------------
# Login page
# -------------------------------
def login():
    st.title("🔐 Late Delivery Prediction Login")

    user_id = st.text_input("User ID")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if user_id == USER_ID and password == PASSWORD:
            st.session_state["logged_in"] = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid User ID or Password")


# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("DataCoSupplyChainDataset.zip", encoding="latin1", compression="zip")
    return df


# -------------------------------
# Train Logistic Regression model
# -------------------------------
@st.cache_resource
def train_model(df):
    # Target column
    target_col = "Late_delivery_risk"

    # Features used for prediction
    feature_cols = [
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Benefit per order",
        "Sales per customer",
        "Order Item Quantity",
        "Order Item Discount",
        "Order Item Product Price",
        "Product Price"
    ]

    # Keep only required columns
    model_df = df[feature_cols + [target_col]].copy()

    # Drop missing values
    model_df = model_df.dropna()

    # Input and target
    X = model_df[feature_cols]
    y = model_df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Prediction on test data
    y_pred = model.predict(X_test_scaled)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, scaler, feature_cols, accuracy, cm, report


# -------------------------------
# Prediction dashboard
# -------------------------------
def dashboard():
    st.title("🚚 Late Delivery Risk Prediction App")
    st.write(
        "This web application uses Logistic Regression to predict whether an order has late delivery risk."
    )

    df = load_data()

    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])

    with st.expander("View Dataset Sample"):
        st.dataframe(df.head(100))

    # Train model
    model, scaler, feature_cols, accuracy, cm, report = train_model(df)

    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Used", "Logistic Regression")
    col2.metric("Target Column", "Late_delivery_risk")
    col3.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted 0", "Predicted 1"],
        index=["Actual 0", "Actual 1"]
    )

    st.dataframe(cm_df)

    st.subheader("Classification Report")

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.subheader("Predict Late Delivery Risk")

    st.write("Enter order details below:")

    col1, col2 = st.columns(2)

    with col1:
        days_shipping_real = st.number_input(
            "Days for shipping (real)",
            min_value=0,
            value=3
        )

        days_shipment_scheduled = st.number_input(
            "Days for shipment (scheduled)",
            min_value=0,
            value=4
        )

        benefit_per_order = st.number_input(
            "Benefit per order",
            value=50.0
        )

        sales_per_customer = st.number_input(
            "Sales per customer",
            min_value=0.0,
            value=200.0
        )

    with col2:
        order_item_quantity = st.number_input(
            "Order Item Quantity",
            min_value=1,
            value=1
        )

        order_item_discount = st.number_input(
            "Order Item Discount",
            min_value=0.0,
            value=0.0
        )

        order_item_product_price = st.number_input(
            "Order Item Product Price",
            min_value=0.0,
            value=100.0
        )

        product_price = st.number_input(
            "Product Price",
            min_value=0.0,
            value=100.0
        )

    if st.button("Predict"):
        input_data = pd.DataFrame(
            [[
                days_shipping_real,
                days_shipment_scheduled,
                benefit_per_order,
                sales_per_customer,
                order_item_quantity,
                order_item_discount,
                order_item_product_price,
                product_price
            ]],
            columns=feature_cols
        )

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"Prediction: Late Delivery Risk | Probability: {probability:.2f}")
        else:
            st.success(f"Prediction: No Late Delivery Risk | Probability: {probability:.2f}")

    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()


# -------------------------------
# App execution
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    dashboard()
else:
    login()