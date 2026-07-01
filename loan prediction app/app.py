import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import pickle

# Initialize the Dash app
dash_app = dash.Dash(__name__, title="Loan Prediction App")
# Expose the Flask server for Vercel's @vercel/python builder
app = dash_app.server

# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Application Layout
dash_app.layout = html.Div(className="stApp", children=[
    html.H1("🏦 Loan Prediction App", className="animated-title"),
    html.P("Enter applicant details to instantly predict loan approval status.", className="subtitle"),
    
    html.Div(className="form-container", children=[
        html.H3("Applicant Details", style={"marginTop": 0}),
        html.Div(className="row", children=[
            html.Div(className="col", children=[
                html.Label("Gender"),
                dcc.Dropdown(id='gender', options=[{'label': i, 'value': i} for i in ["Male", "Female"]], value="Male", clearable=False),
                
                html.Label("Married"),
                dcc.Dropdown(id='married', options=[{'label': i, 'value': i} for i in ["Yes", "No"]], value="Yes", clearable=False),
                
                html.Label("Dependents"),
                dcc.Dropdown(id='dependents', options=[{'label': i, 'value': i} for i in ["0", "1", "2", "3+"]], value="0", clearable=False),
                
                html.Label("Education"),
                dcc.Dropdown(id='education', options=[{'label': i, 'value': i} for i in ["Graduate", "Not Graduate"]], value="Graduate", clearable=False),
                
                html.Label("Self Employed"),
                dcc.Dropdown(id='self_employed', options=[{'label': i, 'value': i} for i in ["Yes", "No"]], value="No", clearable=False),
            ]),
            html.Div(className="col", children=[
                html.Label("Property Area"),
                dcc.Dropdown(id='property_area', options=[{'label': i, 'value': i} for i in ["Urban", "Semiurban", "Rural"]], value="Urban", clearable=False),
                
                html.Label("Credit History"),
                dcc.Dropdown(id='credit_history', options=[{'label': "Good (1.0)", 'value': 1.0}, {'label': "Bad (0.0)", 'value': 0.0}], value=1.0, clearable=False),
                
                html.Label("Applicant Income"),
                dcc.Input(id='applicant_income', type='number', value=5000, min=0, step=100, style={'width': '100%'}),
                
                html.Label("Coapplicant Income"),
                dcc.Input(id='coapplicant_income', type='number', value=0.0, min=0.0, step=100.0, style={'width': '100%'}),
                
                html.Label("Loan Amount (in thousands)"),
                dcc.Input(id='loan_amount', type='number', value=120.0, min=1.0, step=10.0, style={'width': '100%'}),
                
                html.Label("Loan Amount Term (in days)"),
                dcc.Input(id='loan_amount_term', type='number', value=360.0, min=1.0, step=10.0, style={'width': '100%'}),
            ])
        ]),
        html.Hr(style={"marginTop": "30px", "borderColor": "rgba(255,255,255,0.4)"}),
        html.Button("Predict Loan Status", id="submit-btn", n_clicks=0, className="submit-btn"),
        html.Div(id="output-div")
    ])
])

@dash_app.callback(
    Output("output-div", "children"),
    Input("submit-btn", "n_clicks"),
    State("gender", "value"),
    State("married", "value"),
    State("dependents", "value"),
    State("education", "value"),
    State("self_employed", "value"),
    State("property_area", "value"),
    State("credit_history", "value"),
    State("applicant_income", "value"),
    State("coapplicant_income", "value"),
    State("loan_amount", "value"),
    State("loan_amount_term", "value")
)
def predict(n_clicks, gender, married, dependents, education, self_employed, property_area, credit_history, applicant_income, coapplicant_income, loan_amount, loan_amount_term):
    if n_clicks == 0:
        return ""
        
    if model is None:
        return html.Div("Error: Model file 'model.pkl' could not be loaded.", className="error-box")
        
    try:
        # Preprocessing matching the original logic
        total_income = applicant_income + coapplicant_income
        
        # Log transforms (safe checks)
        app_income_log = np.log(applicant_income) if applicant_income and applicant_income > 0 else 0
        loan_amount_log = np.log(loan_amount) if loan_amount and loan_amount > 0 else 0
        loan_amount_term_log = np.log(loan_amount_term) if loan_amount_term and loan_amount_term > 0 else 0
        total_income_log = np.log(total_income) if total_income and total_income > 0 else 0
        
        # Boolean mapping
        male = True if gender == "Male" else False
        married_yes = True if married == "Yes" else False
        dep_1 = True if dependents == "1" else False
        dep_2 = True if dependents == "2" else False
        dep_3 = True if dependents == "3+" else False
        not_graduate = True if education == "Not Graduate" else False
        self_emp_yes = True if self_employed == "Yes" else False
        semiurban = True if property_area == "Semiurban" else False
        urban = True if property_area == "Urban" else False
        
        input_df = pd.DataFrame([[
            credit_history, app_income_log, loan_amount_log, loan_amount_term_log, total_income_log, 
            male, married_yes, dep_1, dep_2, dep_3, not_graduate, self_emp_yes, semiurban, urban
        ]], columns=[
            'Credit_History', 'ApplicantIncomeLog', 'LoanAmountLog', 'Loan_Amount_Term_Log', 'Total_Income_Log', 
            'Gender_Male', 'Married_Yes', 'Dependents_1', 'Dependents_2', 'Dependents_3+', 'Education_Not Graduate', 
            'Self_Employed_Yes', 'Property_Area_Semiurban', 'Property_Area_Urban'
        ])
        
        # Run prediction
        prediction = model.predict(input_df)[0]
        
        # Format results
        if prediction == "Y":
            return html.Div([
                html.Span("🎉 Congratulations! Your loan is likely to be "), html.B("Approved"), html.Span(".")
            ], className="success-box")
        else:
            return html.Div([
                html.Span("⚠️ We're sorry, your loan is likely to be "), html.B("Rejected"), html.Span(".")
            ], className="error-box")
            
    except Exception as e:
        return html.Div(f"An error occurred during prediction: {str(e)}", className="error-box")

if __name__ == '__main__':
    dash_app.run(debug=True)
