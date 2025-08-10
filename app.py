import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained models and preprocessors
@st.cache_resource
def load_models():
    try:
        with open('all_models.pkl', 'rb') as f:
            models = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('model_encoder.pkl', 'rb') as f:
            model_encoder = pickle.load(f)
        with open('model_results.pkl', 'rb') as f:
            results = pickle.load(f)
        return models, preprocessor, model_encoder, results
    except FileNotFoundError as e:
        return None, None, None, None

def predict_price(model, preprocessor, model_encoder, input_data):
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Encode model
    try:
        df['model'] = model_encoder.transform([df['model'].iloc[0]])
    except ValueError:
        st.error("Model name not found in training data. Please select a valid model.")
        return None
    
    # Transform features
    X_transformed = preprocessor.transform(df)
    
    # Predict
    prediction = model.predict(X_transformed)
    return prediction[0]

def main():
    st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")
    
    st.title("🚗 Car Price Prediction System")
    st.write("Enter car details and select a machine learning algorithm to predict the selling price")
    
    # Load models
    models, preprocessor, model_encoder, results = load_models()
    
    if models is None:
        st.error("❌ Model files not found. Please run `car_price_model.py` first to train the models.")
        st.info("Run the following command in your terminal: `python car_price_model.py`")
        return
    
    # Sidebar for model selection and performance
    with st.sidebar:
        st.header("🤖 Model Selection")
        
        # Model selection
        selected_model = st.selectbox(
            "Choose Algorithm:",
            list(models.keys()),
            help="Select the machine learning algorithm for prediction"
        )
        
        # Display model performance if available
        if results and selected_model in results:
            st.subheader("📊 Model Performance")
            metrics = results[selected_model]
            if 'test_r2' in metrics:
                st.metric("R² Score", f"{metrics['test_r2']:.4f}")
            if 'test_rmse' in metrics:
                st.metric("RMSE", f"₹{metrics['test_rmse']:,.0f}")
            if 'test_mae' in metrics:
                st.metric("MAE", f"₹{metrics['test_mae']:,.0f}")
        
        # Model comparison
        if st.button("🔍 Compare All Models"):
            st.subheader("Model Comparison")
            if results:
                comparison_data = []
                for name, metrics in results.items():
                    if 'test_r2' in metrics:
                        comparison_data.append({
                            'Model': name.replace(' Regressor', ''),
                            'R² Score': f"{metrics['test_r2']:.4f}",
                            'RMSE': f"{metrics['test_rmse']:,.0f}"
                        })
                
                df_comparison = pd.DataFrame(comparison_data)
                df_comparison = df_comparison.sort_values('R² Score', ascending=False)
                st.dataframe(df_comparison, hide_index=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔧 Car Details Input")
        
        # Create input form
        with st.form("car_details"):
            input_col1, input_col2 = st.columns(2)
            
            with input_col1:
                model_name = st.selectbox(
                    "Car Model *", 
                    ['Alto', 'Grand', 'i20', 'Ecosport', 'Wagon R', 'i10', 'Venue', 'Swift', 
                     'Verna', 'Duster', 'Cooper', 'Ciaz', 'C-Class', 'Innova', 'Baleno', 
                     'Swift Dzire', 'Vento', 'Creta', 'City', 'Bolero', 'Fortuner', 'KWID', 
                     'Amaze', 'Santro', 'XUV500', 'KUV100', 'Ignis', 'RediGO', 'Scorpio', 
                     'Marazzo', 'Aspire', 'Figo', 'Vitara', 'Tiago', 'Polo', 'Seltos', 
                     'Celerio', 'GO', '5', 'CR-V', 'Endeavour', 'KUV', 'Jazz', '3', 'A4', 
                     'Tigor', 'Ertiga', 'Safari', 'Thar', 'Hexa', 'Rover', 'Eeco', 'A6', 
                     'E-Class', 'Q7', 'Z4', '6', 'XF', 'X5', 'Hector', 'Civic', 'D-Max', 
                     'Cayenne', 'X1', 'Rapid', 'Freestyle', 'Superb', 'Nexon', 'XUV300', 
                     'Dzire VXI', 'S90', 'WR-V', 'XL6', 'Triber', 'ES', 'Wrangler', 'Camry', 
                     'Elantra', 'Yaris', 'GL-Class', '7', 'S-Presso', 'Dzire LXI', 'Aura', 
                     'XC', 'Ghibli', 'Continental', 'CR', 'Kicks', 'S-Class', 'Tucson', 
                     'Harrier', 'X3', 'Octavia', 'Compass', 'CLS', 'redi-GO', 'Glanza', 
                     'Macan', 'X4', 'Dzire ZXI', 'XC90', 'F-PACE', 'A8', 'MUX', 'GTC4Lusso', 
                     'GLS', 'X-Trail', 'XE', 'XC60', 'Panamera', 'Alturas', 'Altroz', 'NX', 
                     'Carnival', 'C', 'RX', 'Ghost', 'Quattroporte', 'Gurkha'],
                    help="Select the car model"
                )
                
                vehicle_age = st.number_input(
                    "Vehicle Age (years) *", 
                    min_value=0, max_value=30, value=5,
                    help="Age of the vehicle in years"
                )
                
                km_driven = st.number_input(
                    "Kilometers Driven *", 
                    min_value=0, max_value=500000, value=50000, step=1000,
                    help="Total kilometers driven"
                )
                
                seller_type = st.selectbox(
                    "Seller Type *", 
                    ['Individual', 'Dealer', 'Trustmark Dealer'],
                    help="Type of seller"
                )
                
                fuel_type = st.selectbox(
                    "Fuel Type *", 
                    ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'],
                    help="Primary fuel type"
                )
            
            with input_col2:
                transmission_type = st.selectbox(
                    "Transmission *", 
                    ['Manual', 'Automatic'],
                    help="Transmission type"
                )
                
                mileage = st.number_input(
                    "Mileage (km/l) *", 
                    min_value=5.0, max_value=40.0, value=18.0, step=0.1,
                    help="Fuel efficiency in km/l"
                )
                
                engine = st.number_input(
                    "Engine Capacity (CC) *", 
                    min_value=500, max_value=5000, value=1200, step=50,
                    help="Engine capacity in cubic centimeters"
                )
                
                max_power = st.number_input(
                    "Max Power (bhp) *", 
                    min_value=30.0, max_value=800.0, value=80.0, step=1.0,
                    help="Maximum power output in brake horsepower"
                )
                
                seats = st.selectbox(
                    "Number of Seats *", 
                    [4, 5, 6, 7, 8, 9, 10],
                    index=1,
                    help="Seating capacity"
                )
            
            submitted = st.form_submit_button(
                "🔮 Predict Price", 
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                # Create input dictionary
                input_data = {
                    'model': model_name,
                    'vehicle_age': vehicle_age,
                    'km_driven': km_driven,
                    'seller_type': seller_type,
                    'fuel_type': fuel_type,
                    'transmission_type': transmission_type,
                    'mileage': mileage,
                    'engine': engine,
                    'max_power': max_power,
                    'seats': seats
                }
                
                try:
                    with st.spinner(f'Predicting price using {selected_model}...'):
                        predicted_price = predict_price(
                            models[selected_model], 
                            preprocessor, 
                            model_encoder, 
                            input_data
                        )
                    
                    if predicted_price is not None:
                        # Display result in the second column
                        with col2:
                            st.header("💰 Prediction Result")
                            st.success("✅ Price Prediction Completed!")
                            
                            # Main price display
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(90deg, #4CAF50, #45a049);
                                padding: 20px;
                                border-radius: 10px;
                                text-align: center;
                                color: white;
                                margin: 10px 0;
                            ">
                                <h2 style="margin: 0; color: white;">Predicted Price</h2>
                                <h1 style="margin: 10px 0; color: white;">₹{predicted_price:,.0f}</h1>
                                <p style="margin: 0; opacity: 0.9;">Using {selected_model}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Price range estimate
                            lower_bound = predicted_price * 0.9
                            upper_bound = predicted_price * 1.1
                            st.info(f"📊 **Estimated Range:** ₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}")
                            
                            # Show input summary
                            with st.expander("📋 Input Summary", expanded=False):
                                for key, value in input_data.items():
                                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        
                        # Additional insights below the form
                        st.header("📈 Additional Insights")
                        
                        insight_col1, insight_col2, insight_col3 = st.columns(3)
                        
                        with insight_col1:
                            # Age impact
                            age_impact = "High" if vehicle_age > 10 else "Medium" if vehicle_age > 5 else "Low"
                            st.metric("Age Impact", age_impact, f"{vehicle_age} years")
                        
                        with insight_col2:
                            # Usage impact
                            usage_impact = "High" if km_driven > 100000 else "Medium" if km_driven > 50000 else "Low"
                            st.metric("Usage Impact", usage_impact, f"{km_driven:,} km")
                        
                        with insight_col3:
                            # Engine performance
                            performance = "High" if max_power > 150 else "Medium" if max_power > 80 else "Economy"
                            st.metric("Performance Class", performance, f"{max_power} bhp")
                        
                except Exception as e:
                    st.error(f"❌ Error in prediction: {str(e)}")
                    st.error("Please check your inputs and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** Predictions are estimates based on historical data and may vary from actual market prices.")

if __name__ == "__main__":
    main()