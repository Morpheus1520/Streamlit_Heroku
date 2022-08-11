import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open(
    r"C:\Users\Huzefa\Machine Learning Jupyter notebook\End-To-End Projects\Big_mart Sales Prediction\xgboost_trained.sav",
    "rb"))

scaler = pickle.load(open(
    r"C:\Users\Huzefa\Machine Learning Jupyter notebook\End-To-End Projects\Big_mart Sales Prediction\data_scaler.pkl",
    "rb"))

# Lets load all the encoders:
enc1 = pickle.load(open(
    r"C:\Users\Huzefa\Machine Learning Jupyter notebook\End-To-End Projects\Big_mart Sales Prediction\Item_Identifier_encoder.pkl",
    "rb"))

enc2 = pickle.load(open(
    r"C:\Users\Huzefa\Machine Learning Jupyter notebook\End-To-End Projects\Big_mart Sales Prediction\Item_Fat_Content_encoder.pkl",
    "rb"))

enc3 = pickle.load(open(
    r"C:\Users\Huzefa\Machine Learning Jupyter notebook\End-To-End Projects\Big_mart Sales Prediction\Item_Type_encoder.pkl",
    "rb"))

enc4 = pickle.load(open(
    r"C:\Users\Huzefa\Machine Learning Jupyter notebook\End-To-End Projects\Big_mart Sales Prediction\Outlet_Identifier_encoder.pkl",
    "rb"))

enc5 = pickle.load(open(
    r"C:\Users\Huzefa\Machine Learning Jupyter notebook\End-To-End Projects\Big_mart Sales Prediction\Outlet_Size_encoder.pkl",
    "rb"))

enc6 = pickle.load(open(
    r"C:\Users\Huzefa\Machine Learning Jupyter notebook\End-To-End Projects\Big_mart Sales Prediction\Outlet_Location_Type_encoder.pkl",
    "rb"))

enc7 = pickle.load(open(
    r"C:\Users\Huzefa\Machine Learning Jupyter notebook\End-To-End Projects\Big_mart Sales Prediction\Outlet_Type_encoder.pkl",
    "rb"))


# Creating a function for prediction


def sales_prediction(input_data):
    print(input_data)

    input_data[0] = item_identifier = enc1.transform(np.asarray(input_data[0]).reshape(1, -1))[0]
    input_data[2] = enc2.transform(np.asarray(input_data[2]).reshape(1, -1))[0]
    input_data[4] = enc3.transform(np.asarray(input_data[4]).reshape(1, -1))[0]
    input_data[6] = enc4.transform(np.asarray(input_data[6]).reshape(1, -1))[0]
    input_data[8] = enc5.transform(np.asarray(input_data[8]).reshape(1, -1))[0]
    input_data[9] = enc6.transform(np.asarray(input_data[9]).reshape(1, -1))[0]
    input_data[10] = enc7.transform(np.asarray(input_data[10]).reshape(1, -1))[0]

    input_data_as_nparray = np.asarray(input_data)
    input_data_reshaped = input_data_as_nparray.reshape(1, -1)
    final_input_data = scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(final_input_data)

    return prediction


def main():
    # Giving a name
    st.title("Sales Prediction Web App")

    col1, col2, col3 = st.columns(3)

    with col1:
        item_identifier = st.text_input("Item Identifier Code")

    with col2:
        item_Weight = st.text_input("Weight of the item")

    with col3:
        item_fat = st.selectbox("Fat content of the item", ("Low Fat", "Regular"))

    with col1:
        Item_Visibility = st.text_input("Visibility of the item")

    with col2:
        Item_Type = st.selectbox("Type of the item", ('Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables',
                                                      'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods',
                                                      'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned',
                                                      'Breads', 'Starchy Foods', 'Others', 'Seafood'))

    with col3:
        Item_MRP = st.text_input("Item's MRP")

    with col1:
        Outlet_Identifier = st.selectbox("Outlet Identifier for the item",
                                         ('OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027', 'OUT045',
                                          'OUT017', 'OUT046', 'OUT035', 'OUT019'))

    with col2:
        Outlet_Establishment_Year = st.text_input("Establishment year of the outlet")

    with col3:
        Outlet_Size = st.selectbox("Size of the outlet", ('Medium', 'Small', 'High'))

    with col1:
        Outlet_Location_Type = st.selectbox("Location Type of the outlet", ('Tier 1', 'Tier 3', 'Tier 2'))

    with col2:
        Outlet_Type = st.selectbox("Type of the outlet", ('Grocery Store', 'Supermarket Type1', 'Supermarket Type2',
                                                          'Supermarket Type3'))

    # Prediction code

    final_sales_prediction = ""

    # Creating a button
    with col2:
        if st.button("Sales Prediction Result"):
            final_sales_prediction = sales_prediction(
                [item_identifier, item_Weight, item_fat, Item_Visibility, Item_Type, Item_MRP, Outlet_Identifier,
                 Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type])

    st.success(final_sales_prediction)


if __name__ == '__main__':
    main()
