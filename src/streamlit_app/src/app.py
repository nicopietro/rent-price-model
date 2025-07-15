import streamlit as st

st.set_page_config(page_title='Animal ML App', page_icon='🐾')

pg = st.navigation(
    [
        st.Page('pages/0_Home.py', title='Home', icon='🏠'),
        st.Page('pages/1_EDA.py', title='EDA', icon='📈'),
        st.Page('pages/2_Models.py', title='Models', icon='📊'),
        st.Page('pages/3_Predict.py', title='Predict', icon='🧠'),
    ]
)
pg.run()
