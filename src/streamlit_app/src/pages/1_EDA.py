import os

import streamlit as st

st.set_page_config(layout='wide', page_title='EDA Report', page_icon='📊')
st.title('📊 Exploratory Data Analysis')

html_path = os.path.join(os.path.dirname(__file__), 'eda.html')

if os.path.exists(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read().replace('"', '&quot;')  # Escape quotes

    # Embed HTML in full screen height iframe
    iframe_html = f"""
    <iframe
        srcdoc="{html_content}"
        width="100%"
        height="100%"
        style="position:absolute; top:0; left:0; bottom:0; right:0; width:100%; height:100vh; border:none; margin:0; padding:0; overflow:hidden; z-index:999999;">
    </iframe>
    """
    # height=0 disables Streamlit's default iframe sizing
    st.components.v1.html(iframe_html, height=900, scrolling=False)
else:
    st.error(f'❌ Could not find "{html_path}".')
