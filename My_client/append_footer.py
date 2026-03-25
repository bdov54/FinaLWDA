import pathlib
p = pathlib.Path('d:/airport_dashboard/app.py')
with p.open('a', encoding='utf-8') as f:
    f.write("\n# footer\nst.markdown(\"<div style='text-align:center; padding:15px 0; font-size:0.8rem; color:#6b7280;'>Build the dashboard with ❤️ by Thái</div>\", unsafe_allow_html=True)\n")
print('footer appended')
