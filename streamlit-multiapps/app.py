import streamlit as st
from multiapp import MultiApp
from apps import home, cards2, face  # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("🃏 Poker Recognize", cards2.app)
app.add_app("Face Detection", face.app)


# The main app
app.run()
