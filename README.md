# 📋 Form Intelligence System

An intelligent, incredibly fast **FastAPI Application** with a modern **HTML/CSS/JS** glassmorphism frontend for **analyzing, understanding, and auto-completing Google Forms & Microsoft Forms**.  

**Previously:** Streamlit monolithic app (~1,100 lines).  
**Now redesigned as:** Microservices-driven FastAPI application (backend) + pure vanilla HTML/JS single-page UI (frontend).

This system can:
- 📝 Extract form structure & questions (Google Forms & Microsoft Forms) via Selenium
- 🤖 Generate smart, context-aware answers using **Gemini AI**
- 🧠 Utilize Persona Profiles & Tone controls
- 📜 View History of past scraped forms
- 📤 Directly export results to custom CSV files

---

## 🚀 Improvements & Features (v4.0 Update)

1. **Brand New UI**: Ditching Streamlit! The new dashboard boasts an incredibly subtle, neat, and highly interactive custom `dark mode` UI featuring CSS grid magic, hover transitions, and a pulsing gradient mesh.
2. **Speed & Stability**: Restructured around asynchronous API architecture avoiding monolithic blocking operations.
3. **No External Libraries for Frontend**: Fully vanilla JS and CSS. No Webpack, npm, or heavy framework headaches required.
4. **Offline Resilience**: Eliminated network dependency issues with NumPy by replacing Pandas reliance with standard `csv` output, preventing `ValueError` binary faults.
5. **Saved State & Profiles**: History table tracking all scraped documents powered by local `sqlite3`.

---

## 🛠️ Tech Stack Upgrade

- **Backend:** Python + `FastAPI` + `Uvicorn`
- **Frontend:** Vanilla HTML5 + CSS3 (Modern Features) + JavaScript (ES6+ Modules & Fetch API)
- **Automation:** Selenium + `webdriver-manager`
- **AI Integration:** Google Gemini API 2.0 Flash

---

## 📦 Installation & Running

### Requirements
Ensure your Python dependencies are met:
```bash
pip3 install fastapi uvicorn google-generativeai selenium webdriver-manager
```
**Make sure `GEMINI_API_KEY` is set in your environment variables:**
```bash
export GEMINI_API_KEY="your-gemini-key-goes-here"
```

### Starting the Application
```bash
python3 server.py
# The application will boot up at http://127.0.0.1:8080
```
Then, simply visit `http://127.0.0.1:8080/` in your browser. All set!

## 👨‍💻 Developer

Architected, developed, and innovated by **1.Sushiiel Anand** **2.Aravind J** **3.Arjun Narayanan**.
