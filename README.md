# Mood Journal - AI-Powered Emotion Tracker

An intelligent web application that analyzes your journal entries in real-time using AI to help you understand and track your emotions. Built for the vibe coding Hackathon.

## 🚀 Live Demo
> **[Click here to experience the live app!](YOUR_LIVE_DEPLOYMENT_LINK_HERE)**
*(We will add this link after the next step)*

## ✨ Features
- **AI-Powered Analysis**: Utilizes the Hugging Face `emotion-english-distilroberta-base` model for accurate sentiment analysis.
- **Interactive History**: View your past entries and mood trends.
- **Beautiful Visualization**: See your emotional data presented in an intuitive chart.
- **Secure & Private**: Your entries are stored securely in a Supabase database.

## 🛠️ Tech Stack
- **Frontend**: HTML5, CSS3, JavaScript (Chart.js)
- **Backend**: Python (Flask)
- **Database**: Supabase (PostgreSQL)
- **AI**: Hugging Face Inference API
- **Deployment**: Railway.app

## 📦 Installation & Setup
Want to run this locally? Follow these steps:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/buseiry/mood-journal-hackathon.git
    cd mood-journal-hackathon
    ```

2.  **Install Python dependencies**
    ```bash
    pip install flask supabase python-dotenv requests
    ```

3.  **Set up environment variables**
    Create a `.env` file in the root directory and add your keys:
    ```bash
    SUPABASE_URL=your_supabase_project_url
    SUPABASE_KEY=your_supabase_anon_key
    HUGGING_FACE_API_KEY=your_hugging_face_api_token
    ```

4.  **Run the application**
    ```bash
    python app.py
    ```
    Open your browser and go to `http://localhost:5000`

## 📸 Screenshots
*(We can add screenshots of your app here later)*

## 🙋‍♂️ Author
**Buseiry**  
- GitHub: [@buseiry](https://github.com/buseiry)
- Project Link: [https://github.com/buseiry/mood-journal-hackathon](https://github.com/buseiry/mood-journal-hackathon)

## 🙏 Acknowledgments
- Hugging Face for their incredible emotion analysis model.
- Supabase for the easy-to-use database platform.
- The hackathon organizers for the opportunity.
