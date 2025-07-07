# 🚀 Deployment Guide

## 💻 Local Deployment

1. **Clone the repository and navigate into it**  
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Create a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Train the model**  
   ```bash
   python scripts/train_high_performance.py
   ```

5. **Run the Streamlit app**  
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Streamlit Community Cloud Deployment

1. Push your code to GitHub.

2. Go to [Streamlit Cloud](https://share.streamlit.io) and log in.

3. Click **New app**, then:
   - Select your GitHub repo.
   - Set `main` branch and `app.py` as the entry point.
   - Ensure `requirements.txt` is present.

4. Click **Deploy**.

---

## 🐳 Docker Deployment (Optional)

1. **Create a `Dockerfile`**  
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app
   COPY . .

   RUN pip install -r requirements.txt

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run the Docker container**  
   ```bash
   docker build -t titanic-app .
   docker run -p 8501:8501 titanic-app
   ```

---

> 💡 Tip: If you're running on Windows, adjust file paths accordingly and ensure Docker Desktop is running before building the container.