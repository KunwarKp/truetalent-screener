# Talent Verification Web App & RAG Pipeline Setup Guide

This guide explains how to configure, set up, and link credentials for the Talent Verification Proof of Concept application.

---

## 1. Quick Start Installation

1. Make sure you have **Python 3.9+** installed on your system.
2. Navigate to this directory in your terminal:
   ```bash
   cd c:/Users/singh/Downloads/talent_verification_pipeline
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment variables template:
   ```bash
   copy .env.example .env
   ```

---

## 2. Linking API Keys & Credentials

Open the `.env` file in a text editor to configure the services:

### A. Grok API Key (xAI LLM Analysis)
*   **Purpose:** Powers the dynamic gap analysis, generates context-aware clarification questions, and provides recruiter-facing synthesis notes.
*   **Where to get it:** Go to the [xAI Console](https://console.x.ai/).
*   **Where to paste:** Add to `XAI_API_KEY=xai-...` in your `.env`.

### B. Scraping API Key (Bucket B Scraper)
*   **Purpose:** Used as a proxy layer to extract portfolio data from third-party sites like LinkedIn, Medium, or personal websites without triggering bot blockades.
*   **Where to get it:** Get a free trial key from [ScrapingBee](https://www.scrapingbee.com/) or [Crawlbase](https://crawlbase.com/).
*   **Where to paste:** Add to `SCRAPING_API_KEY=your_key` in your `.env`.

### C. GitHub Personal Access Token (PAT)
*   **Purpose:** Bypasses basic GitHub API rate-limits to pull repository readme data and metrics for software development portfolios.
*   **Where to get it:** Go to your Github profile Settings -> Developer Settings -> Personal Access Tokens (Classic/Fine-grained) -> Generate new token with read-only scopes.
*   **Where to paste:** Add to `GITHUB_PAT=ghp_...` in your `.env`.

---

## 3. Launching the App

1. Run the local development server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 4. How the PoC RAG Pipeline Works
- **Ingestion:** Candidates submit resumes containing portfolio URLs (e.g., GitHub).
- **Dual-Bucket Streaming:**
  - **Bucket A:** Integrates resume context with candidate answers to intake questions.
  - **Bucket B:** Scrapes raw HTML from portfolio URLs, sanitizer handles the structure, and outputs a flat Markdown text corpus.
- **RAG Scoring:** The Python engine queries this corpus dynamically using a similarity heuristic to verify claims against the Job Description.
