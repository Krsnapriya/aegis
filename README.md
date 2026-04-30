---
title: Plutchik ERC Dashboard
emoji: 🎭
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# Plutchik Emotion Recognition in Conversation (ERC) Dashboard

This project focuses on building an advanced **Emotion Recognition in Conversation (ERC)** system using deep learning. Beyond standard sentiment analysis, the system classifies dialogue utterances across **32 granular emotional states** based on Plutchik’s Wheel of Emotions. It features a multi-task architecture that simultaneously detects sarcasm, predicts emotion intensity, and tracks conversational emotion arcs over time. 

Designed for production, it includes Captum-based model explainability and a Human-in-the-Loop (HITL) feedback pipeline connected to a remote Supabase backend, enabling highly nuanced applications in conversational AI and behavioral analytics.

## Features
- **Analyst Dashboard**: Real-time emotion analysis and model explainability (Root URL).
- **Inference API**: Context-aware FastAPI server (Accessible via `/api/`).
- **Annotation Workspace**: HITL labeling tool.

## Deployment
This space runs via Docker and Nginx to support both the Streamlit dashboard and the FastAPI inference server on a single port.

### Accessing the API
- **Health**: `https://YOUR_SPACE_URL/api/health`
- **Predict**: `https://YOUR_SPACE_URL/api/predict`
