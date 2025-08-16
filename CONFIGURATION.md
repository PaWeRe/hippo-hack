# Configuration Guide

This application uses environment variables for configuration. Create a `.env` file in the project root with the following variables:

## Required Configuration

### Mistral API Configuration
- `API_KEY_MISTRAL`: Your Mistral API key (required)
- `MODEL`: Mistral model to use (default: `mistral-large-latest`)

### LangGraph Configuration
- `LANGSMITH_API_KEY`: Your LangSmith API key for tracing and monitoring (optional)
- `LANGSMITH_TRACING`: Enable LangSmith tracing (default: `true`)
- `LANGSMITH_PROJECT`: LangSmith project name (default: `medical-xray-workflow`)
- `LANGSMITH_ENDPOINT`: LangSmith API endpoint (default: `https://api.smith.langchain.com`)

### Tavily Configuration
- `TAVILY_API_KEY`: Your Tavily API key for web search capabilities (optional)

## Optional Configuration

### Patient Configuration
- `PATIENT_NAME`: Patient's name (default: `John Doe`)
- `PATIENT_AGE`: Patient's age in years (default: `30`)
- `PATIENT_WEIGHT`: Patient's weight in kg (default: `130`)
- `PATIENT_HEIGHT`: Patient's height in cm (default: `170`)
- `PATIENT_BLOOD_TYPE`: Patient's blood type (default: `A+`)
- `PATIENT_GENDER`: Patient's gender (default: `Male`)
- `PATIENT_HYPERTENSION`: Whether patient has hypertension (default: `True`)

### X-Ray Diagnosis Configuration
- `XRAY_MODEL_WEIGHTS`: TorchXRayVision model weights (default: `densenet121-res224-all`)
- `XRAY_IMAGE_SIZE`: Image size for processing (default: `224`)
- `XRAY_NORMALIZATION_RANGE`: Normalization range for images (default: `255`)

### UI Configuration
- `GRADIO_THEME`: Gradio theme color (default: `indigo`)
- `GRADIO_SHARE`: Whether to create a public share link (default: `False`)
- `GRADIO_DEBUG`: Enable debug mode (default: `True`)

## Example .env File

```env
# Mistral API Configuration
API_KEY_MISTRAL=your_mistral_api_key_here
MODEL=mistral-large-latest

# Langgraph Configuration
LANGSMITH_API_KEY=""
LANGSMITH_TRACING=true
LANGSMITH_PROJECT="medical-xray-workflow"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"

# Tavily Configuration
TAVILY_API_KEY=""

# Patient Configuration
PATIENT_NAME=John Doe
PATIENT_AGE=30
PATIENT_WEIGHT=130
PATIENT_HEIGHT=170
PATIENT_BLOOD_TYPE=A+
PATIENT_GENDER=Male
PATIENT_HYPERTENSION=True

# X-Ray Diagnosis Configuration
XRAY_MODEL_WEIGHTS=densenet121-res224-all
XRAY_IMAGE_SIZE=224
XRAY_NORMALIZATION_RANGE=255

# UI Configuration
GRADIO_THEME=indigo
GRADIO_SHARE=False
GRADIO_DEBUG=True
```

## Available TorchXRayVision Models

You can change the `XRAY_MODEL_WEIGHTS` to use different pre-trained models:

- `densenet121-res224-all`: DenseNet121 trained on all datasets (default)
- `densenet121-res224-chex`: DenseNet121 trained on CheXpert
- `densenet121-res224-mimic_nb`: DenseNet121 trained on MIMIC-CXR
- `densenet121-res224-nih`: DenseNet121 trained on NIH Chest X-ray
- `densenet121-res224-pc`: DenseNet121 trained on PadChest
- `densenet121-res224-kaggle`: DenseNet121 trained on Kaggle Pneumonia

## Available Gradio Themes

You can change the `GRADIO_THEME` to use different color schemes:

- `indigo` (default)
- `blue`
- `green`
- `orange`
- `red`
- `purple`
- `pink`
- `yellow`
