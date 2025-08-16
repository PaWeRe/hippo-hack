"""
Gradio Frontend for Medical AI Agent
Direct integration with LangGraph medical agent
"""

import torchxrayvision as xrv
import skimage, torch, torchvision
import gradio as gr
from mistralai import Mistral
from datetime import datetime, timedelta
import os
import pandas as pd
from dotenv import load_dotenv
import tempfile
import shutil
from src.agent.graph import graph, MedicalAgentState

# Load environment variables
load_dotenv()


class Patient:
    def __init__(
        self,
        patient_name,
        patient_age,
        patient_weight,
        patient_height,
        patient_blood_type,
        patient_gender,
        hypertension,
    ):
        self.medical_dict = {
            "key": [
                "patient_name",
                "patient_age",
                "patient_weight",
                "patient_height",
                "patient_blood_type",
                "patient_gender",
                "hypertension",
            ],
            "value": [
                patient_name,
                patient_age,
                patient_weight,
                patient_height,
                patient_blood_type,
                patient_gender,
                hypertension,
            ],
        }

    def calculate_bmi(self):
        # Extracting height and weight from medical_dict
        height = self.medical_dict["value"][
            self.medical_dict["key"].index("patient_height")
        ]
        weight = self.medical_dict["value"][
            self.medical_dict["key"].index("patient_weight")
        ]

        # Checking if height is in meters, if not converting it to meters
        if height > 3:
            height /= 100  # Converting height from centimeters to meters

        # Calculating BMI
        return weight / (height**2)

    def categorize_bmi(bmi):
        if bmi < 18.5:
            return "Underweight"
        elif bmi >= 18.5 and bmi < 25:
            return "Normal"
        else:
            return "Overweight"


class XRayDiagnosis:
    def __init__(
        self, weights="densenet121-res224-all", image_size=224, normalization_range=255
    ):
        self.prediction_results = {}
        self.weights = weights
        self.image_size = image_size
        self.normalization_range = normalization_range

    def predict(self, input_img):
        img = skimage.img_as_float(input_img)  # Ensure correct format
        img = xrv.datasets.normalize(
            input_img, self.normalization_range
        )  # convert 8-bit image to [-1024, 1024] range
        if len(img.shape) == 3:
            img = img.mean(-1)
        img = img[None, ...]  # Make single color channel

        transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(self.image_size)]
        )

        img = transform(img)
        img = torch.from_numpy(img).unsqueeze(0)  # Ensure batch dimension

        # Load model with specified torchxrayvision weights
        model = xrv.models.DenseNet(weights=self.weights)
        outputs = model(img)
        return dict(zip(model.pathologies, outputs[0].detach().numpy()))


class Radiologist:
    def __init__(self, api_key, model_name):
        self.client = Mistral(api_key=api_key)
        self.model_name = model_name

    def prompt(
        self,
        new_user_prompt,
        chat_history,
        patient_med_hist,
        diagnostic_dict,
        initial=False,
    ):
        """
        Roles: System, Assistant, User
        """
        self.messages = [
            {
                "role": "system",
                "content": f"You are an expert Radiologist. The patient you are interacting with has the following medical background {patient_med_hist}. Your patient just received the following medical imaging results {diagnostic_dict}. Please describe briefly what kind of imaging exam was done and then notify the patient about the most important finding ONLY. Help the patient to figure out what the next steps are. Address the patient directly and answer any follow-up questions they might have about the diagnosis.",
            },
        ]
        # TODO: implement a sliding context window approach for more efficient context mgmt
        for chat in chat_history:
            self.messages.extend(
                [
                    {"role": "user", "content": chat[0]},
                    {"role": "assistant", "content": chat[1]},
                ]
            )
        self.messages.append(
            {
                "role": "user",
                "content": "given the medical knowledge you have about me"
                + new_user_prompt,
            }
        )
        print(self.messages)
        completion = self.client.chat.complete(
            model=self.model_name, messages=self.messages
        )
        return completion.choices[0].message.content


# Global state for storing reports as they are created
stored_reports = {}

# Global state for image sharing between technician and patient interfaces
current_analysis_state = {
    "image": None,
    "results": None,
    "ready": False,
}

# Global variable to store last sources for display
last_sources = ""

# Global variable to store tool activity for display
last_tool_activity = ""

# Global variable to store current patient information
current_patient_info = {}


def generate_report():
    """Return current stored reports (starts empty)"""
    return stored_reports.copy()


def add_new_report(image, results, report_name=None):
    """Add a new report to the stored reports"""
    global stored_reports

    if report_name is None:
        from datetime import datetime

        report_name = f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-Chest-XRay"

    stored_reports[report_name] = {"img": image, "results": results}

    return report_name


def update_frame(reports, patient, button_name):
    """Handle clicking on historical report buttons"""
    global current_analysis_state

    if button_name not in reports:
        return None, None, [], [], ""

    latest_report = reports[button_name]
    prediction_results = latest_report["results"]

    # Update global state
    current_analysis_state.update(
        {
            "image": latest_report["img"],
            "results": prediction_results,
            "ready": True,
        }
    )

    # Get the analysis message by asking the agent
    chat_response = chat_with_medical_agent(
        "Please analyze these X-ray results and provide a summary",
        prediction_results,
        current_patient_info,
    )
    analysis_message = chat_response["response"]

    return (
        latest_report["img"],
        latest_report["results"],
        [["", analysis_message]],
        [["", analysis_message]],
        button_name,
    )


def process_new_xray_technician(input_img):
    """Process new X-ray image from technician interface"""
    global xray, current_analysis_state

    if input_img is None:
        return None, "Please upload an image first"

    # Run X-ray analysis
    prediction_results = xray.predict(input_img)

    # Save the uploaded image temporarily
    temp_path = tempfile.mktemp(suffix=".png")
    skimage.io.imsave(temp_path, input_img)

    # Update global state
    current_analysis_state.update(
        {
            "image": input_img,
            "results": prediction_results,
            "ready": False,
            "temp_path": temp_path,
        }
    )

    return (
        prediction_results,
        "Analysis complete. Click 'Send to Patient' to share results.",
    )


def send_to_patient():
    """Send analysis results to patient interface and add to stored reports"""
    global current_analysis_state, stored_reports

    if not current_analysis_state["results"]:
        return "No analysis results to send"

    # Add to stored reports
    report_name = add_new_report(
        current_analysis_state["image"], current_analysis_state["results"]
    )

    current_analysis_state.update(
        {
            "ready": True,
            "report_name": report_name,
        }
    )

    return f"✅ Results sent instantly! Report: {report_name}"


def get_patient_analysis():
    """Get current analysis for patient interface"""
    global current_analysis_state

    if not current_analysis_state["ready"]:
        return (
            None,
            None,
            "No analysis available. Please wait for technician to send results.",
        )

    # Get analysis by asking the agent
    chat_response = chat_with_medical_agent(
        "Please provide a comprehensive analysis of these X-ray results",
        current_analysis_state["results"],
        current_patient_info,
    )

    return (
        current_analysis_state["image"],
        current_analysis_state["results"],
        chat_response["response"],
    )


def chat_with_medical_agent(
    message: str, xray_results: dict, patient_info: dict = None
) -> dict:
    """Direct interface to the medical agent graph"""
    global last_tool_activity, current_patient_info

    from langchain_core.messages import HumanMessage

    # Create initial state with proper LangChain message format
    from langchain_core.messages import SystemMessage

    # Use provided patient_info or fall back to global current_patient_info
    if patient_info is None:
        patient_info = current_patient_info

    # Add contextual system message based on available data
    system_content = """You are HippoChat, a friendly medical AI assistant. You can help with:
- Analyzing X-ray results when available
- Explaining medical conditions and terms in simple language
- Searching for current medical information when needed

Keep your responses concise and focused. Only discuss follow-up steps or recommendations when explicitly asked or when using the generate_recommendations tool. Be conversational, caring, and helpful. Always remind users to consult healthcare professionals for medical decisions."""

    # Add context about available X-ray data
    if xray_results:
        system_content += f"\n\nCURRENT CONTEXT: X-ray analysis results are available. When asked about X-ray findings, use the extract_top_findings tool first."

    # Add patient context if available
    if patient_info:
        patient_context = f"\n\nPATIENT INFORMATION: You are helping {patient_info.get('name', 'a patient')}"
        if patient_info.get("age"):
            patient_context += f", age {patient_info.get('age')}"
        if patient_info.get("gender"):
            patient_context += f", {patient_info.get('gender').lower()}"
        if patient_info.get("hypertension"):
            patient_context += f", with known hypertension"
        patient_context += (
            ". Consider these patient factors in your analysis and recommendations."
        )
        system_content += patient_context

    messages = [SystemMessage(content=system_content), HumanMessage(content=message)]

    initial_state = MedicalAgentState(
        messages=messages,
        xray_results=xray_results,
        patient_info=patient_info,
    )

    # Run the graph
    final_state = graph.invoke(initial_state)

    # Track tool activity and extract AI response
    ai_message = None

    for msg in final_state["messages"]:
        # Extract final AI response (look for the last assistant message without tool calls)
        if (
            hasattr(msg, "content")
            and msg.content
            and not (hasattr(msg, "tool_calls") and msg.tool_calls)
            and getattr(msg, "type", None) != "tool"
        ):
            ai_message = msg.content

    # Format tool activity for display with actual tool calls and responses
    tool_messages = []
    current_tool_call = None

    # Extract tool calls and their responses from the message flow
    for msg in final_state["messages"]:
        # Track tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_messages.append(
                    {
                        "type": "tool_call",
                        "name": tool_call["name"],
                        "args": tool_call.get("args", {}),
                        "id": tool_call.get("id", ""),
                        "response": None,  # Will be filled by matching ToolMessage
                    }
                )

        # Match tool responses
        elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
            # Find the matching tool call and add the response
            for tool_msg in reversed(tool_messages):
                if (
                    tool_msg["type"] == "tool_call"
                    and tool_msg["id"] == msg.tool_call_id
                ):
                    tool_msg["response"] = msg.content
                    break

    if tool_messages:
        activity_text = (
            f"🔧 **Tool Activity Log** ({len(tool_messages)} tool call(s)):\n\n"
        )

        for i, tool_msg in enumerate(tool_messages, 1):
            tool_name = tool_msg["name"]
            args = tool_msg["args"]
            response = tool_msg["response"] or "No response captured"

            activity_text += f"**{i}. {tool_name}**\n"

            # Show arguments
            if args:
                activity_text += f"📥 **Input:** {args}\n"

            # Show full tool response (no truncation)
            activity_text += f"📤 **Output:** {response}\n\n"
            activity_text += "-" * 50 + "\n\n"

        last_tool_activity = activity_text
    else:
        last_tool_activity = "💬 No tools were called for this query."

    return {
        "response": ai_message or "I'm sorry, I couldn't generate a response.",
        "tool_calls": tool_messages,  # Return the detailed tool messages instead
        "raw_messages": final_state["messages"],
    }


def patient_question_handler(message, history):
    """Handle patient questions using the medical agent graph"""
    global current_analysis_state

    if not current_analysis_state["ready"] or not current_analysis_state["results"]:
        return "Please wait for your X-ray analysis to be available."

    # Use the medical agent graph directly
    chat_result = chat_with_medical_agent(
        message, current_analysis_state["results"], current_patient_info
    )

    # Return only the response
    return chat_result["response"]


def get_last_tool_activity():
    """Get the last tool activity for display"""
    global last_tool_activity
    return last_tool_activity


def update_reports_display():
    """Update the reports display with current available reports"""
    global stored_reports

    if not stored_reports:
        return gr.update(visible=False, choices=[]), gr.update(visible=False)

    # Get sorted report names (most recent first)
    report_names = sorted(stored_reports.keys(), reverse=True)

    return (
        gr.update(
            visible=True,
            choices=report_names,
            value=report_names[0] if report_names else None,
        ),
        gr.update(visible=True),
    )


def load_specific_report(report_name):
    """Load a specific report by name"""
    global stored_reports, current_analysis_state

    if report_name not in stored_reports:
        return None, {}

    report = stored_reports[report_name]

    # Update current analysis state to this report
    current_analysis_state.update(
        {
            "image": report["img"],
            "results": report["results"],
            "ready": True,
            "report_name": report_name,
        }
    )

    return report["img"], report["results"]


def main(configs):
    global radiologist, xray, current_patient_info  # Global variables for sharing between interfaces

    # Initialize patient with configuration
    patient = Patient(
        configs["PATIENT_NAME"],
        configs["PATIENT_AGE"],
        configs["PATIENT_WEIGHT"],
        configs["PATIENT_HEIGHT"],
        configs["PATIENT_BLOOD_TYPE"],
        configs["PATIENT_GENDER"],
        configs["PATIENT_HYPERTENSION"],
    )

    # Set global patient information for the medical agent
    current_patient_info = {
        "name": configs["PATIENT_NAME"],
        "age": configs["PATIENT_AGE"],
        "weight": configs["PATIENT_WEIGHT"],
        "height": configs["PATIENT_HEIGHT"],
        "blood_type": configs["PATIENT_BLOOD_TYPE"],
        "gender": configs["PATIENT_GENDER"],
        "hypertension": configs["PATIENT_HYPERTENSION"],
        "bmi": patient.calculate_bmi(),
    }

    # Initialize radiologist and xray with configuration
    radiologist = Radiologist(configs["API_KEY_MISTRAL"], configs["MODEL"])
    xray = XRayDiagnosis(
        weights=configs["XRAY_MODEL_WEIGHTS"],
        image_size=configs["XRAY_IMAGE_SIZE"],
        normalization_range=configs["XRAY_NORMALIZATION_RANGE"],
    )

    reports = generate_report()
    medical_df = pd.DataFrame(patient.medical_dict)

    js_func = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'light') {
            url.searchParams.set('__theme', 'light');
            window.location.href = url.href;
        }
    }
    """
    css = """
    .container {
        width: 20vw;
    }
    """

    theme = gr.themes.Base(
        primary_hue=configs["GRADIO_THEME"],
    )

    with gr.Blocks(js=js_func, theme=theme) as demo:
        reports_state = gr.State(reports)
        patient_state = gr.State(patient)
        with gr.Tab("User Interface") as user_tab:
            bot = gr.Chatbot(render=False, height=800)
            with gr.Row():
                with gr.Column(scale=0.5):
                    report_button = gr.Button("Reports", interactive=False)
                    with gr.Group():
                        # Report selection area
                        refresh_reports_btn = gr.Button(
                            "Refresh Reports", variant="secondary"
                        )
                        reports_dropdown = gr.Dropdown(
                            label="Select Report",
                            choices=[],
                            interactive=True,
                            visible=False,
                        )
                        load_report_btn = gr.Button(
                            "Load Selected Report", variant="primary", visible=False
                        )
                    df = gr.DataFrame(medical_df, headers=None)

                    # Tool Activity section - moved here below personal info
                    with gr.Accordion("🔧 Tool Activity", open=True):
                        tool_activity_display = gr.Textbox(
                            label="Actual Tool Calls & Responses",
                            lines=8,
                            interactive=False,
                            value="Tool activity will appear here after asking questions...",
                            show_copy_button=True,
                        )
                        refresh_tools_btn = gr.Button(
                            "Refresh Tool Activity", size="sm"
                        )
                with gr.Column(scale=1.4):
                    # Simple HippoChat button and interface
                    hippo_chat_button = gr.Button(
                        "HippoChat - AI Medical Assistant", interactive=False
                    )
                    chat = gr.ChatInterface(patient_question_handler, chatbot=bot)

                    # Auto-refresh tool activity when chat is used
                    chat.textbox.submit(
                        lambda: get_last_tool_activity(),
                        outputs=[tool_activity_display],
                        queue=False,
                    )

                    # Debug panel to show agent activity
                    with gr.Accordion("🔍 Agent Activity (Debug)", open=False):
                        debug_info = gr.Textbox(
                            label="Last LLM Call & RAG Activity",
                            lines=8,
                            interactive=False,
                        )
                        refresh_debug_btn = gr.Button("Refresh Debug Info", size="sm")
                with gr.Column(scale=1.4):
                    current_report_button = gr.Button(
                        "Current Analysis", interactive=False, variant="secondary"
                    )
                    results_img = gr.Image(label="X-ray Image")
                    results_barplot = gr.Label(
                        num_top_classes=7, show_label=False, label="AI Analysis Results"
                    )

        with gr.Tab("Technician Interface") as technician_tab:
            gr.Markdown("## Upload and Analyze X-ray Images")
            gr.Markdown(
                "Upload patient X-ray images for AI analysis, then send results to patient interface."
            )

            with gr.Row():
                with gr.Column():
                    input_img_interface = gr.Image(
                        label="Upload X-ray Image", type="numpy"
                    )
                    run_button = gr.Button("Analyze X-ray", variant="primary")

                with gr.Column():
                    results_interface_tech = gr.Label(
                        label="Analysis Results", num_top_classes=10
                    )
                    analysis_status = gr.Textbox(
                        label="Status", value="Ready for upload", interactive=False
                    )

                    with gr.Row():
                        send_to_patient_btn = gr.Button(
                            "Send to Patient", variant="primary"
                        )
                        send_to_export = gr.Button("Send to Expert", variant="stop")

            # Connect technician interface functions
            run_button.click(
                process_new_xray_technician,
                [input_img_interface],
                [results_interface_tech, analysis_status],
            )

            send_to_patient_btn.click(send_to_patient, outputs=[analysis_status])

            # Add refresh functionality for patient interface
            def refresh_reports():
                return update_reports_display()

            def load_selected_report(selected_report):
                global last_tool_activity
                if selected_report:
                    img, results = load_specific_report(selected_report)
                    # Reset chat history and tool activity
                    last_tool_activity = (
                        "Tool activity will appear here after asking questions..."
                    )
                    return (
                        img,
                        results,
                        [],
                        last_tool_activity,
                    )  # Clear chatbot and tool activity
                return (
                    None,
                    {},
                    [],
                    "Tool activity will appear here after asking questions...",
                )

            refresh_reports_btn.click(
                refresh_reports, outputs=[reports_dropdown, load_report_btn]
            )

            load_report_btn.click(
                load_selected_report,
                inputs=[reports_dropdown],
                outputs=[results_img, results_barplot, bot, tool_activity_display],
            )

            # Debug functionality
            def get_debug_info():
                return "Medical Agent Graph is running. Check LangGraph Studio for detailed debugging."

            refresh_debug_btn.click(get_debug_info, outputs=[debug_info])

            refresh_tools_btn.click(
                get_last_tool_activity, outputs=[tool_activity_display]
            )

        demo.queue().launch(
            share=configs["GRADIO_SHARE"], debug=configs["GRADIO_DEBUG"]
        )


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Create comprehensive configs dictionary from environment variables
    configs = {
        # Mistral API Configuration
        "API_KEY_MISTRAL": os.getenv("API_KEY_MISTRAL"),
        "MODEL": os.getenv("MODEL", "mistral-large-latest"),
        # Patient Configuration
        "PATIENT_NAME": os.getenv("PATIENT_NAME", "John Doe"),
        "PATIENT_AGE": int(os.getenv("PATIENT_AGE", "30")),
        "PATIENT_WEIGHT": float(os.getenv("PATIENT_WEIGHT", "130")),  # kg
        "PATIENT_HEIGHT": float(os.getenv("PATIENT_HEIGHT", "170")),  # cm
        "PATIENT_BLOOD_TYPE": os.getenv("PATIENT_BLOOD_TYPE", "A+"),
        "PATIENT_GENDER": os.getenv("PATIENT_GENDER", "Male"),
        "PATIENT_HYPERTENSION": os.getenv("PATIENT_HYPERTENSION", "True").lower()
        == "true",
        # X-Ray Diagnosis Configuration
        "XRAY_MODEL_WEIGHTS": os.getenv("XRAY_MODEL_WEIGHTS", "densenet121-res224-all"),
        "XRAY_IMAGE_SIZE": int(os.getenv("XRAY_IMAGE_SIZE", "224")),
        "XRAY_NORMALIZATION_RANGE": int(os.getenv("XRAY_NORMALIZATION_RANGE", "255")),
        # UI Configuration
        "GRADIO_THEME": os.getenv("GRADIO_THEME", "indigo"),
        "GRADIO_SHARE": os.getenv("GRADIO_SHARE", "False").lower() == "true",
        "GRADIO_DEBUG": os.getenv("GRADIO_DEBUG", "True").lower() == "true",
    }

    # Validate that required environment variables are set
    if not configs["API_KEY_MISTRAL"]:
        raise ValueError(
            "API_KEY_MISTRAL environment variable is required. Please check your .env file."
        )

    main(configs)
