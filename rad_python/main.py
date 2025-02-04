import torchxrayvision as xrv
import skimage, torch, torchvision
import gradio as gr
from openai import OpenAI
from mistralai import Mistral
from datetime import datetime, timedelta
import os
import yaml
import pandas as pd
from dotenv import load_dotenv


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
    def __init__(self):
        self.prediction_results = {}

    def predict(input_img):
        img = skimage.img_as_float(input_img)  # Ensure correct format
        img = xrv.datasets.normalize(
            input_img, 255
        )  # convert 8-bit image to [-1024, 1024] range
        if len(img.shape) == 3:
            img = img.mean(-1)
        img = img[None, ...]  # Make single color channel

        transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )

        img = transform(img)
        img = torch.from_numpy(img).unsqueeze(0)  # Ensure batch dimension

        # Load model and process image
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
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


def generate_report():
    reports = {}
    # TODO: change dummy values to real ouptut from DenseNet!
    reports["2024-04-10-Chest-XRay"] = {
        "img": skimage.io.imread("./rad_python/imgs/00000003_000-Hernia.png"),
        "results": {
            "Atelectasis": 0.5057938,
            "Consolidation": 0.20429493,
            "Infiltration": 0.5312683,
            "Pneumothorax": 0.1890787,
            "Edema": 0.03704696,
            "Emphysema": 0.22286311,
            "Fibrosis": 0.50475776,
            "Effusion": 0.23232938,
            "Pneumonia": 0.050469253,
            "Pleural_Thickening": 0.50521404,
            "Cardiomegaly": 0.4676859,
            "Nodule": 0.49823827,
            "Mass": 0.05959513,
            "Hernia": 0.9981321,
            "Lung Lesion": 0.32798532,
            "Fracture": 0.5010313,
            "Lung Opacity": 0.39410293,
            "Enlarged Cardiomediastinum": 0.3271058,
        },
    }
    reports["2024-03-15-Chest-XRay"] = {
        "img": skimage.io.imread(
            "./rad_python/imgs/00000001_001-Cardiomegaly-Emphysema.png"
        ),
        "results": {
            "Atelectasis": 0.25377643,
            "Consolidation": 0.095511965,
            "Infiltration": 0.2753672,
            "Pneumothorax": 0.13080452,
            "Edema": 0.11183032,
            "Emphysema": 0.30225348,
            "Fibrosis": 0.30676663,
            "Effusion": 0.164064,
            "Pneumonia": 0.0390379,
            "Pleural_Thickening": 0.28225473,
            "Cardiomegaly": 0.43554327,
            "Nodule": 0.30208907,
            "Mass": 0.092965394,
            "Hernia": 0.0033349493,
            "Lung Lesion": 0.09711086,
            "Fracture": 0.22973014,
            "Lung Opacity": 0.17197606,
            "Enlarged Cardiomediastinum": 0.28782028,
        },
    }
    reports["2024-03-05-Breast-MRI"] = {
        "img": skimage.io.imread("./rad_python/imgs/breast_cancer.webp"),
        "results": {"Breast Cancer": 0.25},
    }
    return reports


def update_frame(button_name, reports, patient):
    global radiologist
    latest_report = reports[button_name]
    prediction_results = latest_report["results"]
    first_msg = radiologist.prompt(
        " what do I suffer from",
        [],
        patient.medical_dict,
        prediction_results,
        initial=True,
    )
    return (
        latest_report["img"],
        latest_report["results"],
        [["", first_msg]],
        [["", first_msg]],
        button_name,
    )


def main(configs):

    global radiologist  # not deepcopyable because of Mistral _thread.RLock object

    patient = Patient("John Doe", 30, 130, 170, "A+", "Male", True)
    radiologist = Radiologist(configs["API_KEY_MISTRAL"], configs["MODEL"])
    xray = XRayDiagnosis()

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
        primary_hue="indigo",
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
                        buttons = {}
                        for report_name in reports.keys():
                            buttons[report_name] = gr.Button(report_name)
                    df = gr.DataFrame(medical_df, headers=None)
                with gr.Column(scale=1.4):
                    reportchat_button = gr.Button("HippoChat", interactive=False)
                    chat = gr.ChatInterface(radiologist.prompt, chatbot=bot)
                with gr.Column(scale=1.4):
                    reportreport_button = gr.Button(
                        "Report", interactive=True, variant="primary"
                    )
                    results_img = gr.Image()
                    results_barplot = gr.Label(num_top_classes=7, show_label=False)
            for button_name, report in reports.items():
                buttons[button_name].click(
                    update_frame,
                    [
                        buttons[button_name],
                        reports_state,
                        patient_state,
                    ],
                    [
                        results_img,
                        results_barplot,
                        bot,
                        chat.chatbot_state,
                        reportreport_button,
                    ],
                )

        with gr.Tab("Technician Interface") as technician_tab:
            with gr.Row():
                with gr.Column():
                    input_img_interface = gr.Image()
                    run_button = gr.Button("Predict")
                with gr.Column():
                    results_interface_tech = gr.Label()
                    with gr.Row():
                        send_to_user = gr.Button("Send to Patient")
                        send_to_export = gr.Button("Send to Expert", variant="stop")

            run_button.click(
                xray.predict, [input_img_interface], [results_interface_tech]
            )
        demo.queue().launch(share=False, debug=True)


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # load in YAML configuration
    configs = {}
    base_config_path = os.path.join(os.getcwd() + "/rad_python/configs.yaml")
    with open(base_config_path, "r") as file:
        configs.update(yaml.safe_load(file))
    
    # Load sensitive configurations from environment variables
    configs["API_KEY_MISTRAL"] = os.getenv("API_KEY_MISTRAL")
    configs["PATH_TO_DOWNLOADED_WEIGHTS"] = os.getenv("PATH_TO_DOWNLOADED_WEIGHTS")
    
    # Validate that required environment variables are set
    if not configs["API_KEY_MISTRAL"]:
        raise ValueError("API_KEY_MISTRAL environment variable is required. Please check your .env file.")
    if not configs["PATH_TO_DOWNLOADED_WEIGHTS"]:
        raise ValueError("PATH_TO_DOWNLOADED_WEIGHTS environment variable is required. Please check your .env file.")
    
    main(configs)
