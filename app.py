# app.py
import streamlit as st
import google.generativeai as genai
import json

st.write(f"Using google-generativeai version: {genai.__version__}")

# --- Page and Model Configuration ---

# Set the page configuration, a standard first step for Streamlit apps
st.set_page_config(
    page_title="Biomedical Abstract Triage Assistant",
    page_icon="🔬",
    layout="wide" # Use wide layout for better readability of abstracts
)

# Securely configure the Gemini API using Streamlit's secrets management.
# This is a critical lesson from our previous work.
try:
    genai.configure(api_key=st.secrets["google_api_key"])
except (KeyError, AttributeError):
    st.error("Your Google API key is missing or invalid. Please ensure you have a `.streamlit/secrets.toml` file with your key.")
    st.stop()


# --- Core Classification Function ---

def classify_abstract(abstract_text: str) -> dict:
    """
    Sends the abstract to the Gemini model with detailed instructions for classification.
    Returns the classification label and reason as a dictionary.
    """
    # The detailed, multi-part prompt you provided, including role, labels, definitions, and few-shot examples
    prompt = f"""
You are a biomedical abstract triage assistant. I want you to classify PubMed abstracts into one of five categories. For each input, provide both a classification label and a brief reasoning statement.

Labels:

1. SkyClarys/Omaveloxolone
2. FA Mechanisms (Iron/Ferroptosis/ROS)
3. Other Drugs/Compounds Targeting Iron/Ferroptosis/ROS
4. General FA (Genetics/Clinical)
5. Irrelevant

---

The label definitions are as follows:
- SkyClarys/Omaveloxolone: Abstracts mentioning omaveloxolone (SkyClarys), bardoxolone methyl, or related analogs.
- FA Mechanisms: Abstracts about FA biology including frataxin deficiency, iron overload, mitochondrial dysfunction, oxidative stress, ROS, lipid peroxidation, ferroptosis, GPX4.
- Other Drugs/Compounds: Abstracts about non-omaveloxolone drugs or compounds that modulate ferroptosis, GPX4, iron chelation, lipid peroxidation, or mitochondrial antioxidant pathways, even if studied in other diseases or models (exclude NRF2/KEAP1 activators).
- General FA: Abstracts about FA genetics, prevalence, natural history, or clinical scales, without mechanistic or therapeutic depth.
- Irrelevant: Everything else.

---

Here are several few-shot examples:

Example 1
Input: "In a phase II trial of omaveloxolone in Friedreich’s ataxia, activation of NRF2 improved mFARS scores versus placebo."
Output (JSON only): {{"label":"SkyClarys/Omaveloxolone","reason":"Omaveloxolone activates NRF2 and was tested in FA patients."}}

Example 2
Input: "Frataxin deficiency causes mitochondrial iron accumulation and lipid peroxidation; GPX4 overexpression restored cell viability."
Output (JSON only): {{"label":"FA Mechanisms (Iron/Ferroptosis/ROS)","reason":"FA model shows ferroptosis via iron overload and GPX4 rescue."}}

Example 3
Input: "Ferrostatin-1, a ferroptosis inhibitor, prevented lipid peroxidation and neuronal loss in a mouse model of Parkinson’s disease."
Output (JSON only): {{"label":"Other Drugs/Compounds Targeting Iron/Ferroptosis/ROS","reason":"Ferrostatin-1 blocks ferroptosis and lipid peroxidation, relevant for repurposing."}}

---

Now classify this input and provide the output in this format and do not include additional information beyond what is asked in the JSON output:

Input: {{{abstract_text}}}

Output (JSON only):
"""

    try:
        # Use the latest stable Gemini model, as we learned older names can be deprecated
        model = genai.GenerativeModel('gemini-pro-latest')
        response = model.generate_content(prompt)
        
        # Clean the response to ensure it's valid JSON
        clean_response = response.text.strip().replace("```json", "").replace("```", "")
        
        # Parse the JSON string into a Python dictionary
        result = json.loads(clean_response)
        return result

    except Exception as e:
        # Handle API errors (like the 429 quota error) and JSON parsing errors
        st.error(f"An error occurred. It could be an API quota issue or a malformed response from the model. Details: {e}")
        return None


# --- Streamlit App UI ---

st.title("🔬 Biomedical Abstract Triage Assistant")
st.markdown(
    "This app uses AI to classify biomedical abstracts into one of five categories related to Friedreich's Ataxia (FA)."
)

# Example abstract from your prompt
example_abstract = (
    "Ferroptosis is an iron-dependent form of regulated cell death, arising from the accumulation of lipid-based "
    "reactive oxygen species when glutathione-dependent repair systems are compromised. Lipid peroxidation, "
    "mitochondrial impairment and iron dyshomeostasis are the hallmark of ferroptosis, which is emerging as a "
    "crucial player in neurodegeneration. This review provides an analysis of the most recent advances in "
    "ferroptosis, with a special focus on Friedreich's Ataxia (FA), the most common autosomal recessive "
    "neurodegenerative disease, caused by reduced levels of frataxin, a mitochondrial protein involved in "
    "iron-sulfur cluster synthesis and antioxidant defenses. The hypothesis is that the iron-induced oxidative "
    "damage accumulates over time in FA, lowering the ferroptosis threshold and leading to neuronal cell death "
    "and, at last, to cardiac failure. The use of anti-ferroptosis drugs combined with treatments able to activate "
    "the antioxidant response will be of paramount importance in FA therapy, such as in many other neurodegenerative "
    "diseases triggered by oxidative stress."
)

# Text area for user input
user_input = st.text_area(
    "Paste the abstract text below:",
    value=example_abstract,
    height=300,
)

# Button to trigger the classification
if st.button("Classify Abstract", type="primary"):
    if user_input and user_input.strip():
        with st.spinner("🧠 Classifying..."):
            result_data = classify_abstract(user_input)
            
            if result_data and "label" in result_data and "reason" in result_data:
                st.subheader("Classification Result")
                st.info(f"**Label:** {result_data['label']}")
                st.markdown(f"**Reasoning:** {result_data['reason']}")
            else:
                st.error("Could not parse the result from the AI. The response may have been empty or malformed.")
    else:
        st.warning("Please enter an abstract to classify.")