import validators
import streamlit as st
from streamlit_lottie import st_lottie
import requests
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Load Lottie Animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

loading_animation = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_vuhz9b5b.json")
success_animation = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_xldzoarx.json")

## Streamlit App
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Custom CSS for Better Styling
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            transition: transform 0.2s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
        }
        .stTextInput input {
            border-radius: 5px;
        }
        .summary-box {
            background-color: black;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #ddd;
        }
    </style>
""", unsafe_allow_html=True)

## Get the Groq API Key and URL (YT or website) to be summarized
with st.sidebar:
    if loading_animation:
        st_lottie(loading_animation, height=100, key="loading1")
    else:
        st.write("Animations")
    
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("Enter URL", label_visibility="collapsed")

## Gemma Model Using Groq API
if groq_api_key.strip():
    llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

    prompt_template = """
    Provide a summary of the following content in 300 words:
    Content:{text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    if st.button("Summarize the Content from YT or Website"):
        ## Validate all the inputs
        if not generic_url.strip():
            st.error("Please provide the URL to get started")
        elif not validators.url(generic_url):
            st.error("Please enter a valid URL. It can be a YT video URL or website URL")
        else:
            try:
                with st.spinner("Fetching and summarizing content..."):
                    if loading_animation:
                        st_lottie(loading_animation, height=200, key="loading2")

                    ## Loading the website or YT video data
                    if "youtube.com" in generic_url:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                    else:
                        loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                       headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs = loader.load()

                    ## Chain For Summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)

                    st.markdown("---")  # Add a separator
                    st.markdown("### 🎉 Summary")
                    st.markdown('<div class="summary-box">{}</div>'.format(output_summary), unsafe_allow_html=True)
                    
                    if success_animation:
                        st_lottie(success_animation, height=150, key="success")

                    st.balloons()  # Add balloons animation

            except Exception as e:
                st.exception(f"Exception: {e}")

else:
    st.warning("Please enter your Groq API Key in the sidebar.")
