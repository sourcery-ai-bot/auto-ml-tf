import streamlit as st
import os
import pandas as pd
from streamlit.hashing import _CodeHasher

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server


def main():
    state = _get_state()
    pages = {
        "Dashboard": page_dashboard,
        "Object Tracking": page_object_tracking,
        "Coverter para .Record": page_converted_record,
    }

    st.sidebar.title(":floppy_disk: Page states")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


def page_dashboard(state):
    st.title("Informacoes iniciais para o usuario")

def page_converted_record(state):
    st.title("Converter os arquivos CSV para RECORD")

    if st.button('Converter'):
        try:
           os.system(
            'python scripts/opencv_object_tracking.py \
                --video {} \
                --tracker {} \
                --label {} \
                --vocxml {} \
                --desting train \
                --imagesdirectory images \
                --cropimages {} \
                '.format(state.video, state.track, state.label,
                    state.voc, state.cropImage))
        except:
            st.error("Error while running the script open_object_tracking.py")
            raise Exception("Error while running the script open_object_tracking.py")
        
        try:
            os.system(
                'python scripts/random_samples.py --folder images --train_num {} \
                    '.format(state.data_train)
            )
        except:
            st.error("Error while running the script random_sample.py")
            raise Exception("Error while running the script random_sample.py")
        
        try:
            os.system('python scripts/xml_to_csv.py --input {} \
                --output ./csvs/ --file {}'.format('train', 'train'))
            os.system('python scripts/xml_to_csv.py --input {} \
                --output ./csvs/ --file {}'.format('test', 'test'))
        except:
            st.error("Error while running the script xml_to_csv.py")
            raise Exception("Error while running the script xml_to_csv.py")
        
        try:
            os.system('python scripts/generate_tfrecord.py \
                --csv_input=csvs/train_labels.csv --image_dir=./train \
                --output_path=train.record')
            
            os.system('python scripts/generate_tfrecord.py \
                --csv_input=csvs/test_labels.csv --image_dir=./test \
                --output_path=test.record')
        except:
            st.error("Error while running the script generate_tfrecord.py")
            raise Exception("Error while running the script generate_tfrecord.py")

        st.success("Conversão realizada com sucesso!☺")
        
        st.write("Dados de treino")
        st.dataframe(pd.read_csv("./csvs/train_labels.csv"), width=800, height=500)
        
        st.write("Dados de teste")
        st.dataframe(pd.read_csv("./csvs/teste_labels.csv"), width=800, height=500)

def page_object_tracking(state):
    st.title("Bem vindo ao Object tracking")

    options = ["csrt", "kcf", "boosting", "mil", "tld", "medianflow", "mosse", "goturn"]
    state.track = st.selectbox("Selecione o Tipo de rastreador de objeto OpenCV prefereido", options)
    state.video = st.text_input("Informe o diretório do video/imagem:", state.video or '')
    state.label = st.text_input("Informe o rótulo do objeto:", state.label or '')
    state.voc = st.checkbox("Eu quero gerar arquivos pascal VOC XML 📃", state.voc)
    state.yolo = st.checkbox("Eu quero gerar arquivos yolo TXT 📄", state.yolo)
    state.cropImage = st.checkbox("Eu desejo Recortar os objetos das imagens 🤙", state.cropImage)
    if state.voc == True and state.yolo == True:
        st.error("Selecione apenas uma das Opcoes 😉")

    state.data_train = st.slider("Dividindo os dados para treino e teste", 
    min_value=1, max_value=100, step=1)
    state.data_teste = 100 - state.data_train
    st.write("Dados para treino: {}%".format(state.data_train))
    st.write("Dados para teste: {}%".format(state.data_teste))

def display_state_values(state):
    st.write("Input state:", state.video)
    st.write("Slider state:", state.folder)
    st.write("Radio state:", state.radio)
    st.write("Checkbox state:", state.checkbox)
    st.write("Selectbox state:", state.selectbox)
    st.write("Multiselect state:", state.multiselect)
    
    for i in range(3):
        st.write(f"Value {i}:", state[f"State value {i}"])

    if st.button("Clear state"):
        state.clear()


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()