import numpy as np
import streamlit as st
import pickle
from pipelines.inference_pipeline import prediction_service_loader

def main():
    st.title("End to End Sentiment Analysis Pipeline with ZenML")

    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to analyze of a given text. It can be either Positive, Negative, Natural or Irrelevant """
    )

    text = st.text_input("Sentence (text)")

    if st.button("Predict"):
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False,
        )

        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )

        pred = service.predict(np.array([text]))

        with open("save_model/model.pkl", "rb") as file:
            lvm = pickle.load(file)

        st.success(
            "Text Sentiment (Positive, Negative, Natural, Irrelevant) is :-{}".format(
                lvm.le.inverse_transform(pred)
            )
        )


if __name__ == "__main__":
    main()
