import os
import streamlit as st
from feynman_learning import *
import fitz
import threading
import time

# 这里添加API KEY


def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")  # 仅提取文本
    return text

note: str = ""
initial_input = None
str_formatted_checkpoints = None
current_question = None
current_answer = None
str_verification_results = None
str_teaching_results = None

st.title("费曼学习助手")

if 'learningService' not in st.session_state:
    st.session_state['learningService'] = FeynmanLearning()
learningService = st.session_state['learningService']

if 'learningState' not in st.session_state:
    st.session_state['learningState'] = False

if st.session_state['learningState'] == False:
    st.markdown("学习主题")
    text_area_topic = st.text_area("学习主题", label_visibility="collapsed")
    st.markdown("学习目标")
    text_area_goals = st.text_area("学习目标", label_visibility="collapsed")
    st.markdown("学习材料")
    text_area_context = st.text_area("学习材料", label_visibility="collapsed")

    if st.button("开始学习", disabled=st.session_state['learningState'] == True):
        if text_area_topic.strip() and text_area_goals.strip():
            print("已经读取学习主题和学习目标")
            print(text_area_topic)
            print(text_area_goals)
            st.session_state['textTopic'] = text_area_topic
            st.session_state['textGoals'] = text_area_goals
            st.session_state['textContext'] = text_area_context if text_area_context else ""
            st.session_state['learningState'] = True
            st.rerun()
        else:
            st.warning("请输入学习主题和学习目标！")

else:
    st.markdown("学习主题")
    st.markdown(st.session_state['textTopic'])
    st.markdown("学习目标")
    st.markdown(st.session_state['textGoals'])
    st.markdown("学习材料")
    st.markdown(st.session_state['textContext'])

    if learningService.get_learn_phase() == LearnPhase.INITIALED:
        if 'str_checkpoints' not in st.session_state:
            str_checkpoints = learningService.get_learning_checkpoints(
                st.session_state['textTopic'], st.session_state['textGoals'], st.session_state['textContext'])
            st.session_state['str_checkpoints'] = str_checkpoints
        if 'str_question' not in st.session_state:
            str_question = learningService.get_current_question()
            st.session_state['str_question'] = str_question
        st.markdown(st.session_state['str_checkpoints'])
        st.markdown(st.session_state['str_question'])
    elif learningService.get_learn_phase() == LearnPhase.LEARNING:
        st.session_state['str_question'] = learningService.get_current_question()
        st.markdown(st.session_state['str_checkpoints'])
        st.markdown(st.session_state['str_question'])
        answer = st.chat_input("请输入你的回答")  # todo: 如何在AI运行时禁止用户输入？
        if answer:
            (str_verify_result, str_teaching) = learningService.verify_user_answer(answer)
            st.markdown(str_verify_result)
            st.markdown(str_teaching)
            st.rerun()

# if st.button("隐藏输入1"):
#     input_placeholder1.empty()
    
    # if learningService.get_learn_phase() == LearnPhase.INITIALED:
    #     str_checkpoints = learningService.get_learning_checkpoints(
    #         "Anemia", ['Im medical student, i want to master the diagnosis of Anemia'], prompt)
    #     str_question = learningService.get_current_question()
    #     ai.write(str_checkpoints)
    #     ai.write(str_question)
    # elif learningService.get_learn_phase() == LearnPhase.LEARNING:
    #     (str_verify_result, str_teaching) = learningService.verify_user_answer(prompt)
    #     ai.write(str_verify_result)
    #     ai.write(str_teaching)
