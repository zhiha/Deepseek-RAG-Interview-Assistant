import os
import operator
import uuid
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict
from enum import Enum

from IPython.display import Image, display
from langchain_community.utils.math import cosine_similarity
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from semantic_chunkers import StatisticalChunker
from semantic_router.encoders import OpenAIEncoder
from langchain_deepseek import ChatDeepSeek

################################
# 数据结构定义
################################


class _Goals(BaseModel):
    """Structure for defining learning goals"""
    goals: str = Field(None, description="Learning goals")


class _LearningCheckpoint(BaseModel):
    """Structure for a single checkpoint"""
    description: str = Field(..., description="Main checkpoint description")
    criteria: List[str] = Field(..., description="List of success criteria")
    verification: str = Field(..., description="How to verify this checkpoint")


class _Checkpoints(BaseModel):
    """Main checkpoints container with index tracking"""
    checkpoints: List[_LearningCheckpoint] = Field(
        ...,
        description="List of checkpoints covering foundation, application, and mastery levels"
    )


class _SearchQuery(BaseModel):
    """Structure for search query collection"""
    search_queries: list = Field(
        None, description="Search queries for retrieval.")


class _LearningVerification(BaseModel):
    """Structure for verification results"""
    understanding_level: float = Field(
        ..., description="Understanding level of the student to the problem.")
    feedback: str
    suggestions: List[str]
    context_alignment: bool


class _FeynmanTeaching(BaseModel):
    """Structure for Feynman teaching method"""
    simplified_explanation: str
    key_concepts: List[str]
    analogies: List[str]


class _QuestionOutput(BaseModel):
    """Structure for question generation output"""
    question: str


class _InContext(BaseModel):
    """Structure for context verification"""
    is_in_context: str = Field(..., description="Yes or No")


class _LearningState(TypedDict):
    topic: str
    goals: List[_Goals]
    context: str
    context_chunks: Annotated[list, operator.add]
    context_key: str
    search_queries: _SearchQuery
    checkpoints: _Checkpoints
    verifications: _LearningVerification
    teachings: _FeynmanTeaching
    current_checkpoint: int
    current_question: _QuestionOutput
    current_answer: str


def format_checkpoints(event):
    """Return formatted checkpoints information as a string."""
    checkpoints = event.get('checkpoints', '')
    if not checkpoints:
        return ""

    result = []
    result.append("\n" + "=" * 80)
    result.append("🎯 LEARNING CHECKPOINTS OVERVIEW".center(80))
    result.append("=" * 80 + "\n")

    for i, checkpoint in enumerate(checkpoints.checkpoints, 1):
        result.append(f"📍 CHECKPOINT #{i}".center(80))
        result.append("─" * 80 + "\n")
        result.append("📝 Description:")
        result.append("─" * 40)
        words = checkpoint.description.split()
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= 70:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                result.append(f"  {' '.join(current_line)}")
                current_line = [word]
                current_length = len(word)

        if current_line:
            result.append(f"  {' '.join(current_line)}")
        result.append("")

        result.append("✅ Success Criteria:")
        result.append("─" * 40)
        for j, criterion in enumerate(checkpoint.criteria, 1):
            words = criterion.split()
            current_line = []
            current_length = 0
            first_line = True

            for word in words:
                if current_length + len(word) + 1 <= 66:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    if first_line:
                        result.append(f"  {j}. {' '.join(current_line)}")
                        first_line = False
                    else:
                        result.append(f"     {' '.join(current_line)}")
                    current_line = [word]
                    current_length = len(word)

            if current_line:
                if first_line:
                    result.append(f"  {j}. {' '.join(current_line)}")
                else:
                    result.append(f"     {' '.join(current_line)}")
        result.append("")

        result.append("🔍 Verification Method:")
        result.append("─" * 40)
        words = checkpoint.verification.split()
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= 70:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                result.append(f"  {' '.join(current_line)}")
                current_line = [word]
                current_length = len(word)

        if current_line:
            result.append(f"  {' '.join(current_line)}")
        result.append("")

        if i < len(checkpoints.checkpoints):
            result.append("~" * 80 + "\n")

    result.append("=" * 80 + "\n")

    return "\n".join(result)


def format_verification_results(event):
    """Format verification results into a string with improved formatting"""
    verifications = event.get('verifications', '')
    if not verifications:
        return ""

    result = []
    result.append("\n" + "=" * 50)
    result.append("📊 VERIFICATION RESULTS".center(50))
    result.append("=" * 50 + "\n")

    understanding = verifications.understanding_level
    bar_length = 20
    filled_length = int(understanding * bar_length)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)

    result.append(
        f"📈 Understanding Level: [{bar}] {understanding * 100:.1f}%\n")

    # Feedback section
    result.append("💡 Feedback:")
    result.append(f"{verifications.feedback}\n")

    # Suggestions section
    result.append("🎯 Suggestions:")
    for i, suggestion in enumerate(verifications.suggestions, 1):
        result.append(f"  {i}. {suggestion}")
    result.append("")

    # Context Alignment
    result.append("🔍 Context Alignment:")
    result.append(f"{verifications.context_alignment}\n")

    result.append("-" * 50 + "\n")

    return "\n".join(result)


def format_teaching_results(event):
    """Return formatted Feynman teaching results as a string."""
    teachings = event.get('teachings', '')
    if not teachings:
        return ""

    result = []
    result.append("\n" + "=" * 70)
    result.append("🎓 FEYNMAN TEACHING EXPLANATION".center(70))
    result.append("=" * 70 + "\n")

    # Simplified Explanation section
    result.append("📚 SIMPLIFIED EXPLANATION:")
    result.append("─" * 30)
    paragraphs = teachings.simplified_explanation.split('\n')

    for paragraph in paragraphs:
        words = paragraph.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= 60:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(' '.join(current_line))

        result.extend(lines)
        result.append("")

    # Key Concepts section
    result.append("💡 KEY CONCEPTS:")
    result.append("─" * 30)
    for i, concept in enumerate(teachings.key_concepts, 1):
        result.append(f"  {i}. {concept}")
    result.append("")

    # Analogies section
    result.append("🔄 ANALOGIES & EXAMPLES:")
    result.append("─" * 30)
    for i, analogy in enumerate(teachings.analogies, 1):
        result.append(f"  {i}. {analogy}")
    result.append("")

    result.append("=" * 70 + "\n")

    return "\n".join(result)


################################
# Prompts
################################
# 根据学习材料，生成学习检查点
_msg_generate_checkpoints = SystemMessage(content="""You will be given a learning topic title and learning objectives.
Your goal is to generate clear learning checkpoints that will help verify understanding and progress through the topic.
The output should be in the following dictionary structure:
checkpoint 
-> description (level checkpoint description)
-> criteria
-> verification (How to verify this checkpoint (Feynman Methods))
Requirements for each checkpoint:
- Description should be clear and concise
- Criteria should be specific and measurable (3-5 items)
- Verification method should be practical and appropriate for the level
- Verification will be checked by language model, so it must by in natural language
- All elements should align with the learning objectives
- Use action verbs and clear language
Ensure all checkpoints progress logically from foundation to mastery.
IMPORTANT - ANSWER ONLY 3 CHECKPOINTS""")

# 给定检查点，生成一条检索命令，用于查询该检查点的达成标准
_msg_generate_query = SystemMessage(content="""You will be given learning checkpoints for a topic.
Your goal is to generate search queries that will retrieve content matching each checkpoint's requirements from retrieval systems or web search.
Follow these steps:
1. Analyze each learning checkpoint carefully
2. For each checkpoint, generate ONE targeted search query that will retrieve:
   - Content for checkpoint verification
IMPORTANT: you should give me a structured answer, where the content string is in a single array, 
and the string is surrounded by quotation mark, i.e. format your answer like this: ["the content"]
""")

# 给定一个标准及其上下文，检查该标准是否被上下文回答
_msg_validate_context = SystemMessage(content="""You will be given a learning criteria and context.
Check if the the criteria could be answered using the context.
Always answer YES or NO""")

# 给定检查点、成功标准、检测方法，生成一个问题，能够用于测验学生是否达成该检查点的要求
_msg_generate_question_ = SystemMessage(content="""You will be given a checkpoint description, success criteria, and verification method.
Your goal is to generate an appropriate question that aligns with the checkpoint's verification requirements.
The question should:
1. Follow the specified verification method
2. Cover all success criteria
3. Encourage demonstration of understanding
4. Be clear and specific
Output should be a single, well-formulated question that effectively tests the checkpoint's learning objectives.""")

# 给定学生的回答，分析其是否到达了检查点的标准
_msg_verify_answer = SystemMessage(content="""You will be given a student's answer, question, checkpoint details, and relevant context.
Your goal is to analyze the answer against the checkpoint criteria and provided context.
Analyze considering:
1. Alignment with verification method specified
2. Coverage of all success criteria
3. Use of relevant concepts from context
4. Depth and accuracy of understanding
Output should include:
- understanding_level: float between 0 and 1
- feedback: detailed explanation of the assessment
- suggestions: list of specific improvements
- context_alignment: boolean indicating if the answer aligns with provided context""")

# 给定检测结果、学习的上下文，给学生提出还需加强的知识点（用费曼学习法）
_msg_teach_concept = SystemMessage(content="""You will be given verification results, checkpoint criteria, and learning context.
Your goal is to create a Feynman-style teaching explanation for concepts that need reinforcement.
The explanation should include:
1. Simplified explanation without technical jargon
2. Concrete, relatable analogies
3. Key concepts to remember
Output should follow the Feynman technique:
- simplified_explanation: clear, jargon-free explanation
- key_concepts: list of essential points
- analogies: list of relevant, concrete comparisons
Focus on making complex ideas accessible and memorable.""")


class _ContextStore:
    """Store for managing context chunks and their embeddings in memory.

    A class that provides storage and retrieval of context data using an in-memory store.
    Each context entry consists of context chunks and their corresponding embeddings.
    """

    def __init__(self):
        """Initialize ContextStore with an empty in-memory store."""
        self.store = InMemoryStore()

    def save_context(self, context_chunks: list, embeddings: list, key: str = None):
        """Save context chunks and their embeddings to the store.

        Args:
            context_chunks (list): List of context chunk objects
            embeddings (list): List of corresponding embeddings for the chunks
            key (str, optional): Custom key for storing the context. Defaults to None,
                               in which case a UUID is generated.

        Returns:
            str: The key used to store the context
        """
        namespace = ("context",)

        if key is None:
            key = str(uuid.uuid4())

        value = {
            "chunks": context_chunks,
            "embeddings": embeddings
        }

        self.store.put(namespace, key, value)
        return key

    def get_context(self, context_key: str):
        """Retrieve context data from the store using a key.

        Args:
            context_key (str): The key used to store the context

        Returns:
            dict: The stored context value containing chunks and embeddings
        """
        namespace = ("context",)
        memory = self.store.get(namespace, context_key)
        return memory.value


class LearnPhase(Enum):
    INITIALED = 1
    CHECKPOINTS_GENERATED = 2
    LEARNING = 3
    FINISH = 4


class FeynmanLearning:
    def __init__(self):
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            # base_url = os.getenv("OPENAI_API_BASE"),
            # api_key = os.getenv("OPENAI_API_KEY")
        )

        self.encoder = OpenAIEncoder(
            name="text-embedding-3-large",
            openai_base_url=os.getenv("OPENAI_API_BASE"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        self.tavily_search = TavilySearchResults(
            max_results=3
        )

        self.searcher = StateGraph(_LearningState)
        self.memory = MemorySaver()
        self.context_store = _ContextStore()

        # 添加节点
        self.searcher.add_node("generate_query", self.__generate_query)
        self.searcher.add_node("search_web", self.__search_web)
        self.searcher.add_node("chunk_context", self.__chunk_context)
        self.searcher.add_node("context_validation", self.__context_validation)
        self.searcher.add_node("generate_checkpoints",
                               self.__generate_checkpoints)
        self.searcher.add_node("generate_question", self.__generate_question)
        self.searcher.add_node("next_checkpoint", self.__next_checkpoint)
        self.searcher.add_node("user_answer", self.__user_answer)
        self.searcher.add_node("verify_answer", self.__verify_answer)
        self.searcher.add_node("teach_concept", self.__teach_concept)

        # Flow
        self.searcher.add_edge(START, "generate_checkpoints")
        self.searcher.add_conditional_edges('generate_checkpoints', self.__route_context,
                                            ['chunk_context', 'generate_query'])
        self.searcher.add_edge("generate_query", "search_web")
        self.searcher.add_edge("search_web", "generate_question")
        self.searcher.add_edge("chunk_context", 'context_validation')
        self.searcher.add_conditional_edges('context_validation', self.__route_search,
                                            ['search_web', 'generate_question'])

        self.searcher.add_edge("generate_question", "user_answer")
        self.searcher.add_edge("user_answer", "verify_answer")
        self.searcher.add_conditional_edges(
            "verify_answer",
            self.__route_verification,
            {
                "next_checkpoint": "next_checkpoint",
                "teach_concept": "teach_concept",
                END: END
            }
        )

        self.searcher.add_conditional_edges(
            "teach_concept",
            self.__route_teaching,
            {
                "next_checkpoint": "next_checkpoint",
                END: END
            }
        )
        self.searcher.add_edge("next_checkpoint", "generate_question")

        self.graph = self.searcher.compile(
            interrupt_after=["generate_checkpoints"],
            interrupt_before=["user_answer"],
            checkpointer=self.memory
        )

        self.config = {"configurable": {"thread_id": "20"}}
        self.learn_phase = LearnPhase.INITIALED

    def get_learning_checkpoints(self, topic: str, goals: list[str], context=None):
        initial_input = {
            "topic": topic,
            'goals': goals,
            'context': context,
            'current_checkpoint': 0
        }
        for event in self.graph.stream(initial_input, self.config, stream_mode="values"):
            str_formatted_checkpoints = format_checkpoints(event)
            self.graph.update_state(self.config, {"checkpoints": event.get(
                'checkpoints', '')}, as_node="generate_checkpoints")
        return str_formatted_checkpoints

    def get_current_question(self):
        for event in self.graph.stream(None, self.config, stream_mode="values"):
            current_question = event.get('current_question', '')
        return current_question

    def verify_user_answer(self, current_answer: str):
        self.graph.update_state(
            self.config, {"current_answer": current_answer}, as_node="user_answer")
        for event in self.graph.stream(None, self.config, stream_mode="values"):
            print(self.graph.get_state(self.config).next)
        str_verification_results = format_verification_results(event)
        str_teaching_results = format_teaching_results(event)
        return (str_verification_results, str_teaching_results)

    def get_learn_phase(self):
        return self.learn_phase

    ################################
    # 节点定义
    ################################
    def __generate_checkpoints(self, state: _LearningState):
        """Creates learning checkpoints based on given topic and goals."""
        structured_llm = self.llm.with_structured_output(_Checkpoints)
        messages = [
            _msg_generate_checkpoints,
            HumanMessage(
                content=f"Topic: {state['topic']}\nGoals: {', '.join(str(goal) for goal in state['goals'])}"),
        ]
        checkpoints = structured_llm.invoke(messages)
        self.learn_phase = LearnPhase.CHECKPOINTS_GENERATED
        return {"checkpoints": checkpoints}

    def __chunk_context(self, state: _LearningState):
        """Splits context into manageable chunks and generates their embeddings."""
        def extract_content_from_chunks(chunks):
            content = []
            for chunk in chunks:
                if hasattr(chunk, 'splits') and chunk.splits:
                    chunk_content = ' '.join(chunk.splits)
                    content.append(chunk_content)
            return '\n'.join(content)

        chunker = StatisticalChunker(
            encoder=self.encoder,
            min_split_tokens=128,
            max_split_tokens=512
        )
        chunks = chunker([state['context']])
        content = []

        for chunk in chunks:
            content.append(extract_content_from_chunks(chunk))

        chunk_embeddings = self.embeddings.embed_documents(content)
        context_key = self.context_store.save_context(
            content,
            chunk_embeddings,
            key=state.get('context_key')
        )
        return {"context_chunks": content, "context_key": context_key}

    def __context_validation(self, state: _LearningState):
        """Validates context coverage against checkpoint criteria using stored embeddings."""
        def generate_checkpoint_message(checks: List[_LearningCheckpoint]) -> HumanMessage:
            """Generate a formatted message for learning checkpoints that need context.

            Args:
                checks (List[LearningCheckpoint]): List of learning checkpoint objects

            Returns:
                HumanMessage: Formatted message containing checkpoint descriptions, criteria and 
                            verification methods, ready for context search
            """
            formatted_checks = []

            for check in checks:
                checkpoint_text = f"""
                Description: {check.description}
                Success Criteria:
                {chr(10).join(f'- {criterion}' for criterion in check.criteria)}
                Verification Method: {check.verification}
                """    # chr(10)即'\n'
                formatted_checks.append(checkpoint_text)

            all_checks = "\n---\n".join(formatted_checks)

            checkpoints_message = HumanMessage(content=f"""The following learning checkpoints need additional context:
                {all_checks}
                
                Please generate search queries to find relevant information.""")

            return checkpoints_message

        context = self.context_store.get_context(state['context_key'])
        chunks = context['chunks']
        chunk_embeddings = context['embeddings']

        checks = []
        structured_llm = self.llm.with_structured_output(_InContext)

        for checkpoint in state['checkpoints'].checkpoints:
            query = self.embeddings.embed_query(checkpoint.verification)

            similarities = cosine_similarity([query], chunk_embeddings)[0]
            top_3_indices = sorted(range(len(similarities)),
                                   key=lambda i: similarities[i],
                                   reverse=True)[:3]
            relevant_chunks = [chunks[i] for i in top_3_indices]

            messages = [
                _msg_validate_context,
                HumanMessage(content=f"""
                Criteria:
                {chr(10).join(f"- {c}" for c in checkpoint.criteria)}
                
                Context:
                {chr(10).join(relevant_chunks)}
                """)
            ]

            response = structured_llm.invoke(messages)
            if response.is_in_context.lower() == "no":
                checks.append(checkpoint)

        if checks:
            structured_llm = self.llm.with_structured_output(_SearchQuery)
            checkpoints_message = generate_checkpoint_message(checks)

            messages = [_msg_generate_query, checkpoints_message]
            search_queries = structured_llm.invoke(messages)
            return {"search_queries": search_queries}

        return {"search_queries": None}

    def __generate_query(self, state: _LearningState):
        """Generates search queries based on learning checkpoints from current state."""
        def format_checkpoints_as_message(checkpoints: _Checkpoints) -> str:
            message = "Here are the learning checkpoints:\n\n"
            for i, checkpoint in enumerate(checkpoints.checkpoints, 1):
                message += f"Checkpoint {i}:\n"
                message += f"Description: {checkpoint.description}\n"
                message += "Success Criteria:\n"
                for criterion in checkpoint.criteria:
                    message += f"- {criterion}\n"
            return message

        structured_llm = self.llm.with_structured_output(_SearchQuery)
        checkpoints_message = HumanMessage(
            content=format_checkpoints_as_message(state['checkpoints']))
        messages = [_msg_generate_query, checkpoints_message]
        search_queries = structured_llm.invoke(messages)
        return {"search_queries": search_queries}

    def __search_web(self, state: _LearningState):
        """Retrieves and processes web search results based on search queries."""
        search_queries = state["search_queries"].search_queries

        all_search_docs = []
        for query in search_queries:
            search_docs = self.tavily_search.invoke(query)
            all_search_docs.extend(search_docs)

        formatted_search_docs = [
            f'Context: {doc["content"]}\n Source: {doc["url"]}\n'
            for doc in all_search_docs
        ]

        chunk_embeddings = self.embeddings.embed_documents(
            formatted_search_docs)
        context_key = self.context_store.save_context(
            formatted_search_docs,
            chunk_embeddings,
            key=state.get('context_key')
        )  # todo

        return {"context_chunks": formatted_search_docs}

    def __generate_question(self, state: _LearningState):
        """Generates assessment questions based on current checkpoint verification requirements."""
        structured_llm = self.llm.with_structured_output(_QuestionOutput)
        current_checkpoint = state['current_checkpoint']
        checkpoint_info = state['checkpoints'].checkpoints[current_checkpoint]

        messages = [
            _msg_generate_question_,
            HumanMessage(content=f"""
            Checkpoint Description: {checkpoint_info.description}
            Success Criteria:
            {chr(10).join(f"- {c}" for c in checkpoint_info.criteria)}
            Verification Method: {checkpoint_info.verification}
            
            Generate an appropriate verification question.""")
        ]

        question_output = structured_llm.invoke(messages)
        self.learn_phase = LearnPhase.LEARNING
        return {"current_question": question_output.question}

    def __user_answer(self, state: _LearningState):
        """Placeholder for handling user's answer input."""
        pass

    def __verify_answer(self, state: _LearningState):
        """Evaluates user answers against checkpoint criteria using relevant context chunks."""
        structured_llm = self.llm.with_structured_output(_LearningVerification)
        current_checkpoint = state['current_checkpoint']
        checkpoint_info = state['checkpoints'].checkpoints[current_checkpoint]

        context = self.context_store.get_context(state['context_key'])
        chunks = context['chunks']
        chunk_embeddings = context['embeddings']

        query = self.embeddings.embed_query(checkpoint_info.verification)

        similarities = cosine_similarity([query], chunk_embeddings)[0]
        top_3_indices = sorted(range(len(similarities)),
                               key=lambda i: similarities[i],
                               reverse=True)[:3]
        relevant_chunks = [chunks[i] for i in top_3_indices]

        messages = [
            _msg_verify_answer,
            HumanMessage(content=f"""
            Question: {state['current_question']}
            Answer: {state['current_answer']}
            
            Checkpoint Description: {checkpoint_info.description}
            Success Criteria:
            {chr(10).join(f"- {c}" for c in checkpoint_info.criteria)}
            Verification Method: {checkpoint_info.verification}
            
            Context:
            {chr(10).join(relevant_chunks)}
            
            Assess the answer.""")
        ]

        verification = structured_llm.invoke(messages)
        return {"verifications": verification}

    def __teach_concept(self, state: _LearningState):
        """Creates simplified Feynman-style explanations for concepts that need reinforcement."""
        structured_llm = self.llm.with_structured_output(_FeynmanTeaching)
        current_checkpoint = state['current_checkpoint']
        checkpoint_info = state['checkpoints'].checkpoints[current_checkpoint]

        messages = [
            _msg_teach_concept,
            HumanMessage(content=f"""
            Criteria: {checkpoint_info.criteria}
            Verification: {state['verifications']}
            
            Context:
            {state['context_chunks']}
            
            Create a Feynman teaching explanation.""")
        ]

        teaching = structured_llm.invoke(messages)
        return {"teachings": teaching}

    def __next_checkpoint(self, state: _LearningState):
        """Advances to the next checkpoint in the learning sequence."""
        current_checkpoint = state['current_checkpoint'] + 1
        return {'current_checkpoint': current_checkpoint}

    ################################
    # Routing Logic
    ################################
    def __route_context(self, state: _LearningState):
        """Determines whether to process existing context or generate new search queries."""
        if state.get("context"):
            return 'chunk_context'
        return 'generate_query'

    def __route_verification(self, state: _LearningState):
        """Determines next step based on verification results and checkpoint progress."""
        current_checkpoint = state['current_checkpoint']

        if state['verifications'].understanding_level < 0.7:
            return 'teach_concept'

        if current_checkpoint + 1 < len(state['checkpoints'].checkpoints):
            return 'next_checkpoint'

        return END

    def __route_teaching(self, state: _LearningState):
        """Routes to next checkpoint or ends session after teaching intervention."""
        current_checkpoint = state['current_checkpoint']
        if current_checkpoint + 1 < len(state['checkpoints'].checkpoints):
            return 'next_checkpoint'
        return END

    def __route_search(self, state: _LearningState):
        """Directs flow between question generation and web search based on query status."""
        if state['search_queries'] is None:
            return "generate_question"
        return "search_web"
