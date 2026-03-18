from openai import OpenAI
from math import comb
import os
import datetime
import numpy as np
import pickle
import pprint

from pipeline.utils import read_file, pretty_print
from config import *

import logging
logger = logging.getLogger(__name__)

class BaseEvaluator:
    """Base class for LLM-as-judge evaluators
    Provides prompt loading, placeholder-based prompt augmentation, and a
    thin wrapper around the OpenAI Chat Completions API.  Subclasses define
    the specific evaluation logic
    """

    def __init__(self, model=EVAL_MODEL, prompt_path = '', temp=0.0):
        """
        Args:
            model: OpenAI model used for LLM-as-judge evaluation.
            prompt_path: Path to the prompt template file.
            temp: Temperature for judge responses.
        """
        self.model = model
        self.prompt_path = prompt_path
        self.base_prompt: str
        self.llm = OpenAI()
        self.temp = temp
        self._setup()

    def load_prompt(self, prompt_path = ''):
        if not prompt_path:
            prompt_path = self.prompt_path
        self.base_prompt = read_file(prompt_path)

    def augment_prompt(self, prompt:str, augmentations:dict[str,str]):
        """Replace <PLACEHOLDER> tokens in prompt with desired values."""
        for placeholder, value in augmentations.items():
            prompt = prompt.replace(placeholder, value)
        return prompt
    
    def call_llm(self, messages):
        """Send messages to the judge LLM and return the response text"""
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=1000,
            temperature=self.temp,
            stream = False)
        return response.choices[0].message.content
    
    def _setup(self):
        self.load_prompt()


class BaseAnalyer(BaseEvaluator):

    def __init__(self, model, prompt_path, temp, verbose):
        
        super().__init__(model, prompt_path, temp)
        self.verbose = verbose

    def load_conversation(self, path)->list[dict[str,str]]:
        data = eval(read_file(path))
        return data

    def consolidate_conversation_metrics(self, evaluations:list[dict[str,int]])->dict[str,int]:
        """Aggregate metric values across conversations"""
        metrics = {}
        for evaluation in evaluations:
            for metric, value in evaluation.items():
                metrics[metric] = metrics.get(metric, 0 if metric!='summary' else '') + value
        if 'critical_field_accuracy' in metrics.keys():
            metrics['critical_field_accuracy'] /= len(evaluations)
        return metrics
    
    def analyze_conversations(self, conversations=None, path='', limit=1000):
        """Analyze conversations and return per-conversation and consolidated metrics."""
        conversation_metrics = []
        if conversations is None:
            conversations = self.load_conversation(path)
        conversations = conversations[:limit]
        for conversation in conversations:
            conversation_metrics.append(self.analyze_conversation(conversation))
        return conversation_metrics, self.consolidate_conversation_metrics(conversation_metrics)
    
    def _get_metrics_str(self, desc_path:str):
        """Load a metrics description file and format it as a JSON-like string for prompt injection."""
        metrics_desc:dict[str,str] = eval(read_file(desc_path))
        metrics_str = '{\n'
        for metric in metrics_desc.keys():
            metrics_str += f'"{metric}": "{metrics_desc[metric]}"' + ',\n'
        metrics_str += '\n}'
        return metrics_str
    
    def _get_history(self, history):
        history_str = ''
        for idx, msg in enumerate(history):
            history_str += f'{idx+1}) {self._get_message(msg)}' + '\n'
        return history_str

    def _get_message(self, message):
        return f"{message['role']}: {message['content']}"
    
    def analyze_conversation(self, conversation):
        raise NotImplementedError


class MessageAnalyzer(BaseAnalyer):
    """Evaluates individual messages within a conversation using an LLM judge.
    For each message the judge is given the conversation history up to that
    point and asked to score a set of metrics (segregated agent and user metrics)
    """

    def __init__(self, critical_fields=[], model=EVAL_MODEL, prompt_path = 'Data/Prompts/MM-MessageAnalyzer.txt', temp=0.0, verbose=False):

        super().__init__(model, prompt_path, temp, verbose)
        self.agent_metrics_str = self._get_metrics_str('Data/Prompts/MM-AgentMetricsDescription.txt')
        self.user_metrics_str = self._get_metrics_str('Data/Prompts/MM-UserMetricsDescription.txt')
        self.agent_metrics_str = self.agent_metrics_str.replace('<CRITICAL_FIELDS>', str(critical_fields))
        logger.info("USING\n Agent Metrics: %s\n User Metrics: %s", self.agent_metrics_str, self.user_metrics_str)

    def analyze_conversation(self, conversation):
        """Score every message in conversation and return consolidated metrics."""
        evaluations, count_agent_messages, count_user_messages = self._conversation_evaluation(conversation)
        metrics = self.consolidate_conversation_metrics(evaluations)
        metrics['total_agent_messages'] = count_agent_messages
        metrics['total_user_messages'] = count_user_messages
        logger.info('MESSAGE ANALYSIS')
        pretty_print(conversation)
        logger.info("%s", metrics)
        return metrics
    
    def _conversation_evaluation(self, conversation:list[dict[str,str]])->list[dict[str,int]]:
        evaluation = []
        agent_msg_counter, user_msg_counter = 0, 0
        for idx, message in enumerate(conversation):
            if message['role'] in ('assistant',):
                metrics = self.evaluate_message(conversation[:idx], message, self.agent_metrics_str)
                agent_msg_counter += 1
            elif message['role'] in ('user',):
                metrics = self.evaluate_message(conversation[:idx], message, self.user_metrics_str)
                user_msg_counter += 1
            if metrics:
                evaluation.append(metrics)
        return evaluation, agent_msg_counter, user_msg_counter
    
    def evaluate_message(self, history, message, metrics_str, retry_counter=5):
        """Ask the LLM judge to score a single message given prior history

        Args:
            history: Conversation messages prior to the target message
            message: Current message to evaluate
            metrics_str: JSON-like string describing which metrics to score
            retry_counter: Remaining retry attempts on parse failure

        Returns:
            Dict mapping metric names to integer scores, or None on failure
        """
        prompt = self.base_prompt
        augment = {
            '<HISTORY>': self._get_history(history),
            '<MESSAGE>': self._get_message(message),
            '<METRICS>': metrics_str
        }
        prompt = self.augment_prompt(prompt, augment)
        data = [
            {"role": "system", "content": "You are a helpful assistant designed to analyse conversations"},
            {"role": "user", "content": prompt}
        ]
        response = self.call_llm(data)
        response = response.lstrip('```python').rstrip('```')
        try:
            return eval(response)
        except Exception as e:
            if retry_counter>0:
                logger.warning('Parsing Failed. Retrying...')
                return self.evaluate_message(history, message, metrics_str, retry_counter-1)
            else:
                logger.error('Retry limit reached. Could not parse: %s', response)


class ConversationAnalyzer(BaseAnalyer):
    """Evaluates entire conversations by comparing actual (voice) vs. ideal (text) transcripts"""

    def __init__(self, model=EVAL_MODEL, prompt_path = 'Data/Prompts/MM-ConversationAnalyzer.txt', temp=0.0, verbose=False):

        super().__init__(model, prompt_path, temp, verbose)
        self.conversation_metrics_str = self._get_metrics_str('Data/Prompts/MM-ConversationMetricsDescription.txt')
        logger.info("USING\n Conversation Metrics: %s", self.conversation_metrics_str)

    def evaluate_conversation(self, actual_conversation, ideal_conversation, metrics_str, retry_counter=5)->dict:
        """Ask the LLM judge to compare actual vs. ideal conversations and return scored metrics.
        Retries up to retry_counter times on parse failure.
        """
        prompt = self.base_prompt
        augment = {
            '<CONVERSATION>': self._get_history(actual_conversation, ideal_conversation),
            '<METRICS>': metrics_str
        }
        prompt = self.augment_prompt(prompt, augment)
        data = [
            {"role": "system", "content": "You are a helpful assistant designed to analyse conversations"},
            {"role": "user", "content": prompt}
        ]
        response = self.call_llm(data)
        try:
            return eval(response)
        except Exception as e:
            if retry_counter>0:
                logger.warning('Parsing Failed. Retrying...')
                return self.evaluate_conversation(actual_conversation, ideal_conversation, metrics_str, retry_counter-1)
            else:
                logger.error('Retry limit reached. Could not parse: %s', response)
    
    def _get_msg_count(self, conversation):
        agent_count, user_count = 0,0
        for msg in conversation:
            if msg['role'] == 'assistant':
                agent_count += 1
            elif msg['role'] == 'user':
                user_count += 1
        return agent_count, user_count
    
    def analyze_conversation(self, conversation:tuple[list[dict[str,str]]]):
        
        actual_conversation, ideal_conversation = conversation
        evaluations = self.evaluate_conversation(actual_conversation, ideal_conversation, self.conversation_metrics_str)
        evaluations['agent_msg_counts'], evaluations['user_msg_counts'] = self._get_msg_count(actual_conversation) #agent, user
        logger.info('CONVERSATION ANALYSIS')
        logger.debug("%s", conversation)
        logger.info("%s", evaluations)
        return evaluations
    
    def load_conversation(self, dir_path:str) -> list[tuple[list[dict[str, str]]]]:
        
        actual_conversation = super().load_conversation(dir_path+'/implementation_actual.txt')
        ideal_conversation = super().load_conversation(dir_path+'/implementation_llm.txt')
        return [(actual_conversation, ideal_conversation)]
    
    # override base method
    def consolidate_conversation_metrics(self, evaluations:list[dict[str,int]])->dict[str,int]:
        metrics = super().consolidate_conversation_metrics(evaluations)
        metrics['ARGA'] = self.ARGA(metrics)
        metrics['pass@1'] = self.passAtK(metrics)
        metrics['pass^1'] = self.passPowerK(metrics)
        return metrics
    
    def _get_history(self, actual_conversation, ideal_conversation):
        
        assert len(actual_conversation)==len(ideal_conversation), f"Actual has {len(actual_conversation)} messages, Ideal has {len(ideal_conversation)} messages"
        history_str = ''
        for idx in range(len(actual_conversation)):
            actual_msg, ideal_msg = actual_conversation[idx], ideal_conversation[idx]
            history_str += f"{idx}) [ideal] {ideal_msg['role'].upper()}: {ideal_msg['content']}\n"
            history_str += f"{idx}) [actual] {actual_msg['role'].upper()}: {actual_msg['content']}\n"
        return history_str

    def passAtK(self, metrics, k=1):
        total, correct = metrics['tasks_identified'], metrics['tasks_succeeded']
        score = 1 - (comb(total-correct, k)/comb(total, k)) #TODO: Optimize computation
        return score
    
    def passPowerK(self, metrics, k=1):
        total, correct = metrics['tasks_identified'], metrics['tasks_succeeded']
        score = (correct/total)**k
        return score
            
    def ARGA(self, metrics):
        
        prob_asr_error = metrics['tasks_with_agent_understanding_errors']/metrics['tasks_identified']
        prob_success_and_error = metrics['tasks_with_agent_understanding_errors_succeeded']/metrics['tasks_identified']
        try:
            arga_score = metrics['tasks_with_agent_understanding_errors_succeeded']/metrics['tasks_with_agent_understanding_errors']
        except ZeroDivisionError:
            arga_score = f"{metrics['tasks_with_agent_understanding_errors_succeeded']}/{metrics['tasks_with_agent_understanding_errors']}"
        return arga_score

    def turnOverhead(self, voice_metrics, text_metrics):

        turns_voice, turns_text = voice_metrics['agent_msg_counts'], text_metrics['agent_msg_counts']
        return (turns_voice/turns_text)-1


class MultiModalTauEval:
    """Top-level evaluator that orchestrates message-level and conversation-level analysis
    Loads ground-truth, LLM-intended, and ASR-actual conversation traces, runs
    MessageAnalyzer and ConversationAnalyzer and produces a consolidated report with per-conversation statistics
    """

    def __init__(self, critical_fields=[], metrics_to_skip = [], model=EVAL_MODEL, temp=0.0, verbose=False):
        """
        Args:
            critical_fields: Fields whose accuracy is measured at the message level
            metrics_to_skip: Metric names to exclude from the evaluation prompts
            model: LLM model for LLM-as-judge evaluation
            temp: Temperature for the judge
            verbose: Enable verbose logging in sub-analyzers
        """
        self.messageAnalyzer = MessageAnalyzer(critical_fields=critical_fields, model=model, temp=temp, verbose=verbose)
        self.conversationAnalyzer = ConversationAnalyzer(model=model, temp=temp, verbose=verbose)
        self.metrics_to_skip = metrics_to_skip
        self._setup()

    def _setup(self):
        
        def check_metric_str(s):
            d = eval(s)
            for key in d.keys():
                if key in self.metrics_to_skip:
                    del d[key]
            return str(d).replace(',', ',\n')
        
        self.messageAnalyzer.agent_metrics_str = check_metric_str(self.messageAnalyzer.agent_metrics_str)
        self.messageAnalyzer.user_metrics_str = check_metric_str(self.messageAnalyzer.user_metrics_str)
        self.conversationAnalyzer.conversation_metrics_str = check_metric_str(self.conversationAnalyzer.conversation_metrics_str)

    def messageMetrics(self, folder_path):
        ground, actual, ideal = self.load(folder_path=folder_path)
        return self._messageMetrics((actual[0], ideal[0]))
    
    def conversationMetrics(self, folder_path):
        ground, actual, ideal = self.load(folder_path=folder_path)
        return self._conversationMetrics((actual[0], ideal[0]))
    
    def _messageMetrics(self, conversations):
        
        if isinstance(conversations, list):
            per_conv, consolidated = self.messageAnalyzer.analyze_conversations(conversations=[x[0] for x in conversations], limit = len(conversations))
            return per_conv, consolidated
        elif isinstance(conversations, tuple):
            per_conv, consolidated = self.messageAnalyzer.analyze_conversations(conversations=[conversations[0]], limit = len(conversations))
            return per_conv, consolidated
    
    def _conversationMetrics(self, conversations):
        
        if isinstance(conversations, list):
            per_conv, consolidated = self.conversationAnalyzer.analyze_conversations(conversations=conversations, limit = len(conversations))
            return per_conv, consolidated
        if isinstance(conversations, tuple):
            per_conv, consolidated = self.conversationAnalyzer.analyze_conversations(conversations=[conversations], limit = len(conversations))
            return per_conv, consolidated
    
    def _eval(self, conversations, ground_conversations):
        """Run both message and conversation evaluation, then generate the report

        Args:
            conversations: List of (actual, ideal) conversation pairs
            ground_conversations: List of (ground, ground) pairs for text agent metrics

        Returns:
            Consolidated metrics dict.
        """
        msgPerConv, msgMetrics = self._messageMetrics(conversations)
        convPerConv, convMetrics = self._conversationMetrics(conversations)
        perConvText, text_metrics = self._conversationMetrics(ground_conversations)
        perConv = [{**msgPerConv[idx], **convPerConv[idx]} for idx in range(len(conversations))]
        metrics = {**msgMetrics, **convMetrics}
        self.report(metrics, text_metrics, perConv, perConvText)
        return metrics

    def eval(self, dir_path='', folder_path='', prefix = ''):
        """Load conversations and run full evaluation (message + conversation level).

        Args:
            dir_path: Directory containing multiple conversation folders.
            folder_path: Single conversation folder path.
            prefix: Filter folders by name prefix.

        Returns:
            Consolidated metrics dict.
        """
        ground, actual, ideal = self.load(dir_path=dir_path, folder_path=folder_path, prefix=prefix)
        conversations = [(actual[idx], ideal[idx]) for idx in range(len(actual))]
        ground = [(x, x) for x in ground]  # ground, ground augmented to be compatible with rest of the code. Essentially ideal and actual are both same in this case
        return self._eval(conversations, ground)


    def _load_file(self, file_path, l = None):
            
        if l is None:
            l = []
        try:
            conv = self.messageAnalyzer.load_conversation(file_path)
            l.append(conv)
        except FileNotFoundError:
            logger.warning('Invalid Path: %s', file_path)
        return l

    def _load_folder(self, folder_path, ground = [], actual = [], ideal = []):

        self._load_file(f"{folder_path}/ground.txt", ground)
        self._load_file(f"{folder_path}/implementation_actual.txt", actual)
        self._load_file(f"{folder_path}/implementation_llm.txt", ideal)
        return ground, actual, ideal
    
    def load(self, dir_path = '', folder_path = '', file_path = '', prefix=''):
        """Flexible loader: accepts a single file, a folder, or a directory of folders.

        Args:
            dir_path: Parent directory containing multiple run folders
            folder_path: Single run folder
            file_path: Single conversation file
            prefix: When using dir_path, only load folders matching this prefix.

        Returns:
            Tuple of (ground, actual, ideal) conversation lists.
        """
        if file_path:
            return self._load_file(file_path)
        
        if folder_path:
            return self._load_folder(folder_path)

        if dir_path:
            ground, actual, ideal = [], [], []
            folders = sorted([x for x in os.listdir(dir_path) if x.startswith(prefix)])
            for folder in folders:
                if folder.endswith('.log'):
                    continue
                logger.info('Loading Folder %s', folder)
                self._load_folder(f"{dir_path}/{folder}", ground, actual, ideal)
            return ground, actual, ideal

    def report(self, metrics:dict, text_metrics:dict, perConv = [], perConvText = []):
        """Generate the consolidated evaluation report, log it, and save to a pickle file"""
        save_file = {}
        save_file['metrics'] = metrics
        save_file['text_metrics'] = text_metrics
        save_file['perConv'] = perConv
        save_file['perConvText'] = perConvText

        
        fullReport = self._report(metrics, text_metrics)
        save_file['fullReport'] = fullReport
        logger.info('Consolidated Report')
        logger.info("%s", pprint.pformat(fullReport))
        logger.info('TEXT METRICS')
        logger.info("%s", pprint.pformat(text_metrics))

        reports = []
        for idx in range(len(perConv)):
            logger.info('CONVERSATION %d', idx)
            reports.append(self._report(perConv[idx], perConvText[idx]))
        save_file['reports'] = reports

        stats = self._get_stats(reports)
        save_file['stats'] = stats
        logger.info("Stats\n %s", pprint.pformat(stats))
        logger.info("Per Conversation Report\n  %s", pprint.pformat(perConv))
        logger.info("Per Conversation Text Report\n %s", pprint.pformat(perConvText))

        self._save(save_file, 'Evals')

    def _report(self, metrics, text_metrics):
        def div(a, b):
            return a/b if b!=0 else f'{a}/{b}'
        
        report = {}
        report['Raw Report'] = metrics
        report["Clarification"] = f"\n\tTrue Positives: {metrics['clarification_true_positives']}\n\tFalse Positives: {metrics['clarification_asked']-metrics['clarification_true_positives']}\n\tTrue Negatives: {metrics['total_agent_messages']+metrics['clarification_true_positives']-metrics['clarification_asked']-metrics['clarification_required']}\n\tFalse Negatives: {metrics['clarification_required'] - metrics['clarification_true_positives']}" #(TP+FN)=req + (TP+FP)=ask
        report['Clarification Precision'] = div(metrics['clarification_true_positives'],metrics['clarification_asked'])
        report['Clarification Recall'] =  div(metrics['clarification_true_positives'],metrics['clarification_required'])
        report["Safety"] = f"\n\tTrue Positives: {metrics['confirmation_true_positives']}\n\tFalse Positives: {metrics['confirmation_asked']-metrics['confirmation_true_positives']}\n\tTrue Negatives: {metrics['total_agent_messages']+metrics['confirmation_true_positives']-metrics['confirmation_asked']-metrics['confirmation_required']}\n\tFalse Negatives: {metrics['confirmation_required'] - metrics['confirmation_true_positives']}" #(TP+FN)=req + (TP+FP)=ask
        report['Safety Precision'] = div(metrics['confirmation_true_positives'],metrics['confirmation_asked'])
        report['Safety Recall'] = div(metrics['confirmation_true_positives'],metrics['confirmation_required'])
        report['Critical field Accuracy'] = metrics['critical_field_accuracy']
        report['Turn Efficiency'] =  div(metrics['message_necessity'],metrics['total_agent_messages'])
        report['Turn Overhead'] = self.conversationAnalyzer.turnOverhead(metrics, text_metrics)
        report['Error Recovery Rate'] =  div(metrics['error_identified'],metrics['error_committed'])
        report['Recovery Turn Count'] =  div(metrics['error_recovered'],metrics['error_identified'])
        report['User Effort Score'] = div(metrics['user_effort'], metrics['total_user_messages'])
        report['Voice pass@1'] =  self.conversationAnalyzer.passAtK(metrics)
        report['Voice pass^1'] = self.conversationAnalyzer.passPowerK(metrics)
        report['Text pass^1'] = self.conversationAnalyzer.passPowerK(text_metrics)
        report['ARGA Score'] = self.conversationAnalyzer.ARGA(metrics)
        report['Modality Robustness Score'] = div(self.conversationAnalyzer.passPowerK(metrics), self.conversationAnalyzer.passPowerK(text_metrics))

        return report

    def _get_stats(self, reports):
        keys = reports[0].keys()
        stats = {}
        for metric in keys:
            if metric in ('Clarification', 'Safety', 'Raw Report'):
                continue
            metrics = np.array([report[metric] if not isinstance(report[metric], str) else np.nan for report in reports])
            stats[metric] = f"\n\tMean:{np.nanmean(metrics)}\n\tMedian:{np.nanmedian(metrics)}\n\tStd Dev:{np.nanstd(metrics)}\n\tMin:{np.nanmin(metrics)}\n\tMax:{np.nanmax(metrics)}"
        return stats

    def _save(self, data, file):
        file = file + str(datetime.datetime.now()).replace(' ','-') + '.pkl'
        with open(file, 'wb') as f:
            pickle.dump(data, f)