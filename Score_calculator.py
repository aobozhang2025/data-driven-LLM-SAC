import numpy as np
from collections import Counter
import math
import warnings
from typing import List, Union, Dict, Tuple

import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from rouge_score import rouge_scorer
from bert_score import score as bert_score

class Score_calculator:
    def __init__(self):
        """
        Initialize the evaluator.
        """
        # Download required NLTK data
        try:
            nltk.download('punkt_tab')
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
        except:
            pass

    @staticmethod
    def calculate_bleu(candidate: str, references: Union[List[str], str],
                       max_n: int = 4, weights: List[float] = None) -> float:
        """
        Calculate BLEU score between candidate text and references.

        Args:
            candidate: The generated text to evaluate
            references: Reference text(s) to compare against. Can be a single string or list of strings.
            max_n: Maximum n-gram order to consider (default: 4)
            weights: Weights for n-gram orders (default: uniform weights)

        Returns:
            BLEU score between 0 and 1
        """
        if references is None:
            raise ValueError("Reference texts not provided")

        references = [references] if isinstance(references, str) else references

        if weights is None:
            weights = [1. / max_n] * max_n  # Uniform weights

        # Tokenize
        candidate_tokens = word_tokenize(candidate.lower())
        reference_tokens_list = [word_tokenize(ref.lower()) for ref in references]

        # Calculate modified precision for each n-gram order
        precisions = []
        for n in range(1, max_n + 1):
            candidate_ngrams = list(ngrams(candidate_tokens, n))
            if not candidate_ngrams:
                return 0.0

            # Count maximum possible matches for each n-gram in references
            max_matches = 0
            for ref_tokens in reference_tokens_list:
                ref_ngrams = list(ngrams(ref_tokens, n))
                matches = 0
                counts = Counter(candidate_ngrams)
                for gram in counts:
                    if gram in ref_ngrams:
                        matches += min(counts[gram], ref_ngrams.count(gram))
                max_matches = max(max_matches, matches)

            precisions.append(max_matches / len(candidate_ngrams))

        # Calculate brevity penalty
        candidate_len = len(candidate_tokens)
        closest_ref_len = min(len(ref_tokens) for ref_tokens in reference_tokens_list)
        bp = 1.0 if candidate_len > closest_ref_len else math.exp(1 - closest_ref_len / candidate_len)

        # Calculate geometric mean of modified precisions
        geometric_mean = math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions) if p > 0))

        return bp * geometric_mean

    @staticmethod
    def calculate_rouge(candidate: str, references: Union[List[str], str],
                        rouge_types: List[str] = None) -> Dict[str, float]:
        """
        Calculate ROUGE scores between candidate text and references.

        Args:
            candidate: The generated text to evaluate
            references: Reference text(s) to compare against. Can be a single string or list of strings.
            rouge_types: List of ROUGE types to calculate (e.g., ['rouge1', 'rouge2', 'rougeL'])
                        Default: ['rouge1', 'rouge2', 'rougeL']

        Returns:
            Dictionary of ROUGE scores
        """
        if references is None:
            raise ValueError("Reference texts not provided")

        references = [references] if isinstance(references, str) else references

        if rouge_types is None:
            rouge_types = ['rouge1', 'rouge2', 'rougeL']

        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

        # Calculate ROUGE against each reference and take the max
        scores = {}
        for ref in references:
            current_scores = scorer.score(ref, candidate)
            for key in current_scores:
                if key not in scores or current_scores[key].fmeasure > scores[key].fmeasure:
                    scores[key] = current_scores[key]

        return scores

    @staticmethod
    def calculate_meteor(candidate: str, references: Union[List[str], str]) -> float:
        """
        Calculate METEOR score between candidate text and references.

        Args:
            candidate: The generated text to evaluate
            references: Reference text(s) to compare against. Can be a single string or list of strings.

        Returns:
            METEOR score between 0 and 1
        """
        if references is None:
            raise ValueError("Reference texts not provided")

        references = [references] if isinstance(references, str) else references

        # Tokenize
        candidate_tokens = word_tokenize(candidate.lower())
        reference_tokens_list = [word_tokenize(ref.lower()) for ref in references]

        # Calculate METEOR against each reference and take the max
        max_score = 0.0
        for ref_tokens in reference_tokens_list:
            current_score = meteor_score([ref_tokens], candidate_tokens)
            if current_score > max_score:
                max_score = current_score

        return max_score

    @staticmethod
    def calculate_bertscore(candidate: str, references: Union[List[str], str],
                            lang: str = 'en') -> Dict[str, float]:
        """
        Calculate BERTScore between candidate text and references.

        Args:
            candidate: The generated text to evaluate
            references: Reference text(s) to compare against. Can be a single string or list of strings.
            lang: Language code (default: 'en')

        Returns:
            Dictionary containing precision, recall, and F1 scores
        """
        if references is None:
            raise ValueError("Reference texts not provided")

        references = [references] if isinstance(references, str) else references

        # Handle single candidate case
        candidates = [candidate] * len(references) if len(references) > 1 else [candidate]

        P, R, F1 = bert_score(candidates, references, lang=lang)

        return {
            'bert_precision': P.mean().item(),
            'bert_recall': R.mean().item(),
            'bert_f1': F1.mean().item()
        }

    @staticmethod
    def calculate_perplexity(candidate: str, model=None, tokenizer=None) -> float:
        """
        Calculate perplexity of the candidate text.

        Args:
            candidate: The generated text to evaluate
            model: Optional pre-loaded language model for perplexity calculation
            tokenizer: Optional tokenizer for the model

        Returns:
            Perplexity score (lower is better)
        """
        # This is a placeholder implementation
        # For actual implementation, you would need a language model like GPT-2
        warnings.warn("Perplexity calculation requires a language model. This is a placeholder implementation.")

        # Simple approximation (not accurate without a proper language model)
        words = word_tokenize(candidate.lower())
        if not words:
            return 0.0

        unique_words = set(words)
        vocabulary_size = max(len(unique_words), 1)
        return math.exp(-sum(math.log(1.0 / vocabulary_size) for _ in words) / len(words))

    @staticmethod
    def evaluate_all(candidate: str, references: Union[List[str], str],
                     include_bertscore: bool = True) -> Dict[str, Union[float, Dict]]:
        """
        Calculate all available metrics for the candidate text.

        Args:
            candidate: The generated text to evaluate
            references: Reference text(s) to compare against. Can be a single string or list of strings.
            include_bertscore: Whether to include BERTScore (can be computationally expensive)

        Returns:
            Dictionary containing all calculated metrics
        """
        results = {}

        # Calculate metrics that don't require references
        results['perplexity'] = Score_calculator.calculate_perplexity(candidate)

        # Calculate metrics that require references
        if references is not None:
            results['bleu'] = Score_calculator.calculate_bleu(candidate, references)
            results['rouge'] = Score_calculator.calculate_rouge(candidate, references)
            results['meteor'] = Score_calculator.calculate_meteor(candidate, references)

            if include_bertscore:
                try:
                    results.update(Score_calculator.calculate_bertscore(candidate, references))
                except:
                    warnings.warn("BERTScore calculation failed")

        return results


# Example usage
if __name__ == "__main__":
    references = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over the sleepy dog"
    ]
    candidate = "The fast brown fox jumps over the tired dog"

    evaluator = Score_calculator()

    # Calculate individual metrics
    print("BLEU score:", evaluator.calculate_bleu(candidate, references)) # a float
    print("ROUGE scores:", evaluator.calculate_rouge(candidate, references)) # {'rouge1': Score(precision=0.7777777777777778, recall=0.7777777777777778, fmeasure=0.7777777777777778), 'rouge2': Score(precision=0.5, recall=0.5, fmeasure=0.5), 'rougeL': Score(precision=0.7777777777777778, recall=0.7777777777777778, fmeasure=0.7777777777777778)}
    print("METEOR score:", evaluator.calculate_meteor(candidate, references)) # a float
    print("BERTScore:", evaluator.calculate_bertscore(candidate, references)) # {'bert_precision': 0.98630291223526, 'bert_recall': 0.98630291223526, 'bert_f1': 0.98630291223526}
    print("Perplexity:", evaluator.calculate_perplexity(candidate)) # float

