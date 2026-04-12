"""
Stage 5: Evaluation and validation module.

Provides comprehensive evaluation including:
- Disentanglement metrics
- Correlation analysis between bio-features and musical parameters
- Permutation tests for significance
- Classifier-based information transfer verification
- Human evaluation survey generation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json
from pathlib import Path


class StatisticalValidator:
    """Statistical validation of conditioning effectiveness."""
    
    def __init__(self, significance_level: float = 0.05, n_permutations: int = 1000):
        self.significance_level = significance_level
        self.n_permutations = n_permutations
    
    def compute_disentanglement_gap(self, 
                                   conditioned_scores: np.ndarray,
                                   unconditioned_scores: np.ndarray) -> Dict:
        """
        Compute disentanglement gap between conditioned and unconditioned generation.
        
        Args:
            conditioned_scores: Performance scores from conditioned model
            unconditioned_scores: Performance scores from unconditioned baseline
            
        Returns:
            Dictionary with gap statistics and significance test results
        """
        # Compute mean gap
        mean_gap = np.mean(conditioned_scores) - np.mean(unconditioned_scores)
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(conditioned_scores, unconditioned_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(conditioned_scores)**2 + np.std(unconditioned_scores)**2) / 2)
        cohens_d = mean_gap / pooled_std if pooled_std > 0 else 0
        
        return {
            'mean_gap': float(mean_gap),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < self.significance_level),
            'cohens_d': float(cohens_d),
            'conditioned_mean': float(np.mean(conditioned_scores)),
            'unconditioned_mean': float(np.mean(unconditioned_scores)),
            'conditioned_std': float(np.std(conditioned_scores)),
            'unconditioned_std': float(np.std(unconditioned_scores))
        }
    
    def permutation_test(self, 
                        original_stat: float,
                        data1: np.ndarray,
                        data2: np.ndarray) -> Dict:
        """
        Perform permutation test for significance.
        
        Args:
            original_stat: Original test statistic
            data1: First dataset
            data2: Second dataset
            
        Returns:
            Permutation test results
        """
        combined = np.concatenate([data1, data2])
        n1 = len(data1)
        
        permuted_stats = []
        for _ in range(self.n_permutations):
            np.random.shuffle(combined)
            perm_stat = np.mean(combined[:n1]) - np.mean(combined[n1:])
            permuted_stats.append(perm_stat)
        
        # Two-tailed p-value
        extreme_count = sum(abs(s) >= abs(original_stat) for s in permuted_stats)
        p_value = extreme_count / self.n_permutations
        
        return {
            'original_statistic': float(original_stat),
            'permutation_p_value': float(p_value),
            'significant': bool(p_value < self.significance_level),
            'n_permutations': self.n_permutations
        }
    
    def test_correlation_significance(self, 
                                     x: np.ndarray, 
                                     y: np.ndarray,
                                     method: str = 'pearson') -> Dict:
        """
        Test correlation significance with noise robustness check.
        
        Args:
            x: First variable
            y: Second variable
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation analysis results
        """
        # Compute correlation
        if method == 'pearson':
            corr, p_value = stats.pearsonr(x, y)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(x, y)
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(x, y)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Confidence interval via bootstrap
        n_bootstrap = 1000
        bootstrap_corrs = []
        n = len(x)
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            if method == 'pearson':
                boot_corr, _ = stats.pearsonr(x[indices], y[indices])
            elif method == 'spearman':
                boot_corr, _ = stats.spearmanr(x[indices], y[indices])
            else:
                boot_corr, _ = stats.kendalltau(x[indices], y[indices])
            bootstrap_corrs.append(boot_corr)
        
        ci_lower = np.percentile(bootstrap_corrs, 2.5)
        ci_upper = np.percentile(bootstrap_corrs, 97.5)
        
        # Noise robustness test
        noisy_corrs = []
        for noise_level in [0.01, 0.05, 0.1]:
            noisy_x = x + np.random.normal(0, noise_level * np.std(x), len(x))
            if method == 'pearson':
                noisy_corr, _ = stats.pearsonr(noisy_x, y)
            else:
                noisy_corr, _ = stats.spearmanr(noisy_x, y)
            noisy_corrs.append({
                'noise_level': noise_level,
                'correlation': float(noisy_corr),
                'robust': abs(noisy_corr - corr) < 0.1
            })
        
        return {
            'correlation': float(corr),
            'p_value': float(p_value),
            'significant': bool(p_value < self.significance_level),
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper),
            'method': method,
            'noise_robustness': noisy_corrs
        }
    
    def shuffle_test(self, 
                    bio_vectors: np.ndarray,
                    musical_features: np.ndarray,
                    metric_fn,
                    n_shuffles: int = 100) -> Dict:
        """
        Test if relationship survives feature shuffling.
        
        Args:
            bio_vectors: Bio-feature vectors
            musical_features: Extracted musical features
            metric_fn: Function to compute metric between bio and music features
            n_shuffles: Number of shuffle iterations
            
        Returns:
            Shuffle test results
        """
        # Original metric
        original_metric = metric_fn(bio_vectors, musical_features)
        
        # Shuffled metrics
        shuffled_metrics = []
        for _ in range(n_shuffles):
            shuffled_bio = np.random.permutation(bio_vectors)
            shuffled_metric = metric_fn(shuffled_bio, musical_features)
            shuffled_metrics.append(shuffled_metric)
        
        # Compare
        shuffled_mean = np.mean(shuffled_metrics)
        shuffled_std = np.std(shuffled_metrics)
        
        z_score = (original_metric - shuffled_mean) / shuffled_std if shuffled_std > 0 else 0
        p_value = sum(m >= original_metric for m in shuffled_metrics) / n_shuffles
        
        return {
            'original_metric': float(original_metric),
            'shuffled_mean': float(shuffled_mean),
            'shuffled_std': float(shuffled_std),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'significant': bool(p_value < self.significance_level),
            'metric_retained': bool(original_metric > shuffled_mean + 2 * shuffled_std)
        }


class InformationTransferClassifier:
    """Classifier to verify information transfer from bio-vectors to music."""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.cluster_labels = None
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def cluster_bio_vectors(self, bio_vectors: np.ndarray) -> np.ndarray:
        """Cluster bio-vectors for classification task."""
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(bio_vectors)
        return self.cluster_labels
    
    def extract_musical_features(self, generated_sequences: torch.Tensor) -> np.ndarray:
        """
        Extract statistical features from generated sequences.
        
        Features include:
        - Note density
        - Pitch statistics
        - Rhythm patterns
        - Velocity distribution
        """
        sequences = generated_sequences.cpu().numpy()
        features = []
        
        for seq in sequences:
            # Simple feature extraction
            note_on_count = sum(1 for t in seq if 2 <= t < 103)  # NOTE_ON tokens
            pitch_values = [t - 2 for t in seq if 2 <= t < 103]
            velocity_values = [t - 178 for t in seq if 178 <= t < 306]  # VEL tokens
            
            feat = np.array([
                note_on_count / len(seq) if len(seq) > 0 else 0,  # Density
                np.mean(pitch_values) if pitch_values else 0,  # Mean pitch
                np.std(pitch_values) if len(pitch_values) > 1 else 0,  # Pitch variance
                np.mean(velocity_values) if velocity_values else 0,  # Mean velocity
                len(set(seq)) / 512,  # Vocabulary diversity
            ])
            features.append(feat)
        
        return np.array(features)
    
    def train_and_evaluate(self, 
                          musical_features: np.ndarray,
                          bio_cluster_labels: np.ndarray,
                          n_folds: int = 5) -> Dict:
        """
        Train classifier to predict bio-clusters from musical features.
        
        Args:
            musical_features: Extracted musical features
            bio_cluster_labels: Cluster labels from bio-vectors
            n_folds: Cross-validation folds
            
        Returns:
            Classification results
        """
        # Cross-validated accuracy
        cv_scores = cross_val_score(self.classifier, musical_features, 
                                    bio_cluster_labels, cv=n_folds)
        
        # Train final model
        self.classifier.fit(musical_features, bio_cluster_labels)
        predictions = self.classifier.predict(musical_features)
        train_accuracy = accuracy_score(bio_cluster_labels, predictions)
        
        return {
            'cv_mean_accuracy': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'train_accuracy': float(train_accuracy),
            'n_classes': self.n_clusters,
            'above_chance': bool(cv_scores.mean() > 1.0 / self.n_clusters),
            'classification_report': classification_report(
                bio_cluster_labels, predictions, output_dict=True
            )
        }
    
    def permutation_importance_test(self,
                                   musical_features: np.ndarray,
                                   bio_cluster_labels: np.ndarray,
                                   n_permutations: int = 100) -> Dict:
        """
        Permutation test for classifier accuracy significance.
        
        Args:
            musical_features: Musical features
            bio_cluster_labels: True labels
            n_permutations: Number of permutations
            
        Returns:
            Permutation test results
        """
        # Original accuracy
        self.classifier.fit(musical_features, bio_cluster_labels)
        original_acc = self.classifier.score(musical_features, bio_cluster_labels)
        
        # Permutation accuracies
        perm_accuracies = []
        for _ in range(n_permutations):
            shuffled_labels = np.random.permutation(bio_cluster_labels)
            self.classifier.fit(musical_features, shuffled_labels)
            perm_acc = self.classifier.score(musical_features, shuffled_labels)
            perm_accuracies.append(perm_acc)
        
        # P-value
        p_value = sum(acc >= original_acc for acc in perm_accuracies) / n_permutations
        
        return {
            'original_accuracy': float(original_acc),
            'permutation_p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'chance_level': float(1.0 / self.n_clusters)
        }


class HumanEvaluationSurvey:
    """Generate and analyze human evaluation surveys."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_survey_html(self, 
                            midi_files: Dict[str, List[str]],
                            n_trials: int = 20,
                            seed: int = 42) -> str:
        """
        Generate HTML survey for blind human evaluation.
        
        Args:
            midi_files: Dictionary mapping condition to list of file paths
            n_trials: Number of evaluation trials
            seed: Random seed
            
        Returns:
            Path to generated HTML file
        """
        np.random.seed(seed)
        
        # Prepare trial data with randomization
        conditions = list(midi_files.keys())
        trials = []
        
        for i in range(n_trials):
            # Randomly select condition and file
            condition = np.random.choice(conditions)
            file_idx = np.random.randint(0, len(midi_files[condition]))
            
            trials.append({
                'trial_id': i + 1,
                'condition': condition,
                'file': midi_files[condition][file_idx],
                'random_order': np.random.permutation(len(conditions))
            })
        
        # Add attention checks (every 5th trial)
        attention_check_indices = [4, 9, 14, 19]
        for idx in attention_check_indices:
            if idx < len(trials):
                trials[idx]['is_attention_check'] = True
        
        # Generate HTML
        html_content = self._render_survey_template(trials, conditions)
        
        html_path = self.output_dir / 'human_evaluation_survey.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return str(html_path)
    
    def _render_survey_template(self, trials: List[Dict], conditions: List[str]) -> str:
        """Render survey HTML template."""
        trials_json = json.dumps(trials)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bio-Music Human Evaluation Survey</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .trial {{ background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .rating {{ margin: 10px 0; }}
        .rating label {{ display: inline-block; margin: 0 10px; }}
        button {{ background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }}
        button:hover {{ background: #45a049; }}
        .attention-check {{ background: #fff3cd; border: 2px solid #ffc107; }}
        #results {{ display: none; }}
    </style>
</head>
<body>
    <h1>Bio-Music Generation Evaluation</h1>
    <p>Please listen to each audio sample and rate it on the following dimensions:</p>
    <ul>
        <li>Musicality (1-5): How musical/coherent does it sound?</li>
        <li>Structure (1-5): How well-structured is the piece?</li>
        <li>Variety (1-5): How varied/interesting is the content?</li>
    </ul>
    
    <div id="survey"></div>
    <button onclick="submitSurvey()">Submit Evaluation</button>
    
    <div id="results">
        <h2>Thank you for your participation!</h2>
        <p>Your responses have been recorded.</p>
    </div>
    
    <script>
        const trials = {trials_json};
        const conditions = {json.dumps(conditions)};
        
        function renderSurvey() {{
            const container = document.getElementById('survey');
            let html = '';
            
            trials.forEach((trial, idx) => {{
                const isAttentionCheck = trial.is_attention_check || false;
                const checkClass = isAttentionCheck ? 'attention-check' : '';
                
                html += `<div class="trial ${{checkClass}}" id="trial-${{idx}}">`;
                html += `<h3>Trial ${{trial.trial_id}}</h3>`;
                html += `<audio controls><source src="${{trial.file}}" type="audio/midi"></audio>`;
                
                if (isAttentionCheck) {{
                    html += `<p><strong>Attention Check:</strong> Please select "3" for all ratings in this trial.</p>`;
                }}
                
                html += `<div class="rating">`;
                html += `<label>Musicality: `;
                html += `<select id="musicality-${{idx}}">`;
                for (let i = 1; i <= 5; i++) html += `<option value="${{i}}">${{i}}</option>`;
                html += `</select></label>`;
                
                html += `<label>Structure: `;
                html += `<select id="structure-${{idx}}">`;
                for (let i = 1; i <= 5; i++) html += `<option value="${{i}}">${{i}}</option>`;
                html += `</select></label>`;
                
                html += `<label>Variety: `;
                html += `<select id="variety-${{idx}}">`;
                for (let i = 1; i <= 5; i++) html += `<option value="${{i}}">${{i}}</option>`;
                html += `</select></label>`;
                html += `</div>`;
                html += `</div>`;
            }});
            
            container.innerHTML = html;
        }}
        
        function submitSurvey() {{
            const responses = trials.map((trial, idx) => ({{
                trial_id: trial.trial_id,
                condition: trial.condition,
                musicality: parseInt(document.getElementById(`musicality-${{idx}}`).value),
                structure: parseInt(document.getElementById(`structure-${{idx}}`).value),
                variety: parseInt(document.getElementById(`variety-${{idx}}`).value),
                is_attention_check: trial.is_attention_check || false
            }}));
            
            // Save responses
            localStorage.setItem('surveyResponses', JSON.stringify(responses));
            
            // Show results
            document.getElementById('survey').style.display = 'none';
            document.getElementById('results').style.display = 'block';
            
            console.log('Survey responses:', responses);
        }}
        
        renderSurvey();
    </script>
</body>
</html>"""
        return html
    
    def compute_reliability(self, responses: List[Dict]) -> Dict:
        """
        Compute inter-rater reliability metrics.
        
        Args:
            responses: List of response dictionaries
            
        Returns:
            Reliability metrics
        """
        # Extract ratings
        musicality = [r['musicality'] for r in responses if not r.get('is_attention_check')]
        structure = [r['structure'] for r in responses if not r.get('is_attention_check')]
        variety = [r['variety'] for r in responses if not r.get('is_attention_check')]
        
        # Cronbach's alpha approximation
        def cronbach_alpha(items):
            n_items = len(items)
            if n_items < 2:
                return 0.0
            total_var = np.var(items)
            item_vars = [np.var(item) for item in items]
            alpha = (n_items / (n_items - 1)) * (1 - sum(item_vars) / total_var)
            return max(0, min(1, alpha))
        
        # Attention check performance
        attention_responses = [r for r in responses if r.get('is_attention_check')]
        attention_correct = sum(
            1 for r in attention_responses 
            if r['musicality'] == 3 and r['structure'] == 3 and r['variety'] == 3
        )
        attention_rate = attention_correct / len(attention_responses) if attention_responses else 1.0
        
        return {
            'cronbach_alpha_musicality': float(cronbach_alpha(musicality)),
            'cronbach_alpha_structure': float(cronbach_alpha(structure)),
            'cronbach_alpha_variety': float(cronbach_alpha(variety)),
            'attention_check_pass_rate': float(attention_rate),
            'n_valid_responses': len(responses) - len(attention_responses),
            'n_total_responses': len(responses)
        }


def run_comprehensive_evaluation(config: Dict,
                                 test_data: List,
                                 generated_samples: Dict,
                                 bio_vectors: np.ndarray,
                                 output_dir: str) -> Dict:
    """
    Run comprehensive evaluation suite.
    
    Args:
        config: Configuration dictionary
        test_data: Test dataset samples
        generated_samples: Dictionary of generated samples by condition
        bio_vectors: Bio-feature vectors
        output_dir: Output directory for results
        
    Returns:
        Complete evaluation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validator = StatisticalValidator(
        significance_level=config.get('significance_level', 0.05),
        n_permutations=config.get('n_permutations', 1000)
    )
    
    results = {}
    
    # 1. Disentanglement analysis
    if 'conditioned' in generated_samples and 'unconditional' in generated_samples:
        cond_quality = np.array([compute_sequence_quality(s) for s in generated_samples['conditioned']])
        uncond_quality = np.array([compute_sequence_quality(s) for s in generated_samples['unconditional']])
        
        results['disentanglement'] = validator.compute_disentanglement_gap(cond_quality, uncond_quality)
    
    # 2. Information transfer
    it_classifier = InformationTransferClassifier(n_clusters=5)
    cluster_labels = it_classifier.cluster_bio_vectors(bio_vectors)
    
    if 'conditioned' in generated_samples:
        from collections import namedtuple
        Sample = namedtuple('Sample', ['cpu'])
        mock_generated = torch.stack([torch.tensor(s) for s in generated_samples['conditioned'][:len(bio_vectors)]])
        
        musical_features = it_classifier.extract_musical_features(mock_generated)
        results['information_transfer'] = it_classifier.train_and_evaluate(musical_features, cluster_labels)
        results['permutation_test'] = it_classifier.permutation_importance_test(
            musical_features, cluster_labels
        )
    
    # 3. Save results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def compute_sequence_quality(sequence: np.ndarray) -> float:
    """Compute simple quality metric for a sequence."""
    # Placeholder: implement based on specific metrics
    return float(np.mean(sequence))
